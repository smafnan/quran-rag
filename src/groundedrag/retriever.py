"""Retrieval over the corpus with a relevance gate.

Three matching layers, combined into one hybrid search:

  * **keyword** — suffix-stripping morphological matching over the translation
    text, and token-level matching (diacritic-insensitive, clitic-aware) over
    the Arabic text for Arabic-script queries. This is what makes "every
    occurrence" mode exhaustive: a verse that literally contains the topic word
    is always found.
  * **tfidf** — TF-IDF (1-2 grams) + cosine similarity, offline, no downloads.
  * **semantic** — optional embedding index (see scripts/build_embeddings.py)
    plus a query-embedding callable; catches conceptually-related verses that
    share no vocabulary with the query.

The relevance gate is what makes the system *honest*: if nothing in the book
clears any layer's threshold, search returns nothing — so the answerer can say
"this isn't addressed in the source" instead of forcing a weak passage.

Two modes:
  * ``mode="top"`` (default) — the best ``top_k`` hits, as before.
  * ``mode="all"``  — every hit that clears a gate, ranked; for
    "show me every occurrence of this topic in the whole book".
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize as sk_normalize

from .corpus import Passage

# semantic cosine above this counts as a relevant conceptual match on its own.
# Calibrated against nemotron-3-embed-1b on this corpus: on-topic queries score
# 0.45-0.56 at the top with related verses in the 0.35-0.45 band, while
# off-topic noise (e.g. "kubernetes") peaks around 0.22.
SEMANTIC_THRESHOLD_DEFAULT = 0.35
# keyword hits ranked below full-strength semantic hits get this ranking boost
KEYWORD_BOOST = 0.25
# an Arabic-script query can never be corroborated by TF-IDF (English-only
# vocabulary) or by the semantic index (embedded from English text), so this
# fraction of its terms matching is treated as self-corroborating.
ARABIC_SELF_CORROBORATE = 0.5
# Arabic terms shorter than this are particles/clitics that match nearly
# everything; requiring 3+ characters keeps "من"/"و" from flooding results.
ARABIC_MIN_TERM_LEN = 3

_TOKEN_RE = re.compile(r"[a-z']+")
_ARABIC_CHAR_RE = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")
# combining harakat (U+064B-U+065F covers fatha..maddah/hamza marks), Quranic
# annotation marks, and tatweel. NOTE: U+0670 (dagger alef) is deliberately NOT
# here — it is a *letter* in disguise and gets mapped to plain alef below.
_ARABIC_MARKS_RE = re.compile(r"[ً-ٟـۖ-ࣰۭ-ࣿ]")

# function words that would otherwise dominate Arabic keyword matching. Kept
# deliberately small: only unambiguous particles, never plausible search terms
# (notably "الله" and "علي" are absent — both are things people search for).
_ARABIC_STOPWORDS = frozenset({
    "الي", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "الذين", "كان",
    "اذا", "لكن", "حتي", "هؤلاء", "اولئك", "عليه", "اليه", "ولا", "وما",
})

# English suffixes stripped (longest first) to compare morphological stems.
_SUFFIXES = (
    "ations", "ation", "ness", "ment", "ance", "ence", "ible", "able",
    "ing", "ers", "est", "ity", "ies", "ied", "ive", "ous", "ful",
    "ant", "ent", "ed", "er", "es", "ly", "s",
)
_VOWELS = frozenset("aeiou")


def _stem(word: str) -> str:
    """Light suffix-stripping stemmer.

    Enough to unify patience/patient, mercy/merciful, pray/prayer,
    forgive/forgiveness, believe/believer/believing — while keeping
    accidental prefix twins apart (mercy/merchant, angel/anger,
    charity/chariot, light/lightning, sign/significant, worship/worse).
    """
    w = word
    # -ly before the y->i rule, which would otherwise mask it (patiently)
    if len(w) > 5 and w.endswith("ly"):
        w = w[:-2]
    # y -> i only after a consonant (mercy -> merci, but pray stays pray)
    if len(w) > 3 and w.endswith("y") and w[-2] not in _VOWELS:
        w = w[:-1] + "i"
    # iterate so stacked suffixes reduce to the same stem as the shorter form
    # (forgiveness -> forgive -> forg, matching forgive -> forg)
    for _ in range(2):
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)]
                break
        else:
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    # undo consonant doubling from suffixation (worshipp -> worship)
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz" and w[-1] not in _VOWELS:
        w = w[:-1]
    return w


def _normalize_arabic(text: str, *, dagger_as_alef: bool = True) -> str:
    # Dagger alef marks a long-a that Uthmani script writes as a diacritic.
    # Standard spelling sometimes writes it as a full alef (ٱلْعَٰلَمِينَ ->
    # العالمين) and sometimes omits it (ٱلرَّحْمَٰنِ -> الرحمن), so callers
    # index/query BOTH readings via _arabic_forms rather than picking one.
    text = text.replace("ٰ", "ا" if dagger_as_alef else "")
    text = _ARABIC_MARKS_RE.sub("", text)
    return (text.replace("ٱ", "ا")   # alef wasla -> alef
                .replace("أ", "ا")   # alef hamza above
                .replace("إ", "ا")   # alef hamza below
                .replace("آ", "ا")   # alef madda
                .replace("ؤ", "و")   # waw hamza
                .replace("ئ", "ي")   # ya hamza
                .replace("ى", "ي")   # alef maqsura -> ya
                .replace("ة", "ه"))  # ta marbuta -> ha


def _arabic_forms(text: str) -> set[str]:
    """Both dagger-alef readings of a string (identical when it has none)."""
    return {_normalize_arabic(text, dagger_as_alef=True),
            _normalize_arabic(text, dagger_as_alef=False)}


def _arabic_variants(token: str) -> set[str]:
    """A token plus the forms left after peeling proclitics (و/ف/ب/ك/ل/ال),
    so a query for رحمن matches الرحمن and الله matches والله."""
    out = {token}
    t = token
    for _ in range(3):
        before = t
        if len(t) > 4 and t.startswith("يا"):   # vocative: ياموسي -> موسي
            t = t[2:]
            out.add(t)
        if len(t) > 3 and t[0] in "وف":
            t = t[1:]
            out.add(t)
        if len(t) > 3 and t[0] in "بكل":
            t = t[1:]
            out.add(t)
        if len(t) > 3 and t.startswith("ال"):
            t = t[2:]
            out.add(t)
        if t == before:
            break
    # a word can carry more than one dagger alef, and the two readings flip
    # together (يَٰمُوسَىٰ -> ياموسيا / يموسي, neither of which is موسي), so
    # also offer each form without a trailing alef
    for form in list(out):
        if len(form) > 3 and form.endswith("ا"):
            out.add(form[:-1])
    return out


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    passage: Passage
    score: float
    matched: list[str] = field(default_factory=list)  # which layers matched


class Retriever:
    def __init__(self, passages: list[Passage],
                 embeddings_path: str | Path | None = None,
                 embed_query: Callable[[str], list[float]] | None = None,
                 semantic_threshold: float = SEMANTIC_THRESHOLD_DEFAULT) -> None:
        self.passages = passages
        self.embed_query = embed_query
        self.semantic_threshold = semantic_threshold

        self._vec = TfidfVectorizer(lowercase=True, stop_words="english",
                                    ngram_range=(1, 2), sublinear_tf=True)
        # kept sparse on purpose: densifying this is ~0.9 GB on a full corpus
        # for ~100k non-zeros, and the cosine math is identical either way.
        self._matrix = sk_normalize(self._vec.fit_transform([p.text for p in passages]))

        # per-passage stem sets for the keyword layer
        self._stem_sets = [{_stem(t) for t in _tokens(p.text)} for p in passages]
        self._arabic_tokens: list[set[str]] = []
        for p in passages:
            variants: set[str] = set()
            if p.arabic:
                for form in _arabic_forms(p.arabic):
                    for tok in form.split():
                        variants |= _arabic_variants(tok)
            self._arabic_tokens.append(variants)

        # optional semantic index (refs must align 1:1 with the corpus)
        self._sem_matrix: np.ndarray | None = None
        if embeddings_path and Path(embeddings_path).exists():
            data = np.load(embeddings_path, allow_pickle=False)
            refs = [str(r) for r in data["refs"]]
            corpus_refs = [p.ref for p in passages]
            if refs == corpus_refs:
                # kept at the stored precision (float16) rather than widened to
                # float32: measured bit-identical scores on this corpus for half
                # the resident memory (25 MB vs 50 MB), at ~1.7 ms extra per query
                self._sem_matrix = data["matrix"]
            else:
                print(f"warning: {embeddings_path} does not align with the corpus "
                      f"({self._describe_ref_mismatch(refs, corpus_refs)}) — semantic "
                      f"layer disabled; rebuild with scripts/build_embeddings.py",
                      file=sys.stderr)

    @staticmethod
    def _describe_ref_mismatch(indexed: list[str], corpus: list[str]) -> str:
        if len(indexed) != len(corpus):
            return f"{len(indexed)} indexed vs {len(corpus)} passages"
        for i, (a, b) in enumerate(zip(indexed, corpus)):
            if a != b:
                return (f"same length ({len(corpus)}) but differs at index {i}: "
                        f"indexed '{a}' vs corpus '{b}'")
        return "contents differ"

    # ------------------------------------------------------------------ layers

    def _tfidf_scores(self, query: str) -> np.ndarray:
        q = self._vec.transform([query])
        if q.nnz == 0:
            return np.zeros(len(self.passages), dtype=np.float32)
        q = sk_normalize(q)
        return np.asarray((self._matrix @ q.T).todense()).ravel().astype(np.float32)

    def _semantic_scores(self, query: str) -> np.ndarray | None:
        if self._sem_matrix is None or self.embed_query is None:
            return None
        try:
            q = np.asarray(self.embed_query(query), dtype=np.float32).squeeze()
            # a provider swap / malformed response must degrade to lexical-only,
            # not blow up every search with a matmul shape error
            if q.ndim != 1 or q.shape[0] != self._sem_matrix.shape[1]:
                return None
            n = float(np.linalg.norm(q))
            if n == 0:
                return None
            # float16 matrix @ float32 vector promotes to float32 for the result
            return np.asarray(self._sem_matrix @ (q / n), dtype=np.float32)
        except Exception:
            return None

    def _query_terms(self, query: str) -> tuple[list[str], list[frozenset[str]]]:
        """(english stems, arabic terms as alternative-spelling sets)."""
        stems = [_stem(w) for w in _tokens(query)
                 if len(w) >= 3 and w not in ENGLISH_STOP_WORDS]
        arabic: list[frozenset[str]] = []
        if _ARABIC_CHAR_RE.search(query):
            # position i of each reading is the same source word, so zipping the
            # readings groups a word's alternative spellings together
            readings = [form.split() for form in
                        (_normalize_arabic(query, dagger_as_alef=True),
                         _normalize_arabic(query, dagger_as_alef=False))]
            if len({len(r) for r in readings}) == 1:
                for alts in zip(*readings):
                    primary = alts[0]
                    if (len(primary) >= ARABIC_MIN_TERM_LEN
                            and primary not in _ARABIC_STOPWORDS):
                        arabic.append(frozenset(a for a in alts if a))
        return stems, arabic

    def _keyword_fractions(self, stems: list[str],
                           arabic: list[frozenset[str]]) -> np.ndarray:
        """Per passage: fraction of the query's content terms present in it."""
        n_terms = len(stems) + len(arabic)
        if n_terms == 0:
            return np.zeros(len(self.passages), dtype=np.float32)

        fractions = np.zeros(len(self.passages), dtype=np.float32)
        for i in range(len(self.passages)):
            hits = 0
            if stems:
                passage_stems = self._stem_sets[i]
                hits += sum(1 for s in stems if s in passage_stems)
            if arabic:
                passage_tokens = self._arabic_tokens[i]
                hits += sum(1 for alts in arabic if alts & passage_tokens)
            if hits:
                fractions[i] = hits / n_terms
        return fractions

    # ------------------------------------------------------------------ search

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05,
               mode: str = "top") -> list[Hit]:
        stems, arabic_terms = self._query_terms(query)
        tfidf = self._tfidf_scores(query)
        kw = self._keyword_fractions(stems, arabic_terms)
        sem = self._semantic_scores(query)

        base = tfidf.copy()
        if sem is not None:
            base = np.maximum(base, sem)
        combined = base + KEYWORD_BOOST * kw

        include = np.zeros(len(self.passages), dtype=bool)
        include |= kw >= 1.0                       # every content term occurs
        include |= tfidf >= min_score
        if sem is not None:
            include |= sem >= self.semantic_threshold
        # partial keyword + at least weak corroboration from another layer
        weak = tfidf >= (min_score / 2)
        if sem is not None:
            weak |= sem >= (self.semantic_threshold - 0.05)
        include |= (kw > 0) & weak
        # Arabic-script queries have no other layer that can corroborate them
        # (TF-IDF vocabulary and the embedding index are both English), so a
        # substantial fraction of terms matching stands on its own.
        if arabic_terms and not stems:
            include |= kw >= ARABIC_SELF_CORROBORATE

        order = np.argsort(combined)[::-1]
        hits: list[Hit] = []
        for i in order:
            if not include[i]:
                continue
            matched = []
            if kw[i] >= 1.0:
                matched.append("keyword")
            elif kw[i] > 0:
                matched.append("keyword-partial")
            if sem is not None and sem[i] >= self.semantic_threshold:
                matched.append("semantic")
            if tfidf[i] >= min_score:
                matched.append("tfidf")
            hits.append(Hit(self.passages[i], float(combined[i]), matched))
            if mode == "top" and len(hits) >= top_k:
                break
        return hits
