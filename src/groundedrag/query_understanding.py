"""Work out what the reader actually meant.

Three distinct problems, which need different treatment:

1. **Transliteration.** There is no single correct English spelling of an Arabic
   name. A reader may type Moses, Musa or Moosa; Mohammed or Muhammad; Abraham
   or Ibrahim. The corpus picks one spelling and the others find nothing — this
   one silently returns "not addressed in the Quran" for topics the Quran
   discusses at length, which is the worst failure this app can have.
   Handled by curated equivalence groups, because no algorithm knows that
   "Jesus" and "Isa" are the same person.

2. **Ordinary typos.** "paitence", "forgivness". Handled by edit distance
   against the corpus vocabulary.

3. **Spelling drift the algorithm can catch.** "Moosa" vs "Musa",
   "Dawud" vs "Dawood" — same name, different vowel choices. Handled by a
   transliteration key that ignores the parts of Arabic romanisation that vary.
   The corpus is inconsistent with *itself* here (yaqoub 16 / yaqoob 2), so this
   helps even when the reader spells it the way a scholar would.

The result is an Interpretation, never a silent rewrite: whatever the system
decided is reported back so the reader can see it and overrule it.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# Sets of terms that mean the same thing. Every member is searched when any
# member is typed, so "Noah" finds verses using either "noah" or "nuh" — this
# expands recall rather than replacing the reader's word.
#
# Only equivalences a reader would expect to be interchangeable. Deliberately
# NOT here: theological identifications that are contested, or words that merely
# relate ("mercy" is not "forgiveness").
ALIAS_GROUPS: tuple[frozenset[str], ...] = tuple(frozenset(g) for g in [
    # prophets and figures — anglicised vs transliterated
    {"musa", "moosa", "moses"},
    {"muhammad", "mohammed", "mohammad", "muhammed", "ahmad"},
    {"ibrahim", "ibraheem", "abraham"},
    {"isa", "eesa", "jesus"},
    {"maryam", "mariam", "mary"},
    {"nuh", "nooh", "noah"},
    {"yusuf", "yousuf", "yusof", "joseph"},
    {"yaqoub", "yaqoob", "yakub", "jacob", "israel"},
    {"ishaq", "isaac"},
    {"ismail", "ismaeel", "ishmael"},
    {"dawood", "dawud", "david"},
    {"sulaiman", "sulayman", "solomon"},
    {"haroun", "harun", "aaron"},
    {"yahya", "john"},
    {"zakariyya", "zakariya", "zachariah"},
    {"ayyub", "ayoub", "job"},
    {"yunus", "younus", "jonah"},
    {"lut", "lot"},
    {"idris", "enoch"},
    {"ilyas", "elijah"},
    {"alyasa", "elisha"},
    {"zulqarnayn", "zulqarnain", "dhulqarnayn"},
    {"talut", "saul"},
    {"jalut", "goliath"},
    {"firaun", "firawn", "pharaoh"},
    {"iblees", "iblis", "shaytan", "shaitan", "satan", "devil"},
    {"jibraeel", "jibreel", "jibril", "gabriel"},
    # practices and concepts
    {"salat", "salah", "namaz", "prayer"},
    {"zakat", "zakah", "almsgiving", "alms"},
    {"sawm", "siyam", "fasting", "fast"},
    {"hajj", "pilgrimage"},
    {"sabr", "patience", "steadfastness"},
    {"taqwa", "piety", "godfearing"},
    {"tawbah", "repentance"},
    {"jannah", "paradise", "heaven"},
    {"jahannam", "hellfire", "hell"},
    {"qiyamah", "qiyamat", "resurrection"},
    {"akhirah", "hereafter", "afterlife"},
    {"rizq", "provision", "sustenance"},
    {"tawheed", "tawhid", "monotheism", "oneness"},
    {"shirk", "polytheism", "idolatry"},
    {"quran", "koran", "qur'an"},
    {"kaaba", "kabah", "kaba"},
    {"masjid", "mosque"},
    {"riba", "usury", "interest"},
    {"jihad", "struggle", "striving"},
    {"rasool", "rasul", "messenger"},
    {"nabi", "prophet"},
    {"malaikah", "angels", "angel"},
    {"jinn", "jinns"},
    {"dua", "supplication"},
    {"sadaqah", "charity"},
])

_WORD_RE = re.compile(r"[a-z']+")
# digraphs first: order matters, so kh -> k does not fire before k stays k
_TRANSLIT_RULES = (
    ("ph", "f"), ("kh", "k"), ("gh", "g"), ("dh", "d"), ("th", "t"),
    ("sh", "s"), ("ch", "k"), ("ck", "k"), ("q", "k"), ("x", "ks"),
    ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ou", "u"), ("ei", "i"),
    ("ai", "i"), ("ay", "i"), ("ya", "i"), ("y", "i"), ("w", "u"),
    ("j", "g"),
)
_VOWELS = "aeiou"


def translit_key(word: str) -> str:
    """A spelling-agnostic key for romanised Arabic.

    Collapses the choices that vary between transliteration schemes — vowel
    length, which consonant digraph, doubled letters — so that Musa/Moosa,
    Dawud/Dawood and Yaqoub/Yaqoob land on the same key while genuinely
    different words stay apart.
    """
    w = word.lower().replace("'", "")
    for a, b in _TRANSLIT_RULES:
        w = w.replace(a, b)
    out = []
    for ch in w:
        if out and out[-1] == ch:       # collapse doubles (mohammed -> mohamed)
            continue
        out.append(ch)
    w = "".join(out)
    if len(w) > 2 and w.endswith("h"):  # trailing h is decorative (jannah/janna)
        w = w[:-1]
    # keep the first letter, then drop interior vowels: the consonant skeleton
    # is what survives transliteration
    if len(w) > 3:
        w = w[0] + "".join(c for c in w[1:] if c not in _VOWELS)
    return w or word.lower()


@dataclass
class Interpretation:
    """What the system decided the query means, and how sure it is."""
    original: str
    effective: str                                    # what will actually be searched
    corrections: list[tuple[str, str]] = field(default_factory=list)
    expanded: list[tuple[str, list[str]]] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    needs_confirmation: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.corrections)

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "effective": self.effective,
            "corrections": [{"from": a, "to": b} for a, b in self.corrections],
            "expanded": [{"term": t, "also_searched": v} for t, v in self.expanded],
            "suggestions": self.suggestions,
            "needs_confirmation": self.needs_confirmation,
        }


class QueryUnderstanding:
    def __init__(self, vocabulary, alias_groups=ALIAS_GROUPS,
                 min_word_len: int = 4) -> None:
        # vocabulary may be a Counter (preferred: frequency breaks ties toward
        # the word the corpus actually uses) or any iterable of words
        self.freq = dict(vocabulary) if hasattr(vocabulary, "items") else {
            w: 1 for w in vocabulary}
        self.vocab = set(self.freq)
        self.min_word_len = min_word_len

        # term -> every term meaning the same thing
        self.synonyms: dict[str, frozenset[str]] = {}
        for group in alias_groups:
            for term in group:
                self.synonyms[term] = group

        # transliteration key -> corpus words sharing it
        self._by_key: dict[str, list[str]] = {}
        for w in self.vocab:
            if len(w) >= self.min_word_len:
                self._by_key.setdefault(translit_key(w), []).append(w)
        # alias members that are not corpus words are still valid inputs
        self._known_inputs = self.vocab | set(self.synonyms)

    # ------------------------------------------------------------------ pieces

    def _score(self, word: str, cand: str) -> float:
        """How likely `cand` is what `word` meant.

        Surface similarity leads. The transliteration key is a strong hint but a
        lossy one — it discards vowels, so on ordinary English it will happily
        equate "prayerr" with "purer". Letting it only *boost* a candidate that
        already looks similar keeps its usefulness on names without letting it
        override the evidence of the actual letters.
        """
        ratio = difflib.SequenceMatcher(None, word, cand).ratio()
        score = ratio
        if translit_key(word) == translit_key(cand):
            score += 0.15
        if word[:1] == cand[:1]:            # first letter rarely mistyped
            score += 0.05
        # prefer the spelling the corpus actually favours, but only as a tiebreak
        score += min(self.freq.get(cand, 0), 200) / 200 * 0.04
        return score

    # a candidate must at least look like the word: below this, a "correction"
    # is really a different word and silently substituting it would mislead
    MIN_SIMILARITY = 0.72

    def _candidates(self, word: str) -> list[str]:
        """Plausible intended spellings, best first."""
        pool = set(self._by_key.get(translit_key(word), []))
        pool |= set(difflib.get_close_matches(word, self.vocab, n=8, cutoff=0.72))
        pool |= {t for t in self.synonyms if difflib.SequenceMatcher(None, word, t).ratio() >= 0.85}
        pool.discard(word)
        scored = [(self._score(word, c), c) for c in pool
                  if difflib.SequenceMatcher(None, word, c).ratio() >= self.MIN_SIMILARITY]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [c for _, c in scored]

    def _is_ambiguous(self, word: str, cands: list[str]) -> bool:
        """Two candidates are only genuinely competing if they are not just
        spellings of each other and neither is a clear favourite."""
        if len(cands) < 2:
            return False
        top, second = cands[0], cands[1]
        if self.synonyms.get(top) and self.synonyms.get(top) is self.synonyms.get(second):
            return False                      # same concept, no choice to make
        if translit_key(top) == translit_key(second):
            return False                      # same word, different spelling
        # only genuinely competing if neither spelling is a clear winner
        s1, s2 = self._score(word, top), self._score(word, second)
        return (s1 - s2) < 0.06

    # ------------------------------------------------------------------ public

    def analyse(self, query: str) -> Interpretation:
        words = _WORD_RE.findall(query.lower())
        corrections: list[tuple[str, str]] = []
        expanded: list[tuple[str, list[str]]] = []
        suggestions: list[str] = []
        ambiguous = False

        for w in words:
            if len(w) < self.min_word_len:
                continue
            if w in self._known_inputs:
                # a known word may still be worth widening (noah -> also nuh)
                group = self.synonyms.get(w)
                if group:
                    others = sorted(t for t in group
                                    if t != w and t in self.vocab)
                    if others:
                        expanded.append((w, others))
                continue

            cands = self._candidates(w)
            if not cands:
                continue
            if self._is_ambiguous(w, cands):
                ambiguous = True
                for c in cands[:3]:
                    alt = re.sub(rf"\b{re.escape(w)}\b", c, query, flags=re.I)
                    if alt not in suggestions:
                        suggestions.append(alt)
            else:
                corrections.append((w, cands[0]))

        effective = query
        for wrong, right in corrections:
            effective = re.sub(rf"\b{re.escape(wrong)}\b", right, effective, flags=re.I)

        # re-run expansion over the corrected words too
        for _, right in corrections:
            group = self.synonyms.get(right)
            if group:
                others = sorted(t for t in group if t != right and t in self.vocab)
                if others:
                    expanded.append((right, others))

        return Interpretation(original=query, effective=effective,
                              corrections=corrections, expanded=expanded,
                              suggestions=suggestions,
                              needs_confirmation=ambiguous and not corrections)

    def synonym_map(self) -> dict[str, frozenset[str]]:
        """Equivalences restricted to terms the corpus can actually match."""
        out: dict[str, frozenset[str]] = {}
        for term, group in self.synonyms.items():
            usable = frozenset(t for t in group if t in self.vocab)
            if usable:
                out[term] = usable
        return out
