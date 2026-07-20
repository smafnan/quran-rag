"""Tests for the grounded Quran RAG (engine + grounding guarantees)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from groundedrag import GroundedAnswerer, MockLLM, Retriever, load_corpus

DATA = Path(__file__).resolve().parents[1] / "data" / "quran_sample.jsonl"


@pytest.fixture(scope="module")
def passages():
    return load_corpus(DATA)


@pytest.fixture(scope="module")
def answerer(passages):
    return GroundedAnswerer(Retriever(passages), "the Quran")


def test_corpus_loads(passages):
    assert len(passages) >= 10
    assert all(p.ref and p.text for p in passages)


def test_retrieves_relevant_verse(answerer):
    ans = answerer.ask("hardship and ease")
    assert ans.found
    refs = [p.ref for p in ans.citations]
    assert "94:5" in refs or "94:6" in refs   # the hardship/ease verses


def test_relevance_gate_declines_off_topic(answerer):
    ans = answerer.ask("how to configure a kubernetes cluster")
    assert not ans.found
    assert "not" in ans.text.lower()           # declines instead of guessing


def test_answer_only_contains_book_text(answerer, passages):
    """In passages mode, every line of the answer must come from the corpus."""
    ans = answerer.ask("remembrance of Allah and the heart")
    assert ans.found
    corpus_texts = {p.text for p in passages}
    for cite in ans.citations:
        assert cite.text in corpus_texts        # citations are real passages
        assert cite.text in ans.text            # and appear verbatim in the answer


def test_citations_are_returned(answerer):
    ans = answerer.ask("oneness of Allah")
    assert ans.found and ans.citations
    assert all(":" in p.ref for p in ans.citations)   # verse refs like 112:1


def test_compose_mode_uses_llm_and_passages(answerer):
    # A mock that proves it only saw the retrieved passages.
    def fn(system, user):
        assert "only from the quran" in system.lower()
        return "Grounded answer citing [112:1]."
    composed = GroundedAnswerer(answerer.retriever, "the Quran", llm=MockLLM(fn))
    ans = composed.ask("oneness of Allah", compose=True)
    assert ans.found and "[112:1]" in ans.text
    assert ans.citations                          # still returns sources


def test_not_found_has_no_citations(answerer):
    ans = answerer.ask("xyzzy nonsense token")
    assert not ans.found and ans.citations == []


# ---------------------------------------------------------------- hybrid search

def test_all_mode_finds_every_keyword_occurrence(passages):
    """mode='all' must surface every passage containing the topic word."""
    r = Retriever(passages)
    hits = r.search("hardship", mode="all")
    refs = {h.passage.ref for h in hits}
    expected = {p.ref for p in passages if "hardship" in p.text.lower()}
    assert expected <= refs
    assert all("keyword" in h.matched for h in hits if h.passage.ref in expected)


def test_all_mode_declines_off_topic(passages):
    r = Retriever(passages)
    assert r.search("kubernetes deployment pipeline", mode="all") == []


def test_keyword_matches_morphological_variants(passages):
    """'merciful' should be found when searching 'mercy'."""
    r = Retriever(passages)
    hits = r.search("mercy", mode="all")
    texts = " ".join(h.passage.text.lower() for h in hits)
    assert "merciful" in texts


def test_top_mode_respects_k(passages):
    r = Retriever(passages)
    assert len(r.search("Allah", top_k=3)) <= 3


def test_passage_optional_fields_default_empty(passages):
    assert all(p.arabic == "" and p.tafseer == "" for p in passages)


# ------------------------------------------------- regression: review findings

from groundedrag.corpus import Passage
from groundedrag.retriever import _normalize_arabic, _stem


@pytest.mark.parametrize("a,b", [
    ("patience", "patient"), ("patience", "patiently"), ("mercy", "merciful"),
    ("pray", "prayer"), ("forgive", "forgiveness"), ("believe", "believer"),
    ("believe", "believing"), ("sign", "signs"), ("guidance", "guide"),
    ("worship", "worshippers"), ("fast", "fasting"),
])
def test_stemmer_unifies_true_variants(a, b):
    assert _stem(a) == _stem(b)


@pytest.mark.parametrize("a,b", [
    ("angel", "anger"), ("worship", "worse"), ("light", "lightning"),
    ("charity", "chariot"), ("night", "nigh"), ("sign", "significant"),
    ("mercy", "merchant"), ("ease", "east"), ("hell", "hello"),
])
def test_stemmer_rejects_prefix_twins(a, b):
    """A shared 4-char prefix must not count as a morphological match."""
    assert _stem(a) != _stem(b)


@pytest.mark.parametrize("cp", [0x64B, 0x652, 0x653, 0x654, 0x670, 0x651])
def test_arabic_normalization_strips_all_marks(cp):
    """Harakat through U+065F, plus dagger alef, must not survive."""
    assert chr(cp) not in _normalize_arabic(f"م{chr(cp)}ن")


def test_arabic_dagger_alef_matches_both_spellings():
    """Uthmani dagger alef reads as both 'with alef' and 'without'."""
    verses = [
        Passage("1:2", "The Praise is for Allah, Lord of the Worlds",
                arabic="ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ"),
        Passage("1:3", "The Beneficent, the Merciful", arabic="ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"),
    ]
    r = Retriever(verses)
    assert {h.passage.ref for h in r.search("العالمين", mode="all")} == {"1:2"}
    assert {h.passage.ref for h in r.search("الرحمن", mode="all")} == {"1:3"}


def test_arabic_short_particles_do_not_flood():
    verses = [Passage(f"1:{i}", f"verse {i}", arabic="ٱلْحَمْدُ لِلَّهِ مِن رَبِّ")
              for i in range(1, 6)]
    r = Retriever(verses)
    assert r.search("من", mode="all") == []      # below min term length
    assert r.search("و", mode="all") == []


def test_arabic_partial_match_without_semantic_layer():
    """Arabic queries have no TF-IDF/semantic corroboration available, so a
    substantial fraction of terms matching must stand on its own."""
    verses = [Passage("1:2", "The Praise is for Allah",
                      arabic="ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَٰلَمِينَ")]
    r = Retriever(verses)
    assert {h.passage.ref for h in r.search("الحمد لله العظيم", mode="all")} == {"1:2"}


def test_arabic_clitics_stripped(monkeypatch):
    verses = [Passage("20:11", "O Musa", arabic="يَٰمُوسَىٰ"),
              Passage("2:153", "with patience", arabic="بِٱلصَّبْرِ")]
    r = Retriever(verses)
    assert {h.passage.ref for h in r.search("موسي", mode="all")} == {"20:11"}
    assert {h.passage.ref for h in r.search("الصبر", mode="all")} == {"2:153"}


def test_semantic_layer_degrades_on_bad_embedding(tmp_path, passages):
    """A dimension mismatch or throwing embedder must fall back to lexical-only,
    never raise out of search()."""
    import numpy as np
    idx = tmp_path / "emb.npz"
    np.savez_compressed(idx, refs=np.array([p.ref for p in passages]),
                        matrix=np.zeros((len(passages), 8), dtype=np.float16))
    for bad in (lambda q: [0.1] * 4,                       # wrong dimension
                lambda q: [[0.1] * 8],                     # nested payload
                lambda q: (_ for _ in ()).throw(RuntimeError("boom"))):
        r = Retriever(passages, embeddings_path=idx, embed_query=bad)
        assert r.search("mercy") is not None                # no exception


def test_embeddings_index_mismatch_disables_semantic(tmp_path, passages, capsys):
    import numpy as np
    idx = tmp_path / "emb.npz"
    np.savez_compressed(idx, refs=np.array(["9:9"] * len(passages)),
                        matrix=np.zeros((len(passages), 8), dtype=np.float16))
    r = Retriever(passages, embeddings_path=idx)
    assert r._sem_matrix is None
    assert "does not align" in capsys.readouterr().err


def test_tfidf_matrix_stays_sparse(passages):
    """Densifying costs ~0.9 GB on the full corpus for identical math."""
    from scipy.sparse import issparse
    assert issparse(Retriever(passages)._matrix)


# ------------------------------------------------------------- rate limiting

from groundedrag import RateLimiter


def test_rate_limiter_blocks_past_the_limit():
    rl = RateLimiter(3, 60)
    assert [rl.check("a")[0] for _ in range(5)] == [True, True, True, False, False]


def test_rate_limiter_is_per_client():
    rl = RateLimiter(1, 60)
    assert rl.check("a")[0] and rl.check("b")[0]
    assert not rl.check("a")[0]


def test_rate_limiter_reports_retry_after():
    rl = RateLimiter(1, 60)
    rl.check("a")
    allowed, retry_after = rl.check("a")
    assert not allowed and 0 < retry_after <= 61


def test_rate_limiter_window_expires():
    rl = RateLimiter(1, 0.05)
    assert rl.check("a")[0]
    assert not rl.check("a")[0]
    time.sleep(0.06)
    assert rl.check("a")[0]


def test_rate_limit_of_zero_disables():
    rl = RateLimiter(0, 60)
    assert all(rl.check("a")[0] for _ in range(20))


# --------------------------------------------- regression: deploy-review fixes

import sqlite3
import stat as _stat
import subprocess

from groundedrag import TafseerStore, client_key


class _Req:
    """Minimal stand-in for a Starlette request."""
    def __init__(self, xff=None, peer="203.0.113.9"):
        self.headers = {"x-forwarded-for": xff} if xff is not None else {}
        self.client = type("C", (), {"host": peer})()


def test_spoofed_forwarded_for_cannot_reset_the_rate_limit():
    """The proxy appends the real IP, so the rightmost entry is the trustworthy
    one. Reading the leftmost let a caller mint a fresh bucket per request."""
    keys = {client_key(_Req(f"{i}.{i}.{i}.{i}, 198.51.100.7"), trusted_hops=1)
            for i in range(5)}
    assert keys == {"198.51.100.7"}


def test_client_key_falls_back_to_the_socket():
    assert client_key(_Req(), trusted_hops=1) == "203.0.113.9"
    assert client_key(_Req(""), trusted_hops=1) == "203.0.113.9"
    # no proxy in front: headers must be ignored entirely
    assert client_key(_Req("1.1.1.1, 2.2.2.2"), trusted_hops=0) == "203.0.113.9"
    # more hops claimed than present must clamp, never reach a spoofable entry
    assert client_key(_Req("9.9.9.9"), trusted_hops=2) == "9.9.9.9"


def _build_db(tmp_path, corpus="data/quran_sample.jsonl"):
    root = Path(__file__).resolve().parents[1]
    dest = tmp_path / "t.sqlite3"
    subprocess.run([sys.executable, str(root / "scripts" / "build_tafseer_db.py"),
                    str(root / corpus), str(dest)], check=True,
                   capture_output=True)
    return dest


def test_tafseer_db_is_readable_by_other_users(tmp_path):
    """Built as root in the image but served as uid 1000: mkstemp's 0600 would
    make the app unable to open its own database and crash at startup."""
    db = _build_db(tmp_path)
    mode = _stat.S_IMODE(db.stat().st_mode)
    assert mode & _stat.S_IROTH, f"db mode {oct(mode)} is not world-readable"


def test_tafseer_store_roundtrip(tmp_path):
    db = _build_db(tmp_path)
    store = TafseerStore(db)
    # the sample corpus carries no tafseer, so nothing should report as having any
    assert len(store) == 0
    assert store.has("1:1") is False
    assert store.get("1:1") == ""
    assert store.get("does:not-exist") == ""


def test_unusable_tafseer_db_raises_sqlite_error_not_something_exotic(tmp_path):
    """api.py degrades on sqlite3.Error; anything else would escape and crash
    the process at import."""
    empty = tmp_path / "empty.sqlite3"
    empty.touch()
    with pytest.raises(sqlite3.Error):
        TafseerStore(empty)

    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_text("this is definitely not a database")
    with pytest.raises(sqlite3.Error):
        TafseerStore(garbage)


def test_render_blueprint_does_not_pin_container_paths():
    """A blueprint env var overrides the image's ENV, so a stale path here
    crash-loops the container at import. The image already sets these."""
    blueprint = (Path(__file__).resolve().parents[1] / "render.yaml").read_text()
    for var in ("QURAN_DATA_PATH", "EMBEDDINGS_PATH", "TAFSEER_DB"):
        assert f"key: {var}" not in blueprint, (
            f"{var} is pinned in render.yaml; it would override the image's "
            f"correct value and can silently go stale")
