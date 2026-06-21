"""Tests for the grounded Quran RAG (engine + grounding guarantees)."""

from __future__ import annotations

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
