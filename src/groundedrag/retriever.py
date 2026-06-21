"""Retrieval over the corpus with a relevance gate.

TF-IDF + cosine similarity (offline, no downloads). The relevance gate is what
makes the system *honest*: if nothing in the book clears the similarity
threshold, it returns nothing — so the answerer can say "this isn't addressed in
the source" instead of forcing a weak passage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Passage


@dataclass
class Hit:
    passage: Passage
    score: float


class Retriever:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self._vec = TfidfVectorizer(lowercase=True, stop_words="english",
                                    ngram_range=(1, 2), sublinear_tf=True)
        matrix = self._vec.fit_transform([p.text for p in passages])
        self._matrix = matrix.toarray().astype(np.float32)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix /= norms

    def search(self, query: str, top_k: int = 5, min_score: float = 0.05) -> list[Hit]:
        q = self._vec.transform([query]).toarray().astype(np.float32)[0]
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q /= n
        scores = self._matrix @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [Hit(self.passages[i], float(scores[i]))
                for i in order if scores[i] >= min_score]
