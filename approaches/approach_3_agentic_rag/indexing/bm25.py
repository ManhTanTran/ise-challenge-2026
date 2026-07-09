"""Buoc 0d: BM25 keyword index over retrieval chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from ..shared_src.utils import normalize_for_match


def tokenize(text: str) -> list[str]:
    """Accent-insensitive tokens; CJK runs kept whole."""

    normalized = normalize_for_match(text)
    raw = re.findall(r"[一-鿿]+|[a-z0-9_./*\-]+", normalized)
    return [token for token in raw if len(token) >= 2]


class BM25Index:
    """Okapi BM25 built in memory from the persisted chunk list."""

    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(str(chunk.get("text", ""))) for chunk in chunks]
        self._doc_counts = [Counter(doc) for doc in self._docs]
        self._doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            self._doc_freq.update(set(doc))
        self._avgdl = sum(len(doc) for doc in self._docs) / max(len(self._docs), 1)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of the query against every chunk."""

        query_tokens = tokenize(query)
        if not query_tokens or not self._docs:
            return np.zeros(len(self._docs))

        total_docs = len(self._docs)
        scores = np.zeros(total_docs)
        for index, counts in enumerate(self._doc_counts):
            doc_length = len(self._docs[index]) or 1
            score = 0.0
            for token in query_tokens:
                term_frequency = counts.get(token)
                if not term_frequency:
                    continue
                doc_frequency = self._doc_freq[token]
                idf = math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
                score += idf * (term_frequency * (self.k1 + 1)) / (
                    term_frequency
                    + self.k1 * (1 - self.b + self.b * doc_length / max(self._avgdl, 1e-6))
                )
            scores[index] = score
        return scores
