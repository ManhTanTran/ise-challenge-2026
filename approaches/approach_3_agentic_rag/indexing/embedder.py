"""Buoc 0b: embedding backends for the semantic vector index.

Preferred backend is sentence-transformers (multilingual). When it is not
installed the pipeline falls back to a character n-gram TF-IDF vectorizer so
the whole system still runs offline with the base requirements.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class SentenceTransformerBackend:
    """Dense multilingual embeddings via sentence-transformers."""

    kind = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def fit(self, corpus: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def save(self, directory: Path) -> None:
        return None

    def load(self, directory: Path) -> None:
        return None


class FastEmbedBackend:
    """Dense multilingual embeddings via fastembed (ONNX, no torch).

    A lighter drop-in for sentence-transformers: installs in seconds without
    the ~2GB torch download, which is why it sits between the two tiers.
    """

    kind = "fastembed"

    def __init__(self, model_name: str | None = None) -> None:
        import os

        from fastembed import TextEmbedding

        # Must be a model fastembed actually ships. This multilingual MiniLM is
        # supported and matches the sentence-transformers default; override with
        # ISE_FASTEMBED_MODEL (see TextEmbedding.list_supported_models()).
        self.model_name = model_name or os.getenv(
            "ISE_FASTEMBED_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self._model = TextEmbedding(model_name=self.model_name)

    def fit(self, corpus: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def save(self, directory: Path) -> None:
        return None

    def load(self, directory: Path) -> None:
        return None


class TfidfBackend:
    """Sparse char n-gram TF-IDF fallback (language agnostic, no downloads)."""

    kind = "tfidf-char"

    def __init__(self, model_name: str = "tfidf-char-3-5") -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.model_name = model_name
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus or [""])
        self._fitted = True

    def encode(self, texts: list[str]) -> Any:
        if not self._fitted:
            raise RuntimeError("TfidfBackend.encode called before fit/load.")
        return self._vectorizer.transform(texts)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "tfidf_vectorizer.pkl").open("wb") as handle:
            pickle.dump(self._vectorizer, handle)

    def load(self, directory: Path) -> None:
        with (directory / "tfidf_vectorizer.pkl").open("rb") as handle:
            self._vectorizer = pickle.load(handle)
        self._fitted = True


def create_backend(model_name: str) -> SentenceTransformerBackend | FastEmbedBackend | TfidfBackend:
    """Return the best available embedding backend.

    Preference order: sentence-transformers (best quality) -> fastembed (dense
    multilingual, no torch) -> char TF-IDF (always works, no downloads).
    """

    try:
        backend = SentenceTransformerBackend(model_name)
        LOGGER.info("Embedding backend: sentence-transformers (%s)", model_name)
        return backend
    except Exception as st_exc:
        LOGGER.info("sentence-transformers unavailable (%s); trying fastembed.", st_exc)

    try:
        backend = FastEmbedBackend()
        LOGGER.info("Embedding backend: fastembed (%s)", backend.model_name)
        return backend
    except Exception as fe_exc:
        LOGGER.warning(
            "fastembed unavailable (%s); falling back to char TF-IDF.", fe_exc
        )
        return TfidfBackend()
