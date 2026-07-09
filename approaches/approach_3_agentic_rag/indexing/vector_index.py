"""Buoc 0c: persisted semantic vector index over retrieval chunks.

Dense embeddings use FAISS when installed (inner product over normalized
vectors); otherwise plain numpy. Sparse TF-IDF embeddings use sklearn cosine
similarity. Either way `scores()` returns one similarity per chunk, aligned
with the chunk list, which is what hybrid retrieval needs for score merging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..shared_src.utils import dump_json, ensure_dir, load_json, stable_hash
from .embedder import TfidfBackend, create_backend

LOGGER = logging.getLogger(__name__)

_EMBED_BATCH = 64


class VectorIndex:
    """Chunk-level semantic index with disk persistence."""

    def __init__(self, backend: Any, embeddings: Any, meta: dict[str, Any]) -> None:
        self.backend = backend
        self.embeddings = embeddings
        self.meta = meta
        self._faiss_index = None
        if not self.is_sparse:
            self._faiss_index = _try_build_faiss(embeddings)

    @property
    def is_sparse(self) -> bool:
        return not isinstance(self.embeddings, np.ndarray)

    @classmethod
    def load_or_build(
        cls,
        chunks: list[dict[str, Any]],
        directory: str | Path,
        *,
        model_name: str,
        rebuild: bool = False,
    ) -> "VectorIndex":
        """Load a compatible persisted index or embed all chunks anew."""

        index_dir = ensure_dir(directory)
        fingerprint = _fingerprint(chunks)
        meta = load_json(index_dir / "meta.json", default=None)

        if not rebuild and meta and meta.get("fingerprint") == fingerprint:
            loaded = cls._try_load(index_dir, meta, chunks)
            if loaded is not None:
                LOGGER.info("Loaded vector index (%s) from %s", meta.get("kind"), index_dir)
                return loaded

        backend = create_backend(model_name)
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        backend.fit(texts)
        embeddings = _encode_corpus(backend, texts)
        meta = {
            "kind": backend.kind,
            "model_name": getattr(backend, "model_name", ""),
            "num_chunks": len(chunks),
            "fingerprint": fingerprint,
        }
        cls._persist(index_dir, backend, embeddings, meta)
        LOGGER.info("Built vector index (%s) over %d chunks", backend.kind, len(chunks))
        return cls(backend, embeddings, meta)

    def scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every chunk."""

        if self.meta.get("num_chunks", 0) == 0:
            return np.zeros(0)
        query_vector = self.backend.encode([query])
        if self.is_sparse:
            from sklearn.metrics.pairwise import cosine_similarity

            return cosine_similarity(query_vector, self.embeddings).ravel()
        if self._faiss_index is not None:
            k = int(self.embeddings.shape[0])
            sims, ids = self._faiss_index.search(np.asarray(query_vector, dtype=np.float32), k)
            scores = np.zeros(k, dtype=np.float32)
            scores[ids[0]] = sims[0]
            return scores
        return (self.embeddings @ np.asarray(query_vector, dtype=np.float32).T).ravel()

    @classmethod
    def _try_load(
        cls,
        index_dir: Path,
        meta: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> "VectorIndex | None":
        try:
            if meta.get("kind") == "tfidf-char":
                from scipy import sparse

                backend = TfidfBackend()
                backend.load(index_dir)
                embeddings = sparse.load_npz(index_dir / "embeddings.npz")
                return cls(backend, embeddings, meta)

            backend = create_backend(str(meta.get("model_name", "")))
            if backend.kind != meta.get("kind"):
                return None
            embeddings = np.load(index_dir / "embeddings.npy")
            return cls(backend, embeddings, meta)
        except Exception as exc:
            LOGGER.warning("Could not load persisted vector index (%s); rebuilding.", exc)
            return None

    @staticmethod
    def _persist(index_dir: Path, backend: Any, embeddings: Any, meta: dict[str, Any]) -> None:
        if isinstance(embeddings, np.ndarray):
            np.save(index_dir / "embeddings.npy", embeddings)
        else:
            from scipy import sparse

            sparse.save_npz(index_dir / "embeddings.npz", embeddings.tocsr())
        backend.save(index_dir)
        dump_json(meta, index_dir / "meta.json")


def _encode_corpus(backend: Any, texts: list[str]) -> Any:
    if isinstance(backend, TfidfBackend):
        return backend.encode(texts)
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    batches = [
        backend.encode(texts[start : start + _EMBED_BATCH])
        for start in range(0, len(texts), _EMBED_BATCH)
    ]
    return np.vstack(batches)


def _fingerprint(chunks: list[dict[str, Any]]) -> str:
    payload = "|".join(
        f"{chunk.get('chunk_id')}:{len(str(chunk.get('text', '')))}" for chunk in chunks
    )
    return stable_hash(payload, length=24)


def _try_build_faiss(embeddings: np.ndarray) -> Any | None:
    try:
        import faiss

        index = faiss.IndexFlatIP(int(embeddings.shape[1]))
        index.add(np.asarray(embeddings, dtype=np.float32))
        return index
    except Exception:
        return None
