"""Buoc 0: build or load every offline index in one call."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Approach3Config
from ..core.manifest import load_or_build_chunks, load_or_build_manifest
from .bm25 import BM25Index
from .vector_index import VectorIndex


def build_indexes(
    data_lake_dir: str | Path,
    work_dir: str | Path,
    *,
    config: Approach3Config,
    file_index_path: str | Path | None = None,
    rebuild: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], VectorIndex, BM25Index]:
    """Return (manifest, chunks, vector_index, bm25_index) for one data lake."""

    manifest = load_or_build_manifest(
        data_lake_dir,
        work_dir,
        file_index_path=file_index_path,
        rebuild=rebuild,
    )
    chunks = load_or_build_chunks(
        manifest,
        work_dir,
        rebuild=rebuild,
        max_chars=config.chunk_chars,
        overlap=config.chunk_overlap,
    )
    vector_index = VectorIndex.load_or_build(
        chunks,
        Path(work_dir) / "vector_index",
        model_name=config.embedding_model,
        rebuild=rebuild,
    )
    bm25_index = BM25Index(chunks)
    return manifest, chunks, vector_index, bm25_index
