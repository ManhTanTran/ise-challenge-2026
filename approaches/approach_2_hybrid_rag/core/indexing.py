"""Manifest and chunk index construction for approach 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..shared_src.file_indexer import build_file_index
from ..shared_src.retriever import load_index_text
from ..shared_src.utils import chunk_text, dump_json, ensure_dir, load_json, normalize_spaces


def load_or_build_manifest(
    data_lake_dir: str | Path,
    work_dir: str | Path,
    *,
    file_index_path: str | Path | None = None,
    rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Load an existing manifest or build one from the data lake."""

    work = ensure_dir(work_dir)
    manifest_path = Path(file_index_path) if file_index_path else work / "manifest.json"
    if manifest_path.exists() and not rebuild:
        data = load_json(manifest_path, default=[])
        items = data if isinstance(data, list) else []
        return _repair_manifest_paths(items, data_lake_dir)

    if file_index_path and not manifest_path.exists():
        raise FileNotFoundError(f"File index not found: {manifest_path}")

    items = build_file_index(
        data_lake_dir,
        manifest_path,
        extracted_text_dir=work / "text_cache",
        force=rebuild,
    )
    return _repair_manifest_paths(items, data_lake_dir)


def load_or_build_chunks(
    manifest: list[dict[str, Any]],
    work_dir: str | Path,
    *,
    rebuild: bool = False,
    max_chars: int = 2200,
    overlap: int = 250,
) -> list[dict[str, Any]]:
    """Create a chunk index over extracted text and file metadata."""

    work = ensure_dir(work_dir)
    chunk_path = work / "chunks.json"
    if chunk_path.exists() and not rebuild:
        data = load_json(chunk_path, default=[])
        return data if isinstance(data, list) else []

    chunks: list[dict[str, Any]] = []
    for item in manifest:
        search_text = _searchable_text(item)
        parts = chunk_text(search_text, max_chars=max_chars, overlap=overlap)
        if not parts:
            parts = [_metadata_text(item)]
        for index, text in enumerate(parts):
            chunks.append(
                {
                    "chunk_id": f"{item.get('file_id') or item.get('relative_path')}::{index}",
                    "file_id": item.get("file_id"),
                    "relative_path": item.get("relative_path", ""),
                    "filename": item.get("filename", ""),
                    "extension": item.get("extension", ""),
                    "modality": item.get("modality", "unknown"),
                    "text": text,
                    "chunk_index": index,
                }
            )

    dump_json(chunks, chunk_path)
    return chunks


def _searchable_text(item: dict[str, Any]) -> str:
    metadata = _metadata_text(item)
    body = load_index_text(item, limit=120000)
    text = normalize_spaces(f"{metadata}\n{body}")
    return text


def _metadata_text(item: dict[str, Any]) -> str:
    pieces = [
        f"Filename: {item.get('filename', '')}",
        f"Path: {item.get('relative_path', '')}",
        f"Modality: {item.get('modality', '')}",
    ]
    columns = item.get("columns") or []
    if columns:
        pieces.append("Columns: " + ", ".join(str(col) for col in columns[:80]))
    sheets = item.get("sheet_names") or []
    if sheets:
        pieces.append("Sheets: " + ", ".join(str(sheet) for sheet in sheets[:30]))
    return "\n".join(pieces)


def _repair_manifest_paths(
    manifest: list[dict[str, Any]],
    data_lake_dir: str | Path,
) -> list[dict[str, Any]]:
    """Make reused indexes portable across local folder locations."""

    data_root = Path(data_lake_dir).resolve()
    repaired = []
    for item in manifest:
        current = dict(item)
        relative = str(current.get("relative_path", "")).replace("\\", "/")
        absolute = Path(str(current.get("absolute_path", "")))
        candidate = data_root / relative
        if relative and (not absolute.exists()) and candidate.exists():
            current["absolute_path"] = str(candidate)
        repaired.append(current)
    return repaired
