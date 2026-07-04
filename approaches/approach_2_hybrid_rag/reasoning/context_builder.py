"""Build compact reasoning context from retrieved files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..shared_src.file_readers import read_file
from ..shared_src.utils import read_text_with_fallback, truncate_text

from ..core.models import ContextItem


def build_contexts(
    candidates: list[dict[str, Any]],
    *,
    max_files: int = 8,
    max_chars_per_file: int = 12000,
) -> list[ContextItem]:
    """Prepare text blocks from selected chunks or cached full extraction."""

    contexts: list[ContextItem] = []
    for candidate in candidates[:max_files]:
        text = _candidate_text(candidate)
        if not text:
            continue
        contexts.append(
            ContextItem(
                relative_path=str(candidate.get("relative_path", "")),
                modality=str(candidate.get("modality", "unknown")),
                text=truncate_text(text, max_chars_per_file),
                score=float(candidate.get("score", 0.0)),
                reason=str(candidate.get("reason", "")),
                metadata={
                    "extension": candidate.get("extension", ""),
                    "columns": candidate.get("columns", []),
                    "sheet_names": candidate.get("sheet_names", []),
                },
            )
        )
    return contexts


def _candidate_text(candidate: dict[str, Any]) -> str:
    chunks = candidate.get("chunks") or []
    chunk_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks if chunk.get("text"))
    if chunk_text:
        return chunk_text

    extracted = candidate.get("extracted_text_path")
    if extracted and Path(extracted).exists():
        return read_text_with_fallback(extracted)

    absolute = candidate.get("absolute_path")
    if absolute and Path(absolute).exists():
        result = read_file(absolute)
        return result.content or str(candidate.get("text_preview", ""))

    return str(candidate.get("text_preview", ""))
