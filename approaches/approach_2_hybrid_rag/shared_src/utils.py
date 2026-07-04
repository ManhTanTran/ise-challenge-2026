"""Shared utility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


NOT_ENOUGH_DATA = "Not enough data to answer."
DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "cp1252", "latin-1")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a compact root logger once."""

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def stable_hash(value: str, length: int = 16) -> str:
    """Return a stable short hash for ids and cache names."""

    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def stable_file_id(relative_path: str) -> str:
    """Create a deterministic file id from a relative path."""

    return stable_hash(relative_path.replace("\\", "/").lower())


def detect_mime(path: str | Path) -> str | None:
    """Best-effort MIME type detection based on extension."""

    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type


def read_text_with_fallback(path: str | Path, encodings: Iterable[str] = DEFAULT_ENCODINGS) -> str:
    """Read a text file with a small set of common encodings."""

    file_path = Path(path)
    errors: list[str] = []
    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    return file_path.read_text(encoding="latin-1", errors="replace")


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text, creating parent directories."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def load_json(path: str | Path, default: Any | None = None) -> Any:
    """Load JSON if it exists, otherwise return default."""

    resolved = Path(path)
    if not resolved.exists():
        return default
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(data: Any, path: str | Path) -> None:
    """Write pretty JSON."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def safe_relative_path(path: str | Path, root: str | Path) -> str:
    """Return a POSIX relative path when possible."""

    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def normalize_spaces(text: Any) -> str:
    """Collapse repeated whitespace and strip outer space."""

    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def strip_accents(text: str) -> str:
    """Remove accents for loose matching across Vietnamese text."""

    normalized = unicodedata.normalize("NFKD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_for_match(text: Any) -> str:
    """Lowercase, de-accent, and normalize punctuation for matching."""

    base = strip_accents(normalize_spaces(text).lower())
    return re.sub(r"[^a-z0-9_./* \-\u4e00-\u9fff]+", " ", base).strip()


def truncate_text(text: str, limit: int = 1000) -> str:
    """Return a compact text preview."""

    compact = normalize_spaces(text)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def chunk_text(text: str, max_chars: int = 3500, overlap: int = 300) -> list[str]:
    """Split text into overlapping chunks for context prompts."""

    compact = normalize_spaces(text)
    if not compact:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(compact):
        end = min(len(compact), start + max_chars)
        chunks.append(compact[start:end])
        if end == len(compact):
            break
        start = max(0, end - overlap)
    return chunks


def parse_jsonish_list(value: Any) -> list[str]:
    """Parse a JSON-ish list string from spreadsheet cells."""

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = normalize_spaces(value)
    if not text or text.lower() in {"nan", "none"}:
        return []
    text = (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part.strip().strip("\"'") for part in text.strip("[]").split(",") if part.strip()]
