"""Scan and index all files in a multimodal data lake."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import get_config
from .file_readers import file_metadata, read_file
from .utils import dump_json, ensure_dir, load_json, stable_file_id, truncate_text

LOGGER = logging.getLogger(__name__)


EXPECTED_INDEX_KEYS = {
    "file_id",
    "filename",
    "relative_path",
    "absolute_path",
    "extension",
    "modality",
    "mime_type",
    "size_bytes",
    "text_preview",
    "columns",
    "sheet_names",
    "extracted_text_path",
    "image_parse_path",
    "image_caption",
    "image_description",
    "ocr_confidence",
    "ocr_engine",
    "status",
    "error_message",
}


def build_file_index(
    data_lake_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    extracted_text_dir: str | Path | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Recursively scan a data lake and write file_index.json."""

    data_root = Path(data_lake_dir).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data lake directory not found: {data_root}")
    config = get_config()
    output = Path(output_path or config.file_index_path)
    cache_dir = Path(extracted_text_dir or output.parent / "extracted_texts")
    ensure_dir(cache_dir)

    files = [path for path in data_root.rglob("*") if path.is_file()]
    items: list[dict[str, Any]] = []
    for path in tqdm(files, desc="Indexing files"):
        items.append(index_file(path, data_root, cache_dir=cache_dir, force=force))

    output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(items, output)
    LOGGER.info("Indexed %d files to %s", len(items), output)
    return items


def reindex_failed_files(
    manifest_path: str | Path,
    data_lake_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    statuses: tuple[str, ...] = ("error",),
) -> list[dict[str, Any]]:
    """Re-run indexing only for files marked with a failing status.

    Lets a full indexing pass finish once (accepting some per-file failures -
    e.g. a parser bug that's since been fixed), then retries just those files
    afterward instead of re-scanning and re-extracting every already-good
    file (OCR/Whisper are the expensive steps `read_file`'s own per-file
    cache already skips on a hit; only files that failed never wrote that
    cache, so only they redo real work here). Writes the merged manifest
    back to `manifest_path`.

    IMPORTANT: after calling this, delete the work dir's `chunks.json` (and
    the vector_index it feeds) before the next `build_indexes(...)` call -
    chunks.json is itself only rebuilt when it's missing/stale, so a fixed
    manifest alone won't reach retrieval until `chunks.json` is regenerated
    from it.
    """

    manifest_file = Path(manifest_path)
    items = load_json(manifest_file, default=[])
    if not isinstance(items, list):
        raise ValueError(f"{manifest_file} does not contain a manifest list.")

    data_root = Path(data_lake_dir).expanduser().resolve()
    cache = Path(cache_dir) if cache_dir else manifest_file.parent / "text_cache"
    ensure_dir(cache)

    to_retry = [(index, item) for index, item in enumerate(items) if item.get("status") in statuses]
    if not to_retry:
        LOGGER.info("No files with status in %s - nothing to retry.", statuses)
        return items

    for index, item in tqdm(to_retry, desc="Retrying failed files"):
        relative = str(item.get("relative_path", ""))
        absolute = data_root / relative
        if not absolute.exists():
            LOGGER.warning("Skipping missing file: %s", absolute)
            continue
        items[index] = index_file(absolute, data_root, cache_dir=cache, force=True)

    dump_json(items, manifest_file)
    still_failing = [item for item in items if item.get("status") == "error"]
    LOGGER.info(
        "Retried %d file(s); %d still failing.", len(to_retry), len(still_failing)
    )
    return items


def index_file(
    path: str | Path,
    data_lake_dir: str | Path,
    *,
    cache_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Index one file and return the normalized metadata record."""

    metadata = file_metadata(path, data_lake_dir)
    result = read_file(
        path,
        cache_dir=cache_dir,
        data_lake_dir=data_lake_dir,
        use_cache=not force,
    )
    reader_meta = result.metadata or {}

    columns: list[str] = []
    sheet_names: list[str] = []
    if "columns" in reader_meta:
        columns = [str(col) for col in reader_meta.get("columns", [])]
    if "sheet_names" in reader_meta:
        sheet_names = [str(sheet) for sheet in reader_meta.get("sheet_names", [])]
    if "sheets" in reader_meta:
        for sheet in reader_meta.get("sheets", {}).values():
            columns.extend(str(col) for col in sheet.get("columns", []))
        columns = sorted(set(columns))

    item: dict[str, Any] = {
        "file_id": stable_file_id(metadata["relative_path"]),
        **metadata,
        "text_preview": truncate_text(result.content, 1200),
        "columns": columns,
        "sheet_names": sheet_names,
        "extracted_text_path": reader_meta.get("extracted_text_path"),
        "image_parse_path": reader_meta.get("image_parse_path"),
        "image_caption": reader_meta.get("image_caption"),
        "image_description": reader_meta.get("image_description"),
        "ocr_confidence": reader_meta.get("ocr_confidence"),
        "ocr_engine": reader_meta.get("ocr_engine"),
        "status": "error" if result.error else "ok",
        "error_message": result.error,
    }
    if item["modality"] == "unknown":
        item["status"] = "skipped"
        item["error_message"] = item["error_message"] or "Unsupported file type."
    return item


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the iSE data lake file index.")
    parser.add_argument("--data-lake", required=True, help="Path to the data lake directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write file_index.json. Defaults to outputs/file_index.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract text instead of reusing cached extraction.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_file_index(args.data_lake, args.output, force=args.force)


if __name__ == "__main__":
    main()
