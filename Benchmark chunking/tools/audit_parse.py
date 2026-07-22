"""Grade the Buoc 0 parsing step by content quality, not just status.

`status == "ok"` in the manifest only means no exception was raised. A file can
still be parsed badly: a scanned PDF that yields no text, an image OCR'd with
low confidence, mojibake from a wrong encoding, or a table read with zero
columns. This tool reads the manifest (and the cached extracted text), scores
each file, and flags the ones a human should eyeball.

Usage (from repo root):

    python -X utf8 -m approaches.approach_3_agentic_rag.tools.audit_parse \
        --file-index "approaches/approach_1_solver_baseline/outputs/runs/parse_20260630_095706/file_index.json"

or point it at a work-dir that already contains manifest.json, or at a raw
Data-Lake with --data-lake to parse and audit in one shot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.manifest import load_or_build_manifest
from ..shared_src.file_readers import looks_like_mojibake
from ..shared_src.utils import read_text_with_fallback

# Modalities that MUST yield text; near-empty extraction there is suspicious.
TEXT_MODALITIES = {"document", "table", "audio"}
EMPTY_TEXT_CHARS = 30
SCANNED_PDF_CHARS = 200
LOW_OCR_CONFIDENCE = 40.0
TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def extracted_length(item: dict[str, Any]) -> int:
    """Length of the cached extracted text, falling back to the preview."""

    path = item.get("extracted_text_path")
    if path and Path(path).exists():
        try:
            return len(read_text_with_fallback(path))
        except OSError:
            return -1
    return len(str(item.get("text_preview") or ""))


def _extracted_text(item: dict[str, Any], limit: int = 4000) -> str:
    path = item.get("extracted_text_path")
    if path and Path(path).exists():
        try:
            return read_text_with_fallback(path)[:limit]
        except OSError:
            return ""
    return str(item.get("text_preview") or "")[:limit]


def audit_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return per-file quality flags with a short reason string."""

    modality = str(item.get("modality", "unknown"))
    extension = str(item.get("extension", "")).lower()
    status = str(item.get("status", ""))
    length = extracted_length(item)
    text_sample = _extracted_text(item)

    flags: list[str] = []
    if status == "error":
        flags.append("parse_error")
    if status == "skipped":
        flags.append("unsupported")

    if modality in TEXT_MODALITIES and status == "ok" and 0 <= length < EMPTY_TEXT_CHARS:
        flags.append("empty_text")
    if extension == ".pdf" and status == "ok" and length < SCANNED_PDF_CHARS:
        flags.append("maybe_scanned_pdf")
    if extension in TABLE_EXTENSIONS and not (item.get("columns") or []):
        flags.append("no_columns")

    if modality == "image":
        confidence = item.get("ocr_confidence")
        caption = item.get("image_caption") or item.get("image_description")
        if confidence is not None and float(confidence) < LOW_OCR_CONFIDENCE and not caption:
            flags.append("low_ocr_no_caption")

    if text_sample and looks_like_mojibake(text_sample):
        flags.append("mojibake")

    severity = "error" if any(f in flags for f in ("parse_error", "empty_text", "no_columns")) else (
        "warn" if flags else "ok"
    )
    return {
        "relative_path": item.get("relative_path", ""),
        "modality": modality,
        "extension": extension,
        "status": status,
        "text_chars": length,
        "ocr_confidence": item.get("ocr_confidence"),
        "columns": len(item.get("columns") or []),
        "severity": severity,
        "flags": ",".join(flags),
    }


def audit_manifest(manifest: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [audit_item(item) for item in manifest]
    frame = pd.DataFrame(rows)
    order = {"error": 0, "warn": 1, "ok": 2}
    return frame.sort_values(
        by=["severity", "modality"], key=lambda col: col.map(order).fillna(col)
    ).reset_index(drop=True)


def print_report(frame: pd.DataFrame) -> None:
    total = len(frame)
    by_severity = frame["severity"].value_counts().to_dict()
    print("=" * 70)
    print(f"PARSE AUDIT  -  {total} files")
    print(
        f"  ok:    {by_severity.get('ok', 0)}\n"
        f"  warn:  {by_severity.get('warn', 0)}  (review khi rảnh)\n"
        f"  error: {by_severity.get('error', 0)}  (nên sửa trước khi chạy)"
    )
    print("-" * 70)
    print("Theo modality (files | tổng chars trích được):")
    for modality, group in frame.groupby("modality"):
        chars = int(group["text_chars"].clip(lower=0).sum())
        print(f"  {modality:<10} {len(group):>3} files | {chars:>9,} chars")

    flagged = frame[frame["severity"] != "ok"]
    if flagged.empty:
        print("-" * 70)
        print("Không có file nào bị gắn cờ. Parse trông ổn.")
        return
    print("-" * 70)
    print("FILE CẦN XEM (severity != ok):")
    for _, row in flagged.iterrows():
        conf = "" if row["ocr_confidence"] is None else f" ocr={row['ocr_confidence']}"
        print(
            f"  [{row['severity']:<5}] {row['relative_path']}  "
            f"({row['modality']}, {row['text_chars']} chars{conf})  -> {row['flags']}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Buoc 0 parsing quality.")
    parser.add_argument("--data-lake", default=None, help="Data-Lake to parse then audit.")
    parser.add_argument("--work-dir", default=None, help="Dir holding manifest.json (or to build into).")
    parser.add_argument("--file-index", default=None, help="Existing manifest/file_index.json to audit.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force re-parsing the data lake.")
    parser.add_argument("--output", default=None, help="CSV path for the full audit table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not (args.file_index or args.work_dir or args.data_lake):
        raise SystemExit("Provide --file-index, --work-dir, or --data-lake.")

    work_dir = args.work_dir or (Path(args.output).parent if args.output else "outputs/audit")
    manifest = load_or_build_manifest(
        args.data_lake or ".",
        work_dir,
        file_index_path=args.file_index,
        rebuild=args.rebuild_index,
    )
    frame = audit_manifest(manifest)
    print_report(frame)

    output = Path(args.output) if args.output else Path(work_dir) / "parse_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print("-" * 70)
    print(f"Bảng đầy đủ: {output}")


if __name__ == "__main__":
    main()
