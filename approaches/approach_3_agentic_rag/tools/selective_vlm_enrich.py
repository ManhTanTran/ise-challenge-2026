"""Selective VLM enrichment for image files with weak OCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..shared_src.file_readers import enrich_image_parse_with_vlm, image_parse_cache_path
from ..shared_src.utils import read_text_with_fallback


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _image_manifest_items(
    manifest: list[dict[str, Any]],
    *,
    min_ocr_confidence: float = 40.0,
    min_text_chars: int = 30,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in manifest:
        if str(item.get("modality", "")) != "image":
            continue
        confidence = item.get("ocr_confidence")
        text_chars = len(read_text_with_fallback(item["extracted_text_path"])) if item.get("extracted_text_path") and Path(item["extracted_text_path"]).exists() else len(str(item.get("text_preview") or ""))
        low_conf = confidence is None or (isinstance(confidence, float) and pd.isna(confidence)) or float(confidence) < min_ocr_confidence
        if low_conf or text_chars < min_text_chars:
            items.append({**item, "_text_chars": text_chars})
    return items


def enrich_manifest_images(
    manifest_path: str | Path,
    *,
    min_ocr_confidence: float = 40.0,
    min_text_chars: int = 30,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select weak OCR images, call VLM on them, and update the manifest in place."""

    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    candidates = _image_manifest_items(
        manifest,
        min_ocr_confidence=min_ocr_confidence,
        min_text_chars=min_text_chars,
    )
    if limit is not None:
        candidates = candidates[:limit]

    manifest_by_path = {str(item.get("relative_path", "")): item for item in manifest}
    for item in candidates:
        path = Path(item.get("absolute_path") or "")
        if not path.exists():
            continue
        parse_path = item.get("image_parse_path")
        if not parse_path:
            extracted = item.get("extracted_text_path")
            if not extracted:
                continue
            parse_path = str(image_parse_cache_path(extracted))
        merged = enrich_image_parse_with_vlm(path, parse_cache_path=parse_path)
        if not merged:
            continue
        current = manifest_by_path[str(item.get("relative_path", ""))]
        current["ocr_confidence"] = merged.get("avg_confidence", merged.get("confidence", current.get("ocr_confidence")))
        current["image_caption"] = merged.get("caption", current.get("image_caption", ""))
        current["image_description"] = merged.get("description", current.get("image_description", ""))
        current["status"] = "ok" if current.get("status") == "error" and (current.get("image_caption") or current.get("image_description") or current.get("ocr_confidence")) else current.get("status")
        current["error_message"] = "" if current.get("status") == "ok" else current.get("error_message", "")

    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selective VLM enrichment for low-OCR images.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json.")
    parser.add_argument("--min-ocr-confidence", type=float, default=40.0)
    parser.add_argument("--min-text-chars", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    enriched = enrich_manifest_images(
        args.manifest,
        min_ocr_confidence=args.min_ocr_confidence,
        min_text_chars=args.min_text_chars,
        limit=args.limit,
    )
    print(f"Enriched {len(enriched)} manifest records")


if __name__ == "__main__":
    main()
