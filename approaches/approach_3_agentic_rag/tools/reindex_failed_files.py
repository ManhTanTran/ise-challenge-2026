"""Retry only the files that failed a previous indexing pass.

A full indexing run over a real Data-Lake can take a long time (OCR, Whisper
transcription). If a handful of files fail because of a parser bug, the
practical move is: let the run finish (manifest.json still gets written, with
those files marked `status: "error"`), fix the bug, then retry ONLY the
failed files - not rescan and re-extract every already-successful one.

Usage (repo root), after `build_indexes(...)` has written manifest.json once:

    python -X utf8 -m approaches.approach_3_agentic_rag.tools.reindex_failed_files \
        --manifest "outputs/run1/manifest.json" \
        --data-lake "data/sample_data_lake/Data-Lake"

IMPORTANT: this only updates manifest.json. Delete the work dir's
chunks.json (and re-run build_indexes) afterward so the fixed text actually
reaches retrieval - chunks.json/vector_index are only rebuilt when missing or
stale, so a fixed manifest alone won't be picked up otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..shared_src.file_indexer import reindex_failed_files
from ..shared_src.utils import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry indexing only for files with a failing status.")
    parser.add_argument("--manifest", required=True, help="Path to an existing manifest.json.")
    parser.add_argument("--data-lake", required=True, help="Path to the Data-Lake folder.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Extracted-text cache dir. Defaults to <manifest's folder>/text_cache.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help='Status to retry (repeatable). Defaults to "error" only.',
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=None,
        help='Restrict retry to this extension (repeatable, e.g. --extension .mp4). '
        "Default: no restriction (all files matching --status).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = parse_args(argv)
    statuses = tuple(args.status) if args.status else ("error",)
    extensions = tuple(args.extension) if args.extension else None
    items = reindex_failed_files(
        args.manifest,
        args.data_lake,
        cache_dir=args.cache_dir,
        statuses=statuses,
        extensions=extensions,
    )
    still_failing = [item for item in items if item.get("status") == "error"]
    print(f"Manifest now has {len(items)} files; {len(still_failing)} still failing.")
    for item in still_failing:
        print(f"  - {item.get('relative_path')}: {item.get('error_message')}")
    print(
        "Remember: delete chunks.json (and re-run build_indexes) so the fixed "
        "text reaches retrieval."
    )


if __name__ == "__main__":
    main()
