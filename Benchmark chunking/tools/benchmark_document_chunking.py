"""Benchmark fixed-character vs semantic chunking on text-like sample documents.

Included by default: PDF, DOCX, TXT, Markdown, HTML, EPUB, and PPTX.  Tables,
images, audio, and legacy PPT are intentionally excluded because they need
structure-aware, OCR, transcription, or conversion-specific benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark_pdf_chunking import parse_args, run_benchmark

DOCUMENT_EXTENSIONS = ".pdf,.docx,.txt,.md,.html,.htm,.epub,.pptx"
DEFAULT_OUTPUT_DIR = Path(
    "approaches/approach_3_agentic_rag/outputs/document_chunking_benchmark"
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(
        argv,
        default_extensions=DOCUMENT_EXTENSIONS,
        default_output_dir=DEFAULT_OUTPUT_DIR,
    )
    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
