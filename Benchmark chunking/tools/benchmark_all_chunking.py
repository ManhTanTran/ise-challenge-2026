"""Unified 62-question benchmark for seven chunking configurations.

The corpus, source windows, BGE-M3 query/chunk embeddings, retrieval ranking,
and metrics are shared. GPT-4o-mini and Youtu-HiChunk are used only during
chunk generation and their outputs are cached for repeatable scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..shared_src.file_readers import (
    DEFAULT_ENCODINGS,
    _detect_csv_skiprows,
    _restore_headerless_list_row,
    load_table_file,
    read_file,
)
from ..shared_src.utils import ensure_dir
from .advanced_chunking import (
    JsonSummaryCache,
    build_raptor_rows,
    hichunk_rows,
    load_hichunk_splits,
    openai_summary_generator,
    pic_spans,
)
from .benchmark_late_chunking import (
    _chunk_rows,
    _evaluate,
    _extensions,
    _fixed_spans,
    _semantic_spans,
    _sentence_spans,
    _source_windows,
)
from .benchmark_pdf_chunking import _source_matches, load_document_questions
from .late_chunking import LongContextTokenEmbedder

BASE_METHODS = ("naive_fixed", "naive_semantic", "late_fixed", "late_semantic")
ALL_METHODS = (*BASE_METHODS, "pic", "raptor_all_nodes", "hichunk_flat")
BENCHMARK_EXTENSIONS = ".pdf,.docx,.txt,.md,.html,.htm,.epub,.pptx,.csv,.xlsx,.xls,.json,.sql"


def _cache_stem(relative: str, window_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    safe = hashlib.sha256(relative.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{safe}-{window_index:04d}-{digest}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _clean_preserving_structure(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n"))
    return re.sub(r"\n{4,}", "\n\n\n", text).strip()


def _read_benchmark_source(path: Path, *, output: Path, data_lake: Path) -> tuple[str, str, str | None]:
    """Read complete simple tables; preserve headings/rows in every modality."""

    if path.suffix.lower() in {".csv", ".xlsx", ".xls", ".sql"}:
        try:
            if path.suffix.lower() == ".csv":
                frame = None
                last_error: Exception | None = None
                for encoding in DEFAULT_ENCODINGS:
                    try:
                        skiprows = _detect_csv_skiprows(path, encoding)
                        frame = pd.read_csv(
                            path,
                            encoding=encoding,
                            skiprows=skiprows,
                            sep=None,
                            engine="python",
                            on_bad_lines="skip",
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                if frame is None:
                    raise RuntimeError(f"Could not read full CSV: {last_error}")
                tables = {path.stem: _restore_headerless_list_row(frame)}
            else:
                tables = load_table_file(path)
            text = "\n\n".join(
                f"Sheet: {name}\n{frame.to_csv(index=False)}" for name, frame in tables.items()
            )
            return _clean_preserving_structure(text), "table", None
        except Exception as exc:
            return "", "table", str(exc)
    result = read_file(path, cache_dir=output / "text_cache", data_lake_dir=data_lake, use_cache=True)
    return _clean_preserving_structure(result.content), result.modality, result.error


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_dir(Path(args.output_dir))
    extensions = _extensions(args.extensions)
    requested = tuple(dict.fromkeys(args.methods))
    unknown = set(requested) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    questions = load_document_questions([Path(item) for item in args.questions], extensions=extensions)
    expected_sources = {source for question in questions for source in question["expected_sources"]}
    data_lake = Path(args.data_lake)
    candidates = sorted(
        candidate
        for candidate in data_lake.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in extensions
        and _source_matches(candidate.relative_to(data_lake).as_posix(), expected_sources)
    )
    candidate_paths = [candidate.relative_to(data_lake).as_posix() for candidate in candidates]
    questions = [
        question
        for question in questions
        if all(any(_source_matches(candidate, {source}) for candidate in candidate_paths) for source in question["expected_sources"])
    ]
    active_sources = {source for question in questions for source in question["expected_sources"]}
    candidates = [
        candidate
        for candidate in candidates
        if _source_matches(candidate.relative_to(data_lake).as_posix(), active_sources)
    ]
    if not questions or not candidates:
        raise RuntimeError("The filtered dataset has no complete questions/evidence files.")

    embedder = LongContextTokenEmbedder(
        args.model,
        window_tokens=args.window_tokens,
        overlap_tokens=args.overlap_tokens,
        device=args.device,
    )
    chunk_sets: dict[str, list[dict[str, Any]]] = {method: [] for method in requested}
    vector_lists: dict[str, list[np.ndarray]] = {method: [] for method in requested}
    diagnostics: list[dict[str, Any]] = []
    hichunk_inputs: dict[str, dict[str, Any]] = {}
    hichunk_index = load_hichunk_splits(Path(args.hichunk_splits)) if args.hichunk_splits and Path(args.hichunk_splits).exists() else {}
    summary_cache = JsonSummaryCache(
        output / "generation_cache" / "gpt_summaries.json",
        openai_summary_generator(args.summary_model),
    )
    raptor_cache = ensure_dir(output / "generation_cache" / "raptor")
    started = time.perf_counter()

    for path in candidates:
        relative = path.relative_to(data_lake).as_posix()
        text, modality, parse_error = _read_benchmark_source(path, output=output, data_lake=data_lake)
        if parse_error or not text:
            diagnostics.append({"relative_path": relative, "parse_error": parse_error or "empty extracted text"})
            continue
        windows = _source_windows(text, args.source_window_chars)
        counts = {method: 0 for method in requested}
        for window_index, source_window in enumerate(windows):
            window_text = text[source_window.start : source_window.end]
            lookup_key = f"{relative}::window-{window_index:04d}"
            hichunk_inputs[lookup_key] = {
                "relative_path": relative,
                "window_index": window_index,
                "start_char": source_window.start,
                "end_char": source_window.end,
                "text": window_text,
            }

            need_fixed = any(method in requested for method in ("naive_fixed", "late_fixed"))
            need_semantic = any(method in requested for method in ("naive_semantic", "late_semantic"))
            need_late_context = any(method in requested for method in ("late_fixed", "late_semantic"))
            offsets = token_vectors = None
            sentence_spans = _sentence_spans(window_text, args.semantic_max_chars) if need_semantic else []
            if need_late_context:
                offsets, token_vectors = embedder.contextualize(window_text)
            if need_semantic:
                if offsets is not None and token_vectors is not None:
                    sentence_vectors = embedder.pool_spans(offsets, token_vectors, sentence_spans)
                else:
                    # Naive semantic is intentionally independent of Late
                    # Chunking: sentence embeddings are much faster than
                    # contextualising the complete source window.
                    sentence_vectors = embedder.encode_texts(
                        [window_text[span.start : span.end] for span in sentence_spans]
                    )
                semantic_spans, _ = _semantic_spans(
                    window_text, sentence_spans, sentence_vectors,
                    percentile=args.semantic_percentile,
                    min_chars=args.semantic_min_chars,
                    max_chars=args.semantic_max_chars,
                )
            else:
                semantic_spans = []
            fixed_spans = _fixed_spans(window_text, args.fixed_max_chars, args.fixed_overlap) if need_fixed else []
            for boundary, spans in (("fixed", fixed_spans), ("semantic", semantic_spans)):
                late_method, naive_method = f"late_{boundary}", f"naive_{boundary}"
                if not spans or (late_method not in requested and naive_method not in requested):
                    continue
                base_rows = _chunk_rows(path, relative, modality, window_text, spans, late_method)
                for row in base_rows:
                    row["start_char"] += source_window.start
                    row["end_char"] += source_window.start
                    row["chunk_id"] = f"{relative}::window-{window_index:04d}::{boundary}::{row['chunk_index']}"
                if late_method in requested:
                    if offsets is None or token_vectors is None:
                        raise RuntimeError("Late Chunking requested without contextual token vectors.")
                    chunk_sets[late_method].extend(base_rows)
                    vector_lists[late_method].extend(embedder.pool_spans(offsets, token_vectors, spans))
                    counts[late_method] += len(base_rows)
                if naive_method in requested:
                    rows = [{**row, "method": naive_method} for row in base_rows]
                    chunk_sets[naive_method].extend(rows)
                    vector_lists[naive_method].extend(embedder.encode_texts([row["text"] for row in rows]))
                    counts[naive_method] += len(rows)

            if "pic" in requested:
                summary = summary_cache.summarize(window_text, args.pic_summary_tokens)
                spans, pic_meta = pic_spans(window_text, summary, embedder, max_chars=args.fixed_max_chars)
                rows = _chunk_rows(path, relative, modality, window_text, spans, "pic")
                for row in rows:
                    row["start_char"] += source_window.start
                    row["end_char"] += source_window.start
                    row["chunk_id"] = f"{relative}::window-{window_index:04d}::pic::{row['chunk_index']}"
                    row["pic_threshold"] = pic_meta["threshold"]
                chunk_sets["pic"].extend(rows)
                vector_lists["pic"].extend(embedder.encode_texts([row["text"] for row in rows]))
                counts["pic"] += len(rows)

            if "raptor_all_nodes" in requested:
                stem = _cache_stem(relative, window_index, window_text)
                rows_path, vectors_path = raptor_cache / f"{stem}.jsonl", raptor_cache / f"{stem}.npy"
                if rows_path.exists() and vectors_path.exists():
                    rows, raptor_vectors = _read_jsonl(rows_path), np.load(vectors_path)
                else:
                    rows, raptor_vectors, _ = build_raptor_rows(
                        window_text,
                        relative_path=relative,
                        modality=modality,
                        embedder=embedder,
                        summarizer=summary_cache,
                        official_repo=Path(args.raptor_repo),
                        max_tokens=args.raptor_leaf_tokens,
                        num_layers=args.raptor_layers,
                        summarization_length=args.raptor_summary_tokens,
                    )
                    _write_jsonl(rows_path, rows)
                    np.save(vectors_path, raptor_vectors)
                for row in rows:
                    row["chunk_id"] = f"{relative}::window-{window_index:04d}::raptor::{row['chunk_index']}"
                chunk_sets["raptor_all_nodes"].extend(rows)
                vector_lists["raptor_all_nodes"].extend(raptor_vectors)
                counts["raptor_all_nodes"] += len(rows)

            if "hichunk_flat" in requested and hichunk_index:
                rows = hichunk_rows(relative, modality, hichunk_index, lookup_key=lookup_key)
                for row in rows:
                    row["chunk_id"] = f"{relative}::window-{window_index:04d}::hichunk::{row['chunk_index']}"
                chunk_sets["hichunk_flat"].extend(rows)
                if rows:
                    vector_lists["hichunk_flat"].extend(embedder.encode_texts([row["text"] for row in rows]))
                counts["hichunk_flat"] += len(rows)
        diagnostics.append({"relative_path": relative, "source_chars": len(text), "source_windows": len(windows), **{f"{method}_chunks": count for method, count in counts.items()}})

    (output / "hichunk_inputs.json").write_text(json.dumps(hichunk_inputs, ensure_ascii=False), encoding="utf-8")
    unavailable: dict[str, str] = {}
    if "hichunk_flat" in requested and not hichunk_index:
        unavailable["hichunk_flat"] = "Generate --hichunk-splits from hichunk_inputs.json first."
    active_methods = [method for method in requested if chunk_sets[method]]
    empty = set(requested) - set(active_methods)
    for method in empty:
        unavailable.setdefault(method, "No chunks were generated.")
    if not active_methods:
        raise RuntimeError(f"No methods available: {unavailable}")

    query_vectors = embedder.encode_query([question["question"] for question in questions])
    dense = {method: np.asarray(vector_lists[method], dtype=np.float32) for method in active_methods}
    active_chunks = {method: chunk_sets[method] for method in active_methods}
    results, summary = _evaluate(active_chunks, dense, questions, query_vectors, {method: 0.0 for method in active_methods})
    summary["config"] = {
        "benchmark_contract": "62 complete-evidence questions; 70 evidence files; common source windows; BGE-M3 cosine retrieval",
        "questions": len(questions),
        "documents": len(candidates),
        "requested_methods": list(requested),
        "active_methods": active_methods,
        "unavailable_methods": unavailable,
        "embedding_model": args.model,
        "embedding_backend": embedder.kind,
        "summary_model": args.summary_model,
        "source_window_chars": args.source_window_chars,
        "build_seconds": round(time.perf_counter() - started, 3),
        "raptor_repo": str(Path(args.raptor_repo).resolve()),
        "hichunk_splits": str(Path(args.hichunk_splits).resolve()) if args.hichunk_splits else None,
    }
    for method in active_methods:
        _write_jsonl(output / f"chunks_{method}.jsonl", active_chunks[method])
        np.save(output / f"embeddings_{method}.npy", dense[method])
    pd.DataFrame(diagnostics).to_csv(output / "document_diagnostics.csv", index=False)
    results.to_csv(output / "query_results.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-lake", required=True)
    parser.add_argument("--questions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--extensions", default=BENCHMARK_EXTENSIONS)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-tokens", type=int, default=8192)
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--source-window-chars", type=int, default=16000)
    parser.add_argument("--fixed-max-chars", type=int, default=1200)
    parser.add_argument("--fixed-overlap", type=int, default=0)
    parser.add_argument("--semantic-max-chars", type=int, default=1200)
    parser.add_argument("--semantic-min-chars", type=int, default=180)
    parser.add_argument("--semantic-percentile", type=float, default=25.0)
    parser.add_argument("--summary-model", default="gpt-4o-mini")
    parser.add_argument("--pic-summary-tokens", type=int, default=300)
    parser.add_argument("--raptor-repo", default=Path("raptor"))
    parser.add_argument("--raptor-leaf-tokens", type=int, default=100)
    parser.add_argument("--raptor-layers", type=int, default=3)
    parser.add_argument("--raptor-summary-tokens", type=int, default=150)
    parser.add_argument("--hichunk-splits")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
