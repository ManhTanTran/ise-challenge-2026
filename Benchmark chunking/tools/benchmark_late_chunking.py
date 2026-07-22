"""Phase 2 benchmark: naive versus Late Chunking with one local encoder.

The four result sets deliberately share corpus, query vectors, and boundaries:
``naive_fixed``, ``naive_semantic``, ``late_fixed``, ``late_semantic``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..shared_src.file_readers import read_file
from ..shared_src.utils import ensure_dir, normalize_for_match, normalize_spaces
from .benchmark_pdf_chunking import K_VALUES, _norm_source, _source_matches, load_document_questions
from .late_chunking import LongContextTokenEmbedder, TextSpan

DEFAULT_EXTENSIONS = ".pdf,.docx,.txt,.md,.html,.htm,.epub,.pptx"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+")


def _extensions(value: str) -> set[str]:
    return {part.strip().lower() if part.strip().startswith(".") else f".{part.strip().lower()}" for part in value.split(",") if part.strip()}


def _fixed_spans(text: str, max_chars: int, overlap: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        spans.append(TextSpan(start, end))
        if end == len(text):
            break
        start = end - overlap
    return spans


def _source_windows(text: str, max_chars: int) -> list[TextSpan]:
    """Split oversized source files before token-level encoding.

    Late pooling preserves context inside each source window. This prevents a
    single very long PDF from materialising every token vector in RAM while
    keeping all of its retrieval chunks under the original file path.
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return [TextSpan(0, len(text))]
    windows: list[TextSpan] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        windows.append(TextSpan(start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return windows


def _sentence_spans(text: str, max_chars: int) -> list[TextSpan]:
    spans: list[TextSpan] = []
    start = 0
    for match in list(_SENTENCE_BOUNDARY.finditer(text)) + [None]:
        raw_end = match.start() if match else len(text)
        left = start
        while left < raw_end and text[left].isspace():
            left += 1
        right = raw_end
        while right > left and text[right - 1].isspace():
            right -= 1
        while left < right:
            end = min(right, left + max_chars)
            if end < right:
                whitespace = text.rfind(" ", left, end + 1)
                end = whitespace if whitespace > left else end
            spans.append(TextSpan(left, end))
            left = end
            while left < right and text[left].isspace():
                left += 1
        start = match.end() if match else len(text)
    return spans


def _semantic_spans(
    text: str, sentence_spans: list[TextSpan], sentence_vectors: np.ndarray, *, percentile: float, min_chars: int, max_chars: int
) -> tuple[list[TextSpan], dict[str, Any]]:
    if not sentence_spans:
        return [], {"sentence_count": 0, "threshold": None, "semantic_boundaries": 0}
    similarities = np.sum(sentence_vectors[:-1] * sentence_vectors[1:], axis=1)
    threshold = float(np.percentile(similarities, percentile)) if len(similarities) else None
    cuts = {index for index, value in enumerate(similarities) if threshold is not None and value < threshold}
    groups: list[TextSpan] = []
    current = sentence_spans[0]
    for index, span in enumerate(sentence_spans[1:], start=0):
        if index in cuts:
            groups.append(current)
            current = span
        else:
            current = TextSpan(current.start, span.end)
    groups.append(current)
    # Merge short groups. Prefer the left neighbor; size guards are more
    # important than an extra similarity heuristic in this benchmark control.
    merged: list[TextSpan] = []
    for span in groups:
        if merged and span.end - span.start < min_chars and span.end - merged[-1].start <= max_chars:
            merged[-1] = TextSpan(merged[-1].start, span.end)
        else:
            merged.append(span)
    guarded: list[TextSpan] = []
    for span in merged:
        relative_spans = _fixed_spans(text[span.start:span.end], max_chars, 0)
        guarded.extend(
            TextSpan(span.start + item.start, span.start + item.end) for item in relative_spans
        )
    return guarded, {"sentence_count": len(sentence_spans), "threshold": threshold, "semantic_boundaries": len(cuts)}


def _chunk_rows(path: Path, relative: str, modality: str, text: str, spans: list[TextSpan], method: str) -> list[dict[str, Any]]:
    return [{"chunk_id": f"{relative}::{index}", "relative_path": relative, "filename": path.name, "extension": path.suffix.lower(), "modality": modality, "chunk_index": index, "text": text[span.start:span.end], "char_count": span.end - span.start, "start_char": span.start, "end_char": span.end, "method": method} for index, span in enumerate(spans)]


def _evaluate(chunk_sets: dict[str, list[dict[str, Any]]], vectors: dict[str, np.ndarray], questions: list[dict[str, Any]], query_vectors: np.ndarray, encode_seconds: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for method, chunks in chunk_sets.items():
        paths = [str(chunk["relative_path"]) for chunk in chunks]
        unique_paths = sorted(set(paths))
        for query_index, question in enumerate(questions):
            scores = vectors[method] @ query_vectors[query_index]
            order = np.argsort(-scores)
            expected = {_norm_source(source) for source in question["expected_sources"]}
            chunk_rank = next((rank for rank, index in enumerate(order, 1) if _source_matches(paths[int(index)], expected)), None)
            file_scores = {path: max(float(scores[i]) for i, candidate in enumerate(paths) if candidate == path) for path in unique_paths}
            file_order = sorted(file_scores, key=file_scores.get, reverse=True)
            file_rank = next((rank for rank, path in enumerate(file_order, 1) if _source_matches(path, expected)), None)
            groundtruth = normalize_for_match(question["groundtruth"])
            answer_rank = next((rank for rank, index in enumerate(order, 1) if len(groundtruth) >= 4 and not groundtruth.isdigit() and groundtruth in normalize_for_match(str(chunks[int(index)]["text"]))), None)
            record: dict[str, Any] = {"method": method, "dataset": question["dataset"], "id": question["id"], "question": question["question"], "expected_sources": json.dumps(question["expected_sources"], ensure_ascii=False), "first_correct_chunk_rank": chunk_rank, "first_correct_file_rank": file_rank, "first_answer_text_chunk_rank": answer_rank, "chunk_mrr": 1 / chunk_rank if chunk_rank else 0.0, "file_mrr": 1 / file_rank if file_rank else 0.0, "answer_text_mrr": 1 / answer_rank if answer_rank else (0.0 if len(groundtruth) >= 4 and not groundtruth.isdigit() else None)}
            for k in K_VALUES:
                chunk_sources = {_norm_source(paths[int(index)]) for index in order[:k]}
                file_sources = {_norm_source(path) for path in file_order[:k]}
                chunk_found = sum(any(_source_matches(source, {wanted}) for source in chunk_sources) for wanted in expected)
                file_found = sum(any(_source_matches(source, {wanted}) for source in file_sources) for wanted in expected)
                record[f"chunk_hit@{k}"] = bool(chunk_rank and chunk_rank <= k)
                record[f"chunk_source_recall@{k}"] = chunk_found / len(expected)
                record[f"file_recall@{k}"] = file_found / len(expected)
                record[f"file_fully_covered@{k}"] = file_found == len(expected)
            rows.append(record)
        method_rows = [row for row in rows if row["method"] == method]
        lengths = np.asarray([chunk["char_count"] for chunk in chunks], dtype=np.float32)
        summary[method] = {"documents": len(unique_paths), "chunks": len(chunks), "mean_chars": round(float(lengths.mean()), 2), "p95_chars": round(float(np.percentile(lengths, 95)), 2), "max_chars": int(lengths.max()), "chunk_encode_seconds": round(encode_seconds[method], 3), "questions": len(questions), "chunk_mrr": round(float(np.mean([row["chunk_mrr"] for row in method_rows])), 4), "file_mrr": round(float(np.mean([row["file_mrr"] for row in method_rows])), 4), "answer_text_questions": sum(row["answer_text_mrr"] is not None for row in method_rows), "answer_text_mrr": round(float(np.mean([row["answer_text_mrr"] for row in method_rows if row["answer_text_mrr"] is not None])), 4) if any(row["answer_text_mrr"] is not None for row in method_rows) else None}
        for prefix in ("chunk_hit", "chunk_source_recall", "file_recall", "file_fully_covered"):
            for k in K_VALUES:
                summary[method][f"{prefix}@{k}"] = round(float(np.mean([row[f"{prefix}@{k}"] for row in method_rows])), 4)
    return pd.DataFrame(rows), summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = ensure_dir(Path(args.output_dir))
    extensions = _extensions(args.extensions)
    questions = load_document_questions(
        [Path(item) for item in args.questions], extensions=extensions
    )
    if not questions:
        raise RuntimeError("No questions backed by selected document extensions.")
    expected_sources = {
        source
        for question in questions
        for source in question["expected_sources"]
    }
    embedder = LongContextTokenEmbedder(args.model, window_tokens=args.window_tokens, overlap_tokens=args.overlap_tokens, device=args.device)
    chunk_sets = {key: [] for key in ("naive_fixed", "naive_semantic", "late_fixed", "late_semantic")}
    vectors: dict[str, list[np.ndarray]] = {key: [] for key in chunk_sets}
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    candidates = sorted(
        candidate
        for candidate in Path(args.data_lake).rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in extensions
        and _source_matches(
            candidate.relative_to(args.data_lake).as_posix(), expected_sources
        )
    )
    if not candidates:
        raise RuntimeError("None of the question evidence files were found in the data lake.")
    candidate_paths = [
        candidate.relative_to(args.data_lake).as_posix() for candidate in candidates
    ]
    questions = [
        question
        for question in questions
        if all(
            any(_source_matches(candidate, {source}) for candidate in candidate_paths)
            for source in question["expected_sources"]
        )
    ]
    if not questions:
        raise RuntimeError("No questions have a complete evidence set in the data lake.")
    active_sources = {
        source
        for question in questions
        for source in question["expected_sources"]
    }
    candidates = [
        candidate
        for candidate in candidates
        if _source_matches(candidate.relative_to(args.data_lake).as_posix(), active_sources)
    ]
    for path in candidates:
        result = read_file(path, cache_dir=output / "text_cache", data_lake_dir=Path(args.data_lake), use_cache=True)
        if result.error or not result.content.strip():
            diagnostics.append({"relative_path": path.relative_to(args.data_lake).as_posix(), "parse_error": result.error or "empty extracted text"})
            continue
        text = normalize_spaces(result.content)
        relative = path.relative_to(args.data_lake).as_posix()
        token_count = macro_windows = fixed_count = semantic_count = 0
        for source_window in _source_windows(text, args.source_window_chars):
            window_text = text[source_window.start:source_window.end]
            offsets, token_vectors = embedder.contextualize(window_text)
            sentence_spans = _sentence_spans(window_text, args.semantic_max_chars)
            sentence_vectors = embedder.pool_spans(offsets, token_vectors, sentence_spans)
            semantic_spans, _ = _semantic_spans(window_text, sentence_spans, sentence_vectors, percentile=args.semantic_percentile, min_chars=args.semantic_min_chars, max_chars=args.semantic_max_chars)
            fixed_spans = _fixed_spans(window_text, args.fixed_max_chars, args.fixed_overlap)
            token_count += len(offsets)
            macro_windows += int(np.ceil(max(1, len(offsets) - args.window_tokens) / max(1, args.window_tokens - args.overlap_tokens)) + 1)
            fixed_count += len(fixed_spans)
            semantic_count += len(semantic_spans)
            for boundary, spans in (("fixed", fixed_spans), ("semantic", semantic_spans)):
                late_method, naive_method = f"late_{boundary}", f"naive_{boundary}"
                rows = _chunk_rows(path, relative, result.modality, window_text, spans, late_method)
                for row in rows:
                    row["start_char"] += source_window.start
                    row["end_char"] += source_window.start
                    row["chunk_id"] = f"{relative}::{len(chunk_sets[late_method]) + row['chunk_index']}"
                late = embedder.pool_spans(offsets, token_vectors, spans)
                naive = embedder.encode_texts([row["text"] for row in rows])
                chunk_sets[late_method].extend(rows)
                chunk_sets[naive_method].extend([{**row, "method": naive_method} for row in rows])
                vectors[late_method].extend(late)
                vectors[naive_method].extend(naive)
        diagnostics.append({"relative_path": relative, "source_chars": len(text), "source_windows": len(_source_windows(text, args.source_window_chars)), "token_count": token_count, "macro_windows": macro_windows, "fixed_chunks": fixed_count, "semantic_chunks": semantic_count})
    query_vectors = embedder.encode_query([row["question"] for row in questions])
    dense = {method: np.asarray(value, dtype=np.float32) for method, value in vectors.items()}
    results, summary = _evaluate(chunk_sets, dense, questions, query_vectors, {method: 0.0 for method in chunk_sets})
    summary["config"] = {"data_lake": str(Path(args.data_lake).resolve()), "extensions": sorted(extensions), "embedding_backend": embedder.kind, "embedding_model": args.model, "device": embedder.device, "window_tokens": args.window_tokens, "overlap_tokens": args.overlap_tokens, "source_window_chars": args.source_window_chars, "fixed_max_chars": args.fixed_max_chars, "fixed_overlap": args.fixed_overlap, "semantic_max_chars": args.semantic_max_chars, "semantic_min_chars": args.semantic_min_chars, "semantic_percentile": args.semantic_percentile, "build_seconds": round(time.perf_counter() - started, 3), "document_question_count": len(questions)}
    for method, rows in chunk_sets.items():
        with (output / f"chunks_{method}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.save(output / f"embeddings_{method}.npy", dense[method])
    pd.DataFrame(diagnostics).to_csv(output / "document_diagnostics.csv", index=False)
    results.to_csv(output / "query_results.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path("data/sample_data_lake/Data-Lake")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-lake", default=root)
    parser.add_argument("--questions", nargs="+", default=[Path("data/sample_data_lake/0.Sample_Data.xlsx"), Path("data/sample_data_lake/generated_hard_questions.xlsx"), Path("data/sample_data_lake/generated_sample_data.xlsx")])
    parser.add_argument("--extensions", default=DEFAULT_EXTENSIONS)
    parser.add_argument("--output-dir", default=Path("approaches/approach_3_agentic_rag/outputs/late_chunking_benchmark"))
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-tokens", type=int, default=1024, help="CPU-safe macro window; use 8192 on a capable GPU.")
    parser.add_argument("--overlap-tokens", type=int, default=128)
    parser.add_argument("--source-window-chars", type=int, default=16000, help="Maximum characters processed together for token-level late pooling.")
    parser.add_argument("--fixed-max-chars", type=int, default=1200)
    parser.add_argument("--fixed-overlap", type=int, default=0)
    parser.add_argument("--semantic-max-chars", type=int, default=1200)
    parser.add_argument("--semantic-min-chars", type=int, default=180)
    parser.add_argument("--semantic-percentile", type=float, default=25.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    print(f"Artifacts: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
