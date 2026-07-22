"""Benchmark current fixed-character chunking against semantic chunking on PDFs.

This tool is intentionally isolated from the production index.  It reads only
PDFs below the supplied sample data lake, creates two chunk sets over identical
extracted text, embeds both sets with the same local dense backend, and compares
retrieval for questions whose expected sources include a PDF.

Run from the repository root::

    python -X utf8 -m approaches.approach_3_agentic_rag.tools.benchmark_pdf_chunking
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from ..config import DEFAULT_EMBEDDING_MODEL
from ..indexing.embedder import TfidfBackend, create_backend
from ..shared_src.file_readers import read_file
from ..shared_src.submission import load_questions
from ..shared_src.utils import chunk_text, ensure_dir, normalize_for_match, normalize_spaces

DEFAULT_DATA_LAKE = Path("data/sample_data_lake/Data-Lake")
DEFAULT_QUESTION_FILES = (
    Path("data/sample_data_lake/0.Sample_Data.xlsx"),
    Path("data/sample_data_lake/generated_hard_questions.xlsx"),
    Path("data/sample_data_lake/generated_sample_data.xlsx"),
)
DEFAULT_OUTPUT_DIR = Path("approaches/approach_3_agentic_rag/outputs/pdf_chunking_benchmark")
DEFAULT_EXTENSIONS = ".pdf"
K_VALUES = (1, 3, 5, 8)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|\n+")


@dataclass(slots=True)
class SemanticChunkDiagnostics:
    sentence_count: int
    pair_count: int
    threshold: float | None
    percentile: float
    semantic_boundaries: int
    hard_splits: int
    tiny_merges: int
    min_similarity: float | None
    median_similarity: float | None
    max_similarity: float | None


def split_sentences(text: str) -> list[str]:
    """Split prose into compact sentence units while preserving line boundaries."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        sentence
        for part in _SENTENCE_BOUNDARY.split(normalized)
        if (sentence := normalize_spaces(part))
    ]


def semantic_chunk_text(
    text: str,
    *,
    embed_fn: Callable[[list[str]], np.ndarray],
    max_chars: int = 1200,
    min_chars: int = 180,
    percentile: float = 25.0,
) -> tuple[list[str], SemanticChunkDiagnostics]:
    """Chunk prose at low-similarity boundaries, then apply size guards.

    Sentence vectors are used only to choose boundaries.  Returned chunks are
    embedded again by the benchmark, keeping boundary selection and indexing
    independent for a clean Phase-1 comparison.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if min_chars < 0 or min_chars > max_chars:
        raise ValueError("min_chars must be between 0 and max_chars")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    # HTML/PDF extraction can yield a punctuation-free "sentence" much longer
    # than an embedding endpoint accepts. Split it before sentence embedding,
    # not only after semantic grouping, so providers never truncate/reject it.
    sentences = [
        piece
        for sentence in split_sentences(text)
        for piece in _hard_split(sentence, max_chars=max_chars)
    ]
    if not sentences:
        return [], _empty_diagnostics(percentile)

    similarities = np.zeros(0, dtype=np.float32)
    threshold: float | None = None
    boundary_after: set[int] = set()
    if len(sentences) > 1:
        vectors = _normalize_dense(np.asarray(embed_fn(sentences), dtype=np.float32))
        if vectors.shape[0] != len(sentences):
            raise ValueError("embed_fn returned a different number of vectors than sentences")
        similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)
        threshold = float(np.percentile(similarities, percentile))
        # Strict comparison avoids splitting every pair when all similarities tie.
        boundary_after = {
            index for index, similarity in enumerate(similarities) if float(similarity) < threshold
        }

    groups: list[list[int]] = []
    current: list[int] = []
    for index in range(len(sentences)):
        current.append(index)
        if index in boundary_after:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    groups, tiny_merges = _merge_tiny_groups(
        groups,
        sentences,
        similarities,
        min_chars=min_chars,
        max_chars=max_chars,
    )

    chunks: list[str] = []
    hard_splits = 0
    for group in groups:
        packed = _pack_to_max_chars([sentences[index] for index in group], max_chars=max_chars)
        hard_splits += max(0, len(packed) - 1)
        chunks.extend(packed)

    diagnostics = SemanticChunkDiagnostics(
        sentence_count=len(sentences),
        pair_count=len(similarities),
        threshold=threshold,
        percentile=percentile,
        semantic_boundaries=len(boundary_after),
        hard_splits=hard_splits,
        tiny_merges=tiny_merges,
        min_similarity=_safe_stat(similarities, np.min),
        median_similarity=_safe_stat(similarities, np.median),
        max_similarity=_safe_stat(similarities, np.max),
    )
    return chunks, diagnostics


def _merge_tiny_groups(
    groups: list[list[int]],
    sentences: list[str],
    similarities: np.ndarray,
    *,
    min_chars: int,
    max_chars: int,
) -> tuple[list[list[int]], int]:
    if min_chars == 0 or len(groups) < 2:
        return groups, 0

    merged = [list(group) for group in groups]
    merge_count = 0
    index = 0
    while index < len(merged):
        group = merged[index]
        if len(_join_group(group, sentences)) >= min_chars:
            index += 1
            continue

        candidates: list[tuple[float, int]] = []
        if index > 0 and len(_join_group(merged[index - 1] + group, sentences)) <= max_chars:
            left_boundary = group[0] - 1
            left_similarity = float(similarities[left_boundary]) if left_boundary >= 0 else -math.inf
            candidates.append((left_similarity, index - 1))
        if index + 1 < len(merged) and len(_join_group(group + merged[index + 1], sentences)) <= max_chars:
            right_boundary = group[-1]
            right_similarity = (
                float(similarities[right_boundary])
                if right_boundary < len(similarities)
                else -math.inf
            )
            candidates.append((right_similarity, index + 1))
        if not candidates:
            index += 1
            continue

        _, neighbor = max(candidates, key=lambda item: item[0])
        if neighbor < index:
            merged[neighbor].extend(group)
            del merged[index]
            index = max(0, neighbor)
        else:
            merged[index].extend(merged[neighbor])
            del merged[neighbor]
        merge_count += 1
    return merged, merge_count


def _pack_to_max_chars(sentences: list[str], *, max_chars: int) -> list[str]:
    packed: list[str] = []
    current = ""
    for sentence in sentences:
        for piece in _hard_split(sentence, max_chars=max_chars):
            candidate = f"{current} {piece}".strip() if current else piece
            if current and len(candidate) > max_chars:
                packed.append(current)
                current = piece
            else:
                current = candidate
    if current:
        packed.append(current)
    return packed


def _hard_split(text: str, *, max_chars: int) -> list[str]:
    compact = normalize_spaces(text)
    pieces: list[str] = []
    while len(compact) > max_chars:
        split_at = compact.rfind(" ", 0, max_chars + 1)
        if split_at <= 0:
            split_at = max_chars
        pieces.append(compact[:split_at].strip())
        compact = compact[split_at:].strip()
    if compact:
        pieces.append(compact)
    return pieces


def _join_group(group: list[int], sentences: list[str]) -> str:
    return " ".join(sentences[index] for index in group)


def _empty_diagnostics(percentile: float) -> SemanticChunkDiagnostics:
    return SemanticChunkDiagnostics(0, 0, None, percentile, 0, 0, 0, None, None, None)


def _safe_stat(values: np.ndarray, fn: Callable[[np.ndarray], Any]) -> float | None:
    return float(fn(values)) if len(values) else None


def _normalize_dense(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _encode_batched(backend: Any, texts: list[str], batch_size: int = 64) -> np.ndarray:
    batches = [
        np.asarray(backend.encode(texts[start : start + batch_size]), dtype=np.float32)
        for start in range(0, len(texts), batch_size)
    ]
    return _normalize_dense(np.vstack(batches)) if batches else np.zeros((0, 1), dtype=np.float32)


def build_chunk_sets(
    data_lake: Path,
    backend: Any,
    *,
    extensions: set[str] | None = None,
    extraction_cache: Path | None = None,
    fixed_max_chars: int,
    fixed_overlap: int,
    semantic_max_chars: int,
    semantic_min_chars: int,
    semantic_percentile: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    chunk_sets: dict[str, list[dict[str, Any]]] = {"fixed": [], "semantic": []}
    diagnostics: list[dict[str, Any]] = []
    selected_extensions = extensions or {".pdf"}
    document_paths = sorted(
        path
        for path in data_lake.rglob("*")
        if path.is_file() and path.suffix.lower() in selected_extensions
    )
    if not document_paths:
        raise FileNotFoundError(
            f"No files with extensions {sorted(selected_extensions)} found below {data_lake}"
        )

    for path in document_paths:
        relative_path = path.relative_to(data_lake).as_posix()
        result = read_file(
            path,
            cache_dir=extraction_cache,
            data_lake_dir=data_lake,
            use_cache=True,
        )
        if result.error or not result.content.strip():
            diagnostics.append(
                {
                    "relative_path": relative_path,
                    "extension": path.suffix.lower(),
                    "parse_error": result.error or "empty extracted text",
                }
            )
            continue
        text = result.content
        fixed_parts = chunk_text(text, max_chars=fixed_max_chars, overlap=fixed_overlap)
        semantic_parts, semantic_diag = semantic_chunk_text(
            text,
            embed_fn=lambda sentences: _encode_batched(backend, sentences),
            max_chars=semantic_max_chars,
            min_chars=semantic_min_chars,
            percentile=semantic_percentile,
        )
        for method, parts in (("fixed", fixed_parts), ("semantic", semantic_parts)):
            for chunk_index, part in enumerate(parts):
                chunk_sets[method].append(
                    {
                        "chunk_id": f"{relative_path}::{chunk_index}",
                        "relative_path": relative_path,
                        "filename": path.name,
                        "extension": path.suffix.lower(),
                        "modality": result.modality,
                        "chunk_index": chunk_index,
                        "text": part,
                        "char_count": len(part),
                    }
                )
        diagnostics.append(
            {
                "relative_path": relative_path,
                "extension": path.suffix.lower(),
                "parse_error": None,
                "source_chars": len(text),
                "fixed_chunks": len(fixed_parts),
                "semantic_chunks": len(semantic_parts),
                **asdict(semantic_diag),
            }
        )
    return chunk_sets, diagnostics


def load_document_questions(
    paths: Iterable[Path],
    *,
    extensions: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected_extensions = extensions or {".pdf"}
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        questions = load_questions(path)
        for _, row in questions.iterrows():
            expected = [str(source) for source in (row.get("expected_sources") or [])]
            expected_documents = [
                source
                for source in expected
                if Path(source.replace("*", "placeholder")).suffix.lower() in selected_extensions
            ]
            if not expected_documents:
                continue
            rows.append(
                {
                    "dataset": path.stem,
                    "id": row.get("id"),
                    "question": str(row.get("question", "")),
                    "groundtruth": str(row.get("groundtruth", "")),
                    "expected_sources": expected_documents,
                }
            )
    return rows


def evaluate_retrieval(
    chunk_sets: dict[str, list[dict[str, Any]]],
    questions: list[dict[str, Any]],
    backend: Any,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    query_vectors = _encode_batched(backend, [row["question"] for row in questions])
    result_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    chunk_embeddings: dict[str, np.ndarray] = {}

    for method, chunks in chunk_sets.items():
        started = time.perf_counter()
        chunk_vectors = _encode_batched(backend, [str(chunk["text"]) for chunk in chunks])
        chunk_embeddings[method] = chunk_vectors
        encode_seconds = time.perf_counter() - started
        paths = [str(chunk["relative_path"]) for chunk in chunks]
        unique_paths = sorted(set(paths))

        for query_index, question in enumerate(questions):
            scores = chunk_vectors @ query_vectors[query_index]
            chunk_order = np.argsort(-scores)
            expected = {_norm_source(source) for source in question["expected_sources"]}
            first_chunk_rank = next(
                (
                    rank
                    for rank, chunk_index in enumerate(chunk_order, start=1)
                    if _source_matches(paths[int(chunk_index)], expected)
                ),
                None,
            )
            groundtruth = normalize_for_match(question["groundtruth"])
            groundtruth_is_searchable = len(groundtruth) >= 4 and not groundtruth.isdigit()
            first_answer_chunk_rank = (
                next(
                    (
                        rank
                        for rank, chunk_index in enumerate(chunk_order, start=1)
                        if groundtruth
                        in normalize_for_match(str(chunks[int(chunk_index)]["text"]))
                    ),
                    None,
                )
                if groundtruth_is_searchable
                else None
            )

            file_scores = {
                path: max(float(scores[index]) for index, candidate in enumerate(paths) if candidate == path)
                for path in unique_paths
            }
            file_order = sorted(file_scores, key=file_scores.get, reverse=True)
            first_file_rank = next(
                (
                    rank
                    for rank, path in enumerate(file_order, start=1)
                    if _source_matches(path, expected)
                ),
                None,
            )
            top_index = int(chunk_order[0])
            chunk_source_recalls: dict[int, float] = {}
            file_recalls: dict[int, float] = {}
            file_fully_covered: dict[int, bool] = {}
            for k in K_VALUES:
                top_chunk_sources = {
                    _norm_source(paths[int(index)]) for index in chunk_order[:k]
                }
                top_file_sources = {_norm_source(path) for path in file_order[:k]}
                chunk_found = sum(
                    1
                    for source in expected
                    if any(_source_matches(candidate, {source}) for candidate in top_chunk_sources)
                )
                file_found = sum(
                    1
                    for source in expected
                    if any(_source_matches(candidate, {source}) for candidate in top_file_sources)
                )
                chunk_source_recalls[k] = chunk_found / len(expected)
                file_recalls[k] = file_found / len(expected)
                file_fully_covered[k] = file_found == len(expected)
            result_rows.append(
                {
                    "method": method,
                    "dataset": question["dataset"],
                    "id": question["id"],
                    "question": question["question"],
                    "expected_sources": json.dumps(question["expected_sources"], ensure_ascii=False),
                    "first_correct_chunk_rank": first_chunk_rank,
                    "first_correct_file_rank": first_file_rank,
                    "first_answer_text_chunk_rank": first_answer_chunk_rank,
                    "chunk_mrr": 1.0 / first_chunk_rank if first_chunk_rank else 0.0,
                    "file_mrr": 1.0 / first_file_rank if first_file_rank else 0.0,
                    "answer_text_mrr": (
                        1.0 / first_answer_chunk_rank if first_answer_chunk_rank else None
                    ),
                    **{
                        f"chunk_hit@{k}": bool(first_chunk_rank and first_chunk_rank <= k)
                        for k in K_VALUES
                    },
                    **{
                        f"chunk_source_recall@{k}": chunk_source_recalls[k]
                        for k in K_VALUES
                    },
                    **{f"file_recall@{k}": file_recalls[k] for k in K_VALUES},
                    **{
                        f"file_fully_covered@{k}": file_fully_covered[k]
                        for k in K_VALUES
                    },
                    "top_chunk_source": paths[top_index],
                    "top_8_chunk_sources": json.dumps(
                        [paths[int(index)] for index in chunk_order[:8]], ensure_ascii=False
                    ),
                    "top_8_files": json.dumps(file_order[:8], ensure_ascii=False),
                    "top_chunk_score": round(float(scores[top_index]), 6),
                    "top_chunk_preview": normalize_spaces(str(chunks[top_index]["text"]))[:240],
                }
            )

        lengths = np.asarray([len(str(chunk["text"])) for chunk in chunks], dtype=np.float32)
        method_rows = [row for row in result_rows if row["method"] == method]
        answer_text_mrrs = [
            row["answer_text_mrr"]
            for row in method_rows
            if row["answer_text_mrr"] is not None
        ]
        summary[method] = {
            "documents": len(unique_paths),
            "chunks": len(chunks),
            "mean_chars": round(float(np.mean(lengths)), 2) if len(lengths) else 0.0,
            "p95_chars": round(float(np.percentile(lengths, 95)), 2) if len(lengths) else 0.0,
            "max_chars": int(np.max(lengths)) if len(lengths) else 0,
            "chunk_encode_seconds": round(encode_seconds, 3),
            "questions": len(questions),
            "chunk_mrr": round(float(np.mean([row["chunk_mrr"] for row in method_rows])), 4)
            if method_rows
            else None,
            "file_mrr": round(float(np.mean([row["file_mrr"] for row in method_rows])), 4)
            if method_rows
            else None,
            "answer_text_questions": len(answer_text_mrrs),
            "answer_text_mrr": round(float(np.mean(answer_text_mrrs)), 4)
            if answer_text_mrrs
            else None,
            **{
                f"chunk_hit@{k}": round(
                    float(np.mean([row[f"chunk_hit@{k}"] for row in method_rows])), 4
                )
                if method_rows
                else None
                for k in K_VALUES
            },
            **{
                f"chunk_source_recall@{k}": round(
                    float(np.mean([row[f"chunk_source_recall@{k}"] for row in method_rows])),
                    4,
                )
                if method_rows
                else None
                for k in K_VALUES
            },
            **{
                f"file_recall@{k}": round(
                    float(np.mean([row[f"file_recall@{k}"] for row in method_rows])), 4
                )
                if method_rows
                else None
                for k in K_VALUES
            },
            **{
                f"file_fully_covered@{k}": round(
                    float(np.mean([row[f"file_fully_covered@{k}"] for row in method_rows])),
                    4,
                )
                if method_rows
                else None
                for k in K_VALUES
            },
        }
    return pd.DataFrame(result_rows), summary, chunk_embeddings, query_vectors


def _norm_source(path: str) -> str:
    return str(path).replace("\\", "/").strip().lower()


def _source_matches(candidate: str, expected: set[str]) -> bool:
    normalized = _norm_source(candidate)
    basename = Path(normalized).name
    normalized_expected = {_norm_source(source) for source in expected}
    return any(
        normalized == source or basename == Path(source).name
        for source in normalized_expected
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_embedding_artifacts(
    output_dir: Path,
    chunk_sets: dict[str, list[dict[str, Any]]],
    embeddings: dict[str, np.ndarray],
    query_vectors: np.ndarray,
    questions: list[dict[str, Any]],
    *,
    preview_dimensions: int = 12,
) -> None:
    preview_rows: list[dict[str, Any]] = []
    for method, vectors in embeddings.items():
        np.save(output_dir / f"embeddings_{method}.npy", vectors)
        chunks = chunk_sets[method]
        manifest_rows = [
            {
                "embedding_row": row_index,
                "method": method,
                "chunk_id": chunk.get("chunk_id"),
                "relative_path": chunk.get("relative_path"),
                "chunk_index": chunk.get("chunk_index"),
                "char_count": chunk.get("char_count"),
            }
            for row_index, chunk in enumerate(chunks)
        ]
        pd.DataFrame(manifest_rows).to_csv(
            output_dir / f"embedding_rows_{method}.csv", index=False
        )
        for row_index in range(min(10, len(chunks))):
            vector = vectors[row_index]
            preview_rows.append(
                {
                    **manifest_rows[row_index],
                    "l2_norm": round(float(np.linalg.norm(vector)), 6),
                    **{
                        f"dim_{dimension}": round(float(vector[dimension]), 7)
                        for dimension in range(min(preview_dimensions, len(vector)))
                    },
                }
            )

    np.save(output_dir / "query_embeddings.npy", query_vectors)
    pd.DataFrame(
        [
            {
                "embedding_row": row_index,
                "dataset": question["dataset"],
                "id": question["id"],
                "question": question["question"],
            }
            for row_index, question in enumerate(questions)
        ]
    ).to_csv(output_dir / "query_embedding_rows.csv", index=False)
    pd.DataFrame(preview_rows).to_csv(output_dir / "embedding_preview.csv", index=False)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    extensions = _parse_extensions(args.extensions)
    backend = create_backend(args.embedding_model)
    if isinstance(backend, TfidfBackend):
        raise RuntimeError(
            "PDF semantic chunking requires a dense backend. Install fastembed or "
            "sentence-transformers; TF-IDF fallback is not accepted for this benchmark."
        )

    chunk_started = time.perf_counter()
    chunk_sets, diagnostics = build_chunk_sets(
        Path(args.data_lake),
        backend,
        extensions=extensions,
        extraction_cache=output_dir / "text_cache",
        fixed_max_chars=args.fixed_max_chars,
        fixed_overlap=args.fixed_overlap,
        semantic_max_chars=args.semantic_max_chars,
        semantic_min_chars=args.semantic_min_chars,
        semantic_percentile=args.semantic_percentile,
    )
    chunk_seconds = time.perf_counter() - chunk_started
    questions = load_document_questions(
        [Path(path) for path in args.questions], extensions=extensions
    )
    if not questions:
        raise RuntimeError(
            f"No questions with expected sources in {sorted(extensions)} were found."
        )

    results, summary, embeddings, query_vectors = evaluate_retrieval(
        chunk_sets, questions, backend
    )
    summary["config"] = {
        "data_lake": str(Path(args.data_lake).resolve()),
        "extensions": sorted(extensions),
        "embedding_backend": backend.kind,
        "embedding_model": getattr(backend, "model_name", args.embedding_model),
        "fixed_max_chars": args.fixed_max_chars,
        "fixed_overlap": args.fixed_overlap,
        "semantic_max_chars": args.semantic_max_chars,
        "semantic_min_chars": args.semantic_min_chars,
        "semantic_percentile": args.semantic_percentile,
        "chunk_build_seconds": round(chunk_seconds, 3),
        "document_question_count": len(questions),
        "note": "Only questions backed by the selected document extensions are evaluated.",
    }

    _write_jsonl(output_dir / "chunks_fixed.jsonl", chunk_sets["fixed"])
    _write_jsonl(output_dir / "chunks_semantic.jsonl", chunk_sets["semantic"])
    _write_embedding_artifacts(
        output_dir,
        chunk_sets,
        embeddings,
        query_vectors,
        questions,
    )
    pd.DataFrame(diagnostics).to_csv(output_dir / "document_diagnostics.csv", index=False)
    results.to_csv(output_dir / "query_results.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _parse_extensions(value: str) -> set[str]:
    extensions = {
        extension if extension.startswith(".") else f".{extension}"
        for part in value.split(",")
        if (extension := part.strip().lower())
    }
    if not extensions:
        raise ValueError("At least one extension is required")
    return extensions


def parse_args(
    argv: list[str] | None = None,
    *,
    default_extensions: str = DEFAULT_EXTENSIONS,
    default_output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fixed-character vs semantic chunking on selected "
            "text-like sample files."
        )
    )
    parser.add_argument("--data-lake", default=str(DEFAULT_DATA_LAKE))
    parser.add_argument(
        "--questions",
        action="append",
        default=None,
        help="Question workbook; repeat the flag for multiple files.",
    )
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument(
        "--extensions",
        default=default_extensions,
        help="Comma-separated extensions included in corpus and ground-truth sources.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--fixed-max-chars", type=int, default=2200)
    parser.add_argument("--fixed-overlap", type=int, default=250)
    parser.add_argument("--semantic-max-chars", type=int, default=1200)
    parser.add_argument("--semantic-min-chars", type=int, default=180)
    parser.add_argument("--semantic-percentile", type=float, default=25.0)
    args = parser.parse_args(argv)
    if args.questions is None:
        args.questions = [str(path) for path in DEFAULT_QUESTION_FILES]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_benchmark(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
