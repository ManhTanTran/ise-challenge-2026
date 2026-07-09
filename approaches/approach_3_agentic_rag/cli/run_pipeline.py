"""CLI entrypoint for approach 3: Agentic Semantic RAG.

Pipeline per question: analyze (Buoc 1) -> hybrid retrieve (Buoc 2) ->
modality readers (Buoc 3) -> LLM reasoning (Buoc 4). Offline indexes
(Buoc 0) are built or loaded once per run. Each question is solved
independently and results are written incrementally, so a crash never
loses finished answers.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ..analysis.analyzer import analyze_question, load_analysis_cache, save_analysis_cache
from ..config import Approach3Config, get_config
from ..core.validation import SUBMISSION_COLUMNS, evidence_json, validate_submission
from ..indexing.build import build_indexes
from ..readers.dispatcher import build_context_blocks
from ..reasoning.reasoner import answer_question, finalize_answer
from ..retrieval.hybrid import retrieve
from ..shared_src.submission import load_questions, write_error_analysis
from ..shared_src.utils import dump_json, ensure_dir, setup_logging

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    *,
    question_path: str | Path,
    data_lake_dir: str | Path,
    output_path: str | Path,
    work_dir: str | Path | None = None,
    file_index_path: str | Path | None = None,
    rebuild_index: bool = False,
    limit: int | None = None,
    config: Approach3Config | None = None,
) -> pd.DataFrame:
    """Run approach 3 end-to-end and write submission plus debug artifacts."""

    setup_logging()
    config = config or get_config()
    output = Path(output_path)
    work = ensure_dir(work_dir or output.parent)
    ensure_dir(output.parent)
    vision_cache_dir = work / "vision_cache"

    questions = load_questions(question_path)
    if limit is not None:
        questions = questions.head(limit).reset_index(drop=True)

    manifest, chunks, vector_index, bm25_index = build_indexes(
        data_lake_dir,
        work,
        config=config,
        file_index_path=file_index_path,
        rebuild=rebuild_index,
    )
    valid_paths = {str(item.get("relative_path", "")) for item in manifest}
    analysis_cache = load_analysis_cache(work)

    submission_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    partial_path = work / "submission_partial.csv"

    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Approach 3"):
        profile = analyze_question(row.to_dict(), config=config, cache=analysis_cache)
        candidates = retrieve(
            profile, manifest, chunks, vector_index, bm25_index, config=config
        )
        blocks = build_context_blocks(
            profile, candidates, config=config, vision_cache_dir=vision_cache_dir
        )
        result = finalize_answer(
            profile,
            answer_question(profile, blocks, config=config),
            valid_paths=valid_paths,
        )

        submission_rows.append(
            {
                "id": profile.question_id,
                "answer": result.answer,
                "evidences": evidence_json(result.evidences),
            }
        )
        profile_rows.append(profile.to_dict())
        debug_rows.append(
            {
                "id": profile.question_id,
                "question": profile.question,
                "answer": result.answer,
                "evidences": result.evidences,
                "strategy": result.strategy,
                "retrieved": [
                    {
                        "path": item.get("relative_path"),
                        "score": round(float(item.get("score", 0.0)), 4),
                        "reason": item.get("reason"),
                        "modality": item.get("modality"),
                    }
                    for item in candidates
                ],
                "readers": [block.metadata.get("reader") for block in blocks],
                "debug": result.debug,
            }
        )
        # Incremental artifacts: a crash mid-run keeps all finished answers.
        pd.DataFrame(submission_rows, columns=SUBMISSION_COLUMNS).to_csv(partial_path, index=False)
        save_analysis_cache(work, analysis_cache)

    submission = pd.DataFrame(submission_rows, columns=SUBMISSION_COLUMNS)
    validate_submission(submission, data_lake_dir)
    submission.to_csv(output, index=False)
    pd.DataFrame(debug_rows).to_csv(work / "predictions_debug.csv", index=False)
    dump_json(profile_rows, work / "question_profiles.json")
    _write_jsonl(debug_rows, work / "retrieval_debug.jsonl")

    if "groundtruth" in questions.columns:
        write_error_analysis(questions, submission, work / "error_analysis.csv")

    LOGGER.info("Wrote approach 3 submission to %s", output)
    return submission


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run approach 3 agentic semantic RAG pipeline.")
    parser.add_argument("--questions", required=True, help="Path to question Excel/CSV file.")
    parser.add_argument("--data-lake", required=True, help="Path to Data-Lake folder.")
    parser.add_argument(
        "--output",
        default="approaches/approach_3_agentic_rag/outputs/submission.csv",
        help="Path to write submission.csv.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for indexes, caches, and debug files. Defaults to output parent.",
    )
    parser.add_argument(
        "--file-index",
        default=None,
        help="Optional existing manifest/file_index.json to reuse.",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild all offline indexes.")
    parser.add_argument("--top-k", type=int, default=None, help="Files to retrieve per question.")
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=None,
        help="Semantic score threshold below which the answer is 'Not enough data'.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="sentence-transformers model name for the vector index.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: solve only the first N questions.")
    parser.add_argument("--no-llm", action="store_true", help="Disable every LLM call (offline mode).")
    parser.add_argument(
        "--no-llm-analysis",
        action="store_true",
        help="Use heuristic-only question analysis (skip the Buoc 1 LLM call).",
    )
    parser.add_argument("--no-vision", action="store_true", help="Disable vision QA for images.")
    parser.add_argument(
        "--no-table-compute",
        action="store_true",
        help="Disable LLM pandas code generation for tables.",
    )
    parser.add_argument(
        "--no-vision-count",
        action="store_true",
        help="Disable code-first counting for 'how many images...' folder questions.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Approach3Config:
    config = get_config()
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.min_relevance is not None:
        config.min_relevance = args.min_relevance
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    if args.no_llm:
        config.use_llm = False
        config.use_llm_analysis = False
        config.use_vision = False
        config.use_table_compute = False
        config.use_vision_count_compute = False
    if args.no_llm_analysis:
        config.use_llm_analysis = False
    if args.no_vision:
        config.use_vision = False
    if args.no_table_compute:
        config.use_table_compute = False
    if args.no_vision_count:
        config.use_vision_count_compute = False
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(
        question_path=args.questions,
        data_lake_dir=args.data_lake,
        output_path=args.output,
        work_dir=args.work_dir,
        file_index_path=args.file_index,
        rebuild_index=args.rebuild_index,
        limit=args.limit,
        config=config_from_args(args),
    )


if __name__ == "__main__":
    main()
