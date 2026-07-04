"""CLI entrypoint for approach 2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ..shared_src.submission import load_questions, write_error_analysis
from ..shared_src.utils import dump_json, ensure_dir, setup_logging

from ..core.indexing import load_or_build_chunks, load_or_build_manifest
from ..core.question_analysis import analyze_question
from ..core.validation import SUBMISSION_COLUMNS, evidence_json, validate_submission
from ..reasoning.engine import answer_question, finalize_answer
from ..retrieval.hybrid import retrieve


LOGGER = logging.getLogger(__name__)


def run_pipeline(
    *,
    question_path: str | Path,
    data_lake_dir: str | Path,
    output_path: str | Path,
    work_dir: str | Path | None = None,
    file_index_path: str | Path | None = None,
    rebuild_index: bool = False,
    top_k: int = 8,
    limit: int | None = None,
    use_llm_analysis: bool = False,
    use_expected_sources: bool = False,
) -> pd.DataFrame:
    """Run approach 2 end-to-end and write submission plus debug artifacts."""

    setup_logging()
    output = Path(output_path)
    work = ensure_dir(work_dir or output.parent)
    ensure_dir(output.parent)

    questions = load_questions(question_path)
    if limit is not None:
        questions = questions.head(limit).reset_index(drop=True)

    manifest = load_or_build_manifest(
        data_lake_dir,
        work,
        file_index_path=file_index_path,
        rebuild=rebuild_index,
    )
    chunks = load_or_build_chunks(manifest, work, rebuild=rebuild_index)

    submission_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Approach 2"):
        row_dict = row.to_dict()
        if not use_expected_sources:
            row_dict["expected_sources"] = []
        profile = analyze_question(row_dict, use_llm=use_llm_analysis)
        candidates = retrieve(profile, manifest, chunks, top_k=top_k)
        result = finalize_answer(profile, answer_question(profile, candidates))

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
                        "score": item.get("score"),
                        "reason": item.get("reason"),
                        "modality": item.get("modality"),
                    }
                    for item in candidates
                ],
                "debug": result.debug,
            }
        )

    submission = pd.DataFrame(submission_rows, columns=SUBMISSION_COLUMNS)
    validate_submission(submission, data_lake_dir)
    submission.to_csv(output, index=False)
    submission.to_csv(output.parent / "submission.csv", index=False)
    pd.DataFrame(debug_rows).to_csv(output.parent / "predictions_debug.csv", index=False)
    dump_json(profile_rows, output.parent / "question_profiles.json")
    _write_jsonl(debug_rows, output.parent / "retrieval_debug.jsonl")

    if "groundtruth" in questions.columns:
        write_error_analysis(questions, submission, output.parent / "error_analysis.csv")

    LOGGER.info("Wrote approach 2 submission to %s", output)
    return submission


def _write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run approach 2 hybrid RAG pipeline.")
    parser.add_argument("--questions", required=True, help="Path to question Excel/CSV file.")
    parser.add_argument("--data-lake", required=True, help="Path to Data-Lake folder.")
    parser.add_argument(
        "--output",
        default="approaches/approach_2_hybrid_rag/outputs/submission.csv",
        help="Path to write submission.csv.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for manifest, chunks, cache, and debug files. Defaults to output parent.",
    )
    parser.add_argument(
        "--file-index",
        default=None,
        help="Optional existing file_index.json to reuse instead of rebuilding.",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild manifest and chunks.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of files to retrieve per question.")
    parser.add_argument("--limit", type=int, default=None, help="Debug: solve only the first N questions.")
    parser.add_argument(
        "--use-llm-analysis",
        action="store_true",
        help="Use OpenRouter for question analysis when OPENROUTER_API_KEY is set.",
    )
    parser.add_argument(
        "--use-expected-sources",
        action="store_true",
        help="Debug only: use sample Data Sources hints from the question file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(
        question_path=args.questions,
        data_lake_dir=args.data_lake,
        output_path=args.output,
        work_dir=args.work_dir,
        file_index_path=args.file_index,
        rebuild_index=args.rebuild_index,
        top_k=args.top_k,
        limit=args.limit,
        use_llm_analysis=args.use_llm_analysis,
        use_expected_sources=args.use_expected_sources,
    )


if __name__ == "__main__":
    main()
