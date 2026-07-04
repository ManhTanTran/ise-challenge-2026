"""Question loading, submission generation, and validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import get_config
from .file_indexer import build_file_index
from .formatter import exact_match, normalize_answer
from .solvers import solve_question
from .utils import NOT_ENOUGH_DATA, ensure_dir, load_json, parse_jsonish_list, setup_logging

LOGGER = logging.getLogger(__name__)
SUBMISSION_COLUMNS = ["id", "answer", "evidences"]


def load_questions(question_path: str | Path) -> pd.DataFrame:
    """Load and normalize challenge questions."""

    path = Path(question_path)
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        dataframe = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported question file: {path}")

    column_map = {_normalize_column_name(column): column for column in dataframe.columns}
    id_column = column_map.get("id") or column_map.get("stt")
    question_column = column_map.get("question")
    if question_column is None:
        raise ValueError("Question file must contain a Question column.")

    normalized = pd.DataFrame()
    normalized["id"] = dataframe[id_column] if id_column else range(1, len(dataframe) + 1)
    normalized["question"] = dataframe[question_column].fillna("").astype(str)

    answer_type_column = column_map.get("answer_type")
    data_sources_column = column_map.get("data_sources")
    groundtruth_column = column_map.get("groundtruth")

    normalized["answer_type"] = (
        dataframe[answer_type_column].fillna("").astype(str) if answer_type_column else ""
    )
    normalized["expected_sources"] = (
        dataframe[data_sources_column].apply(parse_jsonish_list) if data_sources_column else [[] for _ in range(len(dataframe))]
    )
    if groundtruth_column:
        normalized["groundtruth"] = dataframe[groundtruth_column].fillna("").astype(str)

    normalized = normalized[normalized["question"].str.strip().astype(bool)].reset_index(drop=True)
    return normalized


def generate_submission(
    question_path: str | Path,
    data_lake_dir: str | Path,
    output_path: str | Path,
    *,
    file_index_path: str | Path | None = None,
    rebuild_index: bool = False,
    use_expected_sources: bool = False,
) -> pd.DataFrame:
    """Run the full pipeline and write submission.csv."""

    setup_logging()
    question_file = Path(question_path)
    data_root = Path(data_lake_dir)
    output = Path(output_path)
    ensure_dir(output.parent)
    index_path = Path(file_index_path) if file_index_path else output.parent / "file_index.json"

    questions = load_questions(question_file)
    if rebuild_index or not index_path.exists():
        file_index = build_file_index(data_root, index_path, extracted_text_dir=output.parent / "extracted_texts")
    else:
        file_index = load_json(index_path, default=[])

    rows: list[dict[str, Any]] = []
    for _, row in tqdm(questions.iterrows(), total=len(questions), desc="Solving questions"):
        solved = solve_question(
            row,
            file_index,
            use_expected_sources=use_expected_sources,
        )
        rows.append(
            {
                "id": solved["id"],
                "answer": normalize_answer(
                    solved.get("answer") or NOT_ENOUGH_DATA,
                    question=str(row["question"]),
                    answer_type=str(row.get("answer_type", "")),
                ),
                "evidences": json.dumps(solved.get("evidences", []), ensure_ascii=False),
            }
        )

    submission = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    validate_submission(submission, data_root)
    submission.to_csv(output, index=False)
    submission.to_csv(output.parent / "predictions.csv", index=False)

    if "groundtruth" in questions.columns:
        write_error_analysis(questions, submission, output.parent / "error_analysis.csv")
    return submission


def validate_submission(submission: pd.DataFrame, data_lake_dir: str | Path | None = None) -> None:
    """Validate challenge submission shape and evidence paths."""

    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}.")
    if submission["id"].isna().any():
        raise ValueError("Submission contains missing ids.")
    if submission["answer"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Submission contains empty answers.")
    data_root = Path(data_lake_dir) if data_lake_dir else None
    for row_number, value in enumerate(submission["evidences"], start=1):
        try:
            evidences = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError as exc:
            raise ValueError(f"Row {row_number} evidences is not valid JSON.") from exc
        if not isinstance(evidences, list):
            raise ValueError(f"Row {row_number} evidences must be a JSON list.")
        if data_root:
            for evidence in evidences:
                if not (data_root / evidence).exists():
                    raise ValueError(f"Evidence file does not exist: {evidence}")


def write_error_analysis(
    questions: pd.DataFrame,
    submission: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    """Write exact-match oriented error analysis when groundtruth exists."""

    merged = questions.merge(submission, on="id", how="left")
    rows = []
    for _, row in merged.iterrows():
        is_exact_type = str(row.get("answer_type", "")).lower() == "exact_match"
        is_correct = exact_match(row.get("answer", ""), row.get("groundtruth", "")) if is_exact_type else ""
        rows.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "predicted_answer": row.get("answer"),
                "groundtruth": row.get("groundtruth"),
                "answer_type": row.get("answer_type"),
                "evidences": row.get("evidences"),
                "is_correct": is_correct,
                "error_type": "" if is_correct is True else ("semantic_or_unjudged" if not is_exact_type else "exact_mismatch"),
            }
        )
    analysis = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output_path, index=False)
    return analysis


def _normalize_column_name(column: Any) -> str:
    return str(column).strip().lower().replace(" ", "_")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate iSE Challenge submission.csv.")
    parser.add_argument("--questions", required=True, help="Path to question Excel/CSV file.")
    parser.add_argument("--data-lake", required=True, help="Path to data lake directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Submission CSV path. Defaults to outputs/submission.csv.",
    )
    parser.add_argument(
        "--file-index",
        default=None,
        help="Existing or target file_index.json path.",
    )
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding the index.")
    parser.add_argument(
        "--use-expected-sources",
        action="store_true",
        help="Use Data Sources hints when present in the question file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = get_config().with_overrides(
        data_lake_dir=args.data_lake,
        question_path=args.questions,
        output_path=args.output,
        file_index_path=args.file_index,
    )
    generate_submission(
        config.question_path,
        config.data_lake_dir,
        config.submission_path,
        file_index_path=config.file_index_path,
        rebuild_index=args.rebuild_index,
        use_expected_sources=args.use_expected_sources,
    )


if __name__ == "__main__":
    main()
