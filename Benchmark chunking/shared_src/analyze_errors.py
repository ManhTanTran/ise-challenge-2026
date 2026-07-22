"""Summarize prediction and error-analysis outputs for iteration planning."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .submission import load_questions
from .utils import NOT_ENOUGH_DATA, ensure_dir, load_json, normalize_for_match, parse_jsonish_list


def analyze_outputs(
    *,
    error_analysis_path: str | Path,
    output_dir: str | Path,
    file_index_path: str | Path | None = None,
    questions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create error-analysis summary CSVs and a Markdown report."""

    error_path = Path(error_analysis_path)
    if not error_path.exists():
        raise FileNotFoundError(f"error_analysis.csv not found: {error_path}")

    output = ensure_dir(output_dir)
    dataframe = pd.read_csv(error_path)
    index_map = _load_index_map(file_index_path)
    question_hints = _load_question_hints(questions_path)

    enriched = dataframe.copy()
    enriched["evidence_list"] = enriched["evidences"].apply(parse_evidences)
    enriched["evidence_count"] = enriched["evidence_list"].apply(len)
    enriched["evidence_modalities"] = enriched["evidence_list"].apply(
        lambda evidences: ",".join(_modalities_for_evidences(evidences, index_map))
    )
    enriched["primary_modality"] = enriched["evidence_modalities"].apply(
        lambda value: value.split(",")[0] if value else "none"
    )
    enriched["predicted_not_enough"] = enriched["predicted_answer"].apply(is_not_enough)
    enriched["groundtruth_not_enough"] = enriched["groundtruth"].apply(is_not_enough)
    enriched["is_correct_bool"] = enriched["is_correct"].apply(to_bool_or_none)
    enriched["expected_sources"] = enriched["id"].apply(
        lambda question_id: question_hints.get(str(question_id), [])
    )
    enriched["evidence_hit_expected"] = enriched.apply(
        lambda row: evidence_hits_expected(row["evidence_list"], row["expected_sources"]),
        axis=1,
    )
    enriched["analysis_bucket"] = enriched.apply(bucket_row, axis=1)
    enriched["question_family"] = enriched["question"].apply(classify_question_family)

    enriched_path = output / "error_analysis_enriched.csv"
    enriched.drop(columns=["evidence_list"]).to_csv(enriched_path, index=False)

    exact = enriched[enriched["answer_type"].fillna("").str.lower() == "exact_match"].copy()
    exact_mismatches = exact[exact["is_correct_bool"] != True].copy()  # noqa: E712
    exact_mismatches.drop(columns=["evidence_list"]).to_csv(
        output / "exact_mismatches.csv",
        index=False,
    )

    not_enough_rows = enriched[enriched["predicted_not_enough"] | enriched["groundtruth_not_enough"]]
    not_enough_rows.drop(columns=["evidence_list"]).to_csv(output / "not_enough_cases.csv", index=False)

    summary = build_summary(enriched)
    summary_tables = write_summary_tables(enriched, output)
    report_path = output / "error_report.md"
    report_path.write_text(
        render_markdown_report(summary, summary_tables, exact_mismatches),
        encoding="utf-8",
    )

    return {
        "rows": len(enriched),
        "summary": summary,
        "report_path": str(report_path),
        "enriched_path": str(enriched_path),
    }


def parse_evidences(value: Any) -> list[str]:
    """Parse evidence JSON list strings safely."""

    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return parse_jsonish_list(text)


def is_not_enough(value: Any) -> bool:
    return normalize_for_match(value) == normalize_for_match(NOT_ENOUGH_DATA)


def to_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def evidence_hits_expected(evidences: list[str], expected_sources: list[str]) -> bool | None:
    """Return whether predicted evidences overlap expected sources, if available."""

    if not expected_sources:
        return None
    normalized_evidences = [path.replace("\\", "/") for path in evidences]
    canonical_evidences = [_canonical_path_for_match(path) for path in normalized_evidences]
    for expected in expected_sources:
        pattern = str(expected).replace("\\", "/")
        canonical_pattern = _canonical_path_for_match(pattern)
        for evidence, canonical_evidence in zip(normalized_evidences, canonical_evidences):
            if (
                evidence == pattern
                or fnmatch.fnmatch(evidence, pattern)
                or canonical_evidence == canonical_pattern
                or fnmatch.fnmatch(canonical_evidence, canonical_pattern)
            ):
                return True
    return False


def _canonical_path_for_match(path: str) -> str:
    """Normalize filenames for loose source matching across accents/spaces."""

    return normalize_for_match(path).replace(" ", "")


def bucket_row(row: pd.Series) -> str:
    """Assign a high-level error bucket for prioritization."""

    answer_type = str(row.get("answer_type", "")).lower()
    is_correct = row.get("is_correct_bool")
    if answer_type == "exact_match" and is_correct is True:
        return "exact_correct"
    if row.get("predicted_not_enough") and not row.get("groundtruth_not_enough"):
        return "not_enough_when_answer_exists"
    if not row.get("predicted_not_enough") and row.get("groundtruth_not_enough"):
        return "answered_when_should_not"
    if row.get("evidence_hit_expected") is False:
        return "evidence_miss"
    if answer_type == "exact_match":
        return "exact_value_mismatch"
    return "llm_judge_review"


def classify_question_family(question: Any) -> str:
    """Coarse question family from wording."""

    normalized = normalize_for_match(question)
    if any(term in normalized for term in ["correlation", "average", "mean", "sum", "count", "how many", "bao nhieu", "trung binh"]):
        return "calculation"
    if any(term in normalized for term in ["image", "anh", "visible", "digit", "blue"]):
        return "image"
    if any(term in normalized for term in ["audio", "meeting", "workshop"]):
        return "audio"
    if any(term in normalized for term in ["yes", "no", "did", "co phai"]):
        return "yes_no"
    return "document_open"


def build_summary(enriched: pd.DataFrame) -> dict[str, Any]:
    exact = enriched[enriched["answer_type"].fillna("").str.lower() == "exact_match"]
    correct_exact = int((exact["is_correct_bool"] == True).sum())  # noqa: E712
    exact_total = int(len(exact))
    return {
        "total_rows": int(len(enriched)),
        "exact_total": exact_total,
        "exact_correct": correct_exact,
        "exact_accuracy": round(correct_exact / exact_total, 4) if exact_total else None,
        "llm_judge_total": int((enriched["answer_type"].fillna("").str.lower() == "llm_judge").sum()),
        "not_enough_predictions": int(enriched["predicted_not_enough"].sum()),
        "empty_evidence_rows": int((enriched["evidence_count"] == 0).sum()),
    }


def write_summary_tables(enriched: pd.DataFrame, output: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "bucket_summary": _value_counts(enriched, "analysis_bucket"),
        "modality_summary": _value_counts(enriched, "primary_modality"),
        "question_family_summary": _value_counts(enriched, "question_family"),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return tables


def render_markdown_report(
    summary: dict[str, Any],
    summary_tables: dict[str, pd.DataFrame],
    exact_mismatches: pd.DataFrame,
) -> str:
    lines = [
        "# Error Analysis Report",
        "",
        "## Summary",
        "",
        f"- Total rows: {summary['total_rows']}",
        f"- Exact-match: {summary['exact_correct']}/{summary['exact_total']} correct",
        f"- Exact accuracy: {summary['exact_accuracy']}",
        f"- LLM-judge rows: {summary['llm_judge_total']}",
        f"- Not-enough predictions: {summary['not_enough_predictions']}",
        f"- Empty-evidence rows: {summary['empty_evidence_rows']}",
        "",
    ]
    for title, table in summary_tables.items():
        lines.extend([f"## {title.replace('_', ' ').title()}", "", dataframe_to_markdown(table), ""])

    lines.extend(["## Exact Mismatches", ""])
    if exact_mismatches.empty:
        lines.append("No exact-match mismatches.")
    else:
        cols = ["id", "analysis_bucket", "question_family", "predicted_answer", "groundtruth", "primary_modality"]
        lines.append(dataframe_to_markdown(exact_mismatches[cols]))
    lines.append("")
    lines.extend(
        [
            "## Suggested Iteration Order",
            "",
            "1. Fix evidence misses before answer-format issues.",
            "2. Improve deterministic solvers for calculation/SQL/table rows.",
            "3. Improve image OCR or add task-specific image prompts for visual counting/color questions.",
            "4. Review LLM-judge rows manually or with a separate judge prompt.",
        ]
    )
    return "\n".join(lines)


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Render a small dataframe as a GitHub-flavored Markdown table without extra deps."""

    if dataframe.empty:
        return "_No rows._"
    columns = [str(column) for column in dataframe.columns]
    rows = []
    for _, row in dataframe.iterrows():
        rows.append([_markdown_cell(row[column]) for column in dataframe.columns])
    header = "| " + " | ".join(_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _markdown_cell(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _value_counts(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = dataframe[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, "count"]
    counts["percent"] = (counts["count"] / len(dataframe) * 100).round(2) if len(dataframe) else 0
    return counts


def _load_index_map(file_index_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not file_index_path:
        return {}
    rows = load_json(file_index_path, default=[])
    return {str(row.get("relative_path", "")).replace("\\", "/"): row for row in rows}


def _load_question_hints(questions_path: str | Path | None) -> dict[str, list[str]]:
    if not questions_path:
        return {}
    questions = load_questions(questions_path)
    if "expected_sources" not in questions.columns:
        return {}
    return {
        str(row["id"]): [str(item).replace("\\", "/") for item in row["expected_sources"]]
        for _, row in questions.iterrows()
    }


def _modalities_for_evidences(evidences: list[str], index_map: dict[str, dict[str, Any]]) -> list[str]:
    modalities: list[str] = []
    for evidence in evidences:
        modality = index_map.get(evidence.replace("\\", "/"), {}).get("modality", "unknown")
        if modality not in modalities:
            modalities.append(str(modality))
    return modalities


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize iSE error_analysis.csv outputs.")
    parser.add_argument("--error-analysis", required=True, help="Path to error_analysis.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for report and summary CSVs.")
    parser.add_argument("--file-index", default=None, help="Optional file_index.json for modality summaries.")
    parser.add_argument("--questions", default=None, help="Optional question file for expected-source checks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = analyze_outputs(
        error_analysis_path=args.error_analysis,
        output_dir=args.output_dir,
        file_index_path=args.file_index,
        questions_path=args.questions,
    )
    print(f"Wrote report: {result['report_path']}")


if __name__ == "__main__":
    main()
