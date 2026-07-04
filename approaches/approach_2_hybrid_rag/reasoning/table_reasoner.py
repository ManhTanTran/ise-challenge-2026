"""Deterministic table reasoning for exact-match style questions."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..shared_src.file_readers import load_table_file
from ..shared_src.formatter import format_float
from ..shared_src.utils import NOT_ENOUGH_DATA, normalize_for_match, normalize_spaces

from ..core.models import QuestionProfile


def try_answer_tables(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
) -> tuple[str, list[str], dict[str, Any]]:
    """Try to answer with pandas before asking an LLM."""

    table_candidates = [
        item
        for item in candidates
        if item.get("modality") == "table" or str(item.get("extension", "")).lower() == ".sql"
    ]
    if not table_candidates:
        return NOT_ENOUGH_DATA, [], {}

    compare_files = _asks_which_file_extreme(profile.question)
    per_file_results = []
    for candidate in _ordered_table_candidates(profile, table_candidates):
        path = candidate.get("absolute_path")
        if not path:
            continue
        try:
            tables = load_table_file(path)
            answer, debug = _answer_from_tables(profile, tables)
        except Exception as exc:
            per_file_results.append(
                {"path": candidate.get("relative_path"), "answer": NOT_ENOUGH_DATA, "error": str(exc)}
            )
            continue
        per_file_results.append(
            {
                "path": candidate.get("relative_path", ""),
                "answer": answer,
                "debug": debug,
                "candidate": candidate,
            }
        )
        if answer != NOT_ENOUGH_DATA and not compare_files:
            return str(answer), [candidate.get("relative_path", "")], {"table_attempts": per_file_results}

    valid = [item for item in per_file_results if item.get("answer") != NOT_ENOUGH_DATA]
    if not valid:
        return NOT_ENOUGH_DATA, [], {"table_attempts": per_file_results}

    normalized = normalize_for_match(profile.question)
    if "which file" in normalized and any(term in normalized for term in ["highest", "largest", "max"]):
        best = max(valid, key=lambda item: _numeric_value(item.get("answer")))
        return str(Path(best["path"]).name), [best["path"]], {"table_attempts": per_file_results}
    if "which file" in normalized and any(term in normalized for term in ["lowest", "smallest", "min"]):
        best = min(valid, key=lambda item: _numeric_value(item.get("answer")))
        return str(Path(best["path"]).name), [best["path"]], {"table_attempts": per_file_results}

    best = valid[0]
    return str(best["answer"]), [best["path"]], {"table_attempts": per_file_results}


def _ordered_table_candidates(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = normalize_for_match(profile.question)
    keywords = [normalize_for_match(item) for item in profile.keywords]

    def score(candidate: dict[str, Any]) -> tuple[int, float]:
        text = " ".join(
            [
                normalize_for_match(candidate.get("relative_path", "")),
                normalize_for_match(candidate.get("filename", "")),
                normalize_for_match(" ".join(str(item) for item in candidate.get("sheet_names", []) or [])),
                normalize_for_match(" ".join(str(item) for item in candidate.get("columns", []) or [])),
                normalize_for_match(" ".join(str(chunk.get("text", "")) for chunk in candidate.get("chunks", []) or [])),
            ]
        )
        value = sum(1 for keyword in keywords if keyword and keyword in text)
        if "acetylproteomics" in normalized and "acetyl" in text:
            value += 100
        if "phosphoproteomics" in normalized and "phospho" in text:
            value += 100
        if "significant gene" in normalized and "significant genes" in text:
            value += 50
        return value, float(candidate.get("score", 0.0))

    return sorted(candidates, key=score, reverse=True)


def _asks_which_file_extreme(question: str) -> bool:
    normalized = normalize_for_match(question)
    return "which file" in normalized and any(
        term in normalized for term in ["highest", "largest", "max", "lowest", "smallest", "min"]
    )


def _answer_from_tables(
    profile: QuestionProfile,
    tables: dict[str, pd.DataFrame],
) -> tuple[str, dict[str, Any]]:
    normalized = normalize_for_match(profile.question)
    ordered = _ordered_tables(tables, profile.question)

    if "correlation" in normalized:
        for sheet_name, dataframe in ordered:
            answer = _correlation_answer(profile, dataframe)
            if answer != NOT_ENOUGH_DATA:
                return answer, {"operation": "correlation", "sheet": sheet_name}

    for sheet_name, dataframe in ordered:
        answer, debug = _answer_from_dataframe(profile, dataframe)
        if answer != NOT_ENOUGH_DATA:
            debug["sheet"] = sheet_name
            return answer, debug

    return NOT_ENOUGH_DATA, {}


def _answer_from_dataframe(
    profile: QuestionProfile,
    dataframe: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    normalized = normalize_for_match(profile.question)
    filtered = _apply_value_filters(dataframe, profile.question)
    operation = _operation(normalized)
    if operation == "count":
        gene_count = _single_column_gene_count(profile, dataframe)
        if gene_count is not None:
            return str(gene_count), {"operation": "count_gene_list", "rows": gene_count}
        return str(len(filtered)), {"operation": "count", "rows": len(filtered)}

    measure_column = _measure_column(filtered, profile.question)
    if not measure_column:
        return NOT_ENOUGH_DATA, {}
    values = pd.to_numeric(filtered[measure_column], errors="coerce").dropna()
    if values.empty:
        return NOT_ENOUGH_DATA, {}

    decimals = profile.format_instructions.get("decimals")
    if operation == "mean":
        return format_float(float(values.mean()), decimals), {"operation": "mean", "column": measure_column}
    if operation == "sum":
        return format_float(float(values.sum()), decimals), {"operation": "sum", "column": measure_column}
    if operation == "median":
        return format_float(float(values.median()), decimals), {"operation": "median", "column": measure_column}
    if operation in {"max", "min"}:
        idx = values.idxmax() if operation == "max" else values.idxmin()
        if _asks_for_label(normalized):
            label = _label_for_row(filtered, idx, measure_column, profile.question)
            if label:
                return label, {"operation": operation, "column": measure_column, "row": str(idx)}
        value = values.loc[idx]
        return format_float(float(value), decimals), {"operation": operation, "column": measure_column}

    return NOT_ENOUGH_DATA, {}


def _ordered_tables(
    tables: dict[str, pd.DataFrame],
    question: str,
) -> list[tuple[str, pd.DataFrame]]:
    normalized_question = normalize_for_match(question)
    q_tokens = set(normalize_for_match(question).split())

    def score(item: tuple[str, pd.DataFrame]) -> int:
        name, dataframe = item
        text = " ".join([normalize_for_match(name), *(normalize_for_match(col) for col in dataframe.columns)])
        value = sum(1 for token in q_tokens if token in text or any(part in token for part in text.split()))
        if "acetylproteomics" in normalized_question and "acetyl" in text:
            value += 50
        if "phosphoproteomics" in normalized_question and "phospho" in text:
            value += 50
        if "proteomics" in normalized_question and "proteomics" in text:
            value += 25
        if "significant gene" in normalized_question and dataframe.shape[1] == 1:
            value += 10
        return value

    return sorted(tables.items(), key=score, reverse=True)


def _single_column_gene_count(profile: QuestionProfile, dataframe: pd.DataFrame) -> int | None:
    normalized = normalize_for_match(profile.question)
    if not ("significant" in normalized and "gene" in normalized):
        return None
    if dataframe.shape[1] != 1:
        return None
    column = str(dataframe.columns[0])
    values = [
        normalize_spaces(value)
        for value in dataframe.iloc[:, 0].dropna().astype(str)
        if normalize_spaces(value)
    ]
    count = len(values)
    if _looks_like_gene_symbol(column):
        count += 1
    return count if count > 0 else None


def _looks_like_gene_symbol(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{1,15}", normalize_spaces(value)))


def _operation(normalized_question: str) -> str:
    if "correlation" in normalized_question:
        return "correlation"
    if any(term in normalized_question for term in ["average", "mean", "trung binh"]):
        return "mean"
    if any(term in normalized_question for term in ["sum", "total", "tong"]):
        return "sum"
    if "median" in normalized_question:
        return "median"
    if any(term in normalized_question for term in ["count", "how many", "bao nhieu", "so luong"]):
        return "count"
    if any(term in normalized_question for term in ["lowest", "smallest", "minimum", "min"]):
        return "min"
    if any(term in normalized_question for term in ["highest", "largest", "maximum", "max"]):
        return "max"
    return "unknown"


def _measure_column(dataframe: pd.DataFrame, question: str) -> str | None:
    quoted = [match.group(1).strip() for match in re.finditer(r"[\"']([^\"']+)[\"']", question)]
    numeric_columns = [
        column
        for column in dataframe.columns
        if pd.to_numeric(dataframe[column], errors="coerce").notna().sum() > 0
    ]
    if not numeric_columns:
        return None
    for phrase in quoted:
        matched = _match_column(numeric_columns, phrase)
        if matched:
            return matched

    q_tokens = set(normalize_for_match(question).split())

    def score(column: Any) -> tuple[int, int]:
        normalized = normalize_for_match(column)
        tokens = set(normalized.split())
        overlap = len(q_tokens & tokens)
        if any(term in normalized for term in ["id", "index", "stt"]):
            overlap -= 2
        if any(term in normalized for term in ["q2", "sales", "score", "diem", "acre", "cost", "price"]):
            overlap += 2
        non_null = pd.to_numeric(dataframe[column], errors="coerce").notna().sum()
        return overlap, int(non_null)

    return str(max(numeric_columns, key=score))


def _match_column(columns: list[Any], phrase: str) -> str | None:
    phrase_norm = normalize_for_match(phrase)
    for column in columns:
        column_norm = normalize_for_match(column)
        if phrase_norm == column_norm or phrase_norm in column_norm or column_norm in phrase_norm:
            return str(column)
    return None


def _apply_value_filters(dataframe: pd.DataFrame, question: str) -> pd.DataFrame:
    filtered = dataframe
    normalized_question = normalize_for_match(question)
    for column in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            continue
        values = dataframe[column].dropna().astype(str).unique()
        if len(values) > 200:
            continue
        matches = []
        for value in values:
            value_norm = normalize_for_match(value)
            if len(value_norm) >= 3 and value_norm in normalized_question:
                matches.append(value)
        if matches:
            filtered = filtered[filtered[column].astype(str).isin(matches)]
    return filtered if not filtered.empty else dataframe


def _correlation_answer(profile: QuestionProfile, dataframe: pd.DataFrame) -> str:
    quoted = [match.group(1).strip() for match in re.finditer(r"[\"']([^\"']+)[\"']", profile.question)]
    numeric_columns = [
        column
        for column in dataframe.columns
        if pd.to_numeric(dataframe[column], errors="coerce").notna().sum() > 1
    ]
    selected = []
    for phrase in quoted:
        matched = _match_column(numeric_columns, phrase)
        if matched and matched not in selected:
            selected.append(matched)
    if len(selected) < 2 and len(numeric_columns) >= 2:
        selected = [str(numeric_columns[0]), str(numeric_columns[1])]
    if len(selected) < 2:
        return NOT_ENOUGH_DATA
    a = pd.to_numeric(dataframe[selected[0]], errors="coerce")
    b = pd.to_numeric(dataframe[selected[1]], errors="coerce")
    value = a.corr(b)
    if value is None or math.isnan(float(value)):
        return NOT_ENOUGH_DATA
    return format_float(float(value), profile.format_instructions.get("decimals"))


def _asks_for_label(normalized_question: str) -> bool:
    return any(term in normalized_question for term in ["which", "what", "who", "file", "name", "ten", "nhan"])


def _label_for_row(dataframe: pd.DataFrame, row_index: Any, measure_column: str, question: str) -> str:
    row = dataframe.loc[row_index]
    q_tokens = set(normalize_for_match(question).split())
    candidates = []
    for column in dataframe.columns:
        if str(column) == measure_column:
            continue
        value = normalize_spaces(row[column])
        if not value or value.lower() == "nan":
            continue
        column_norm = normalize_for_match(column)
        score = len(set(column_norm.split()) & q_tokens)
        if any(term in column_norm for term in ["name", "label", "class", "file", "project", "student", "country"]):
            score += 3
        candidates.append((score, value))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


def _numeric_value(value: Any) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not match:
        return float("-inf")
    try:
        return float(match.group(0))
    except ValueError:
        return float("-inf")
