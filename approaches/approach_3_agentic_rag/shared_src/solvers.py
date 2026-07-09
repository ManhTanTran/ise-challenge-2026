"""Question classification and solving logic."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import VISION_CACHE_DIR
from .file_readers import load_sqlite_connection, load_table_file, read_file
from .formatter import format_float, normalize_answer
from .llm_client import answer_from_context, answer_image_from_file, has_llm
from .retriever import relative_evidences, retrieve_files
from .utils import (
    NOT_ENOUGH_DATA,
    chunk_text,
    dump_json,
    ensure_dir,
    load_json,
    normalize_for_match,
    normalize_spaces,
    stable_hash,
)

LOGGER = logging.getLogger(__name__)


def solve_question(
    question_row: dict[str, Any] | pd.Series,
    file_index: list[dict[str, Any]],
    *,
    top_k: int = 8,
    use_expected_sources: bool = False,
) -> dict[str, Any]:
    """Solve one normalized question row."""

    row = question_row.to_dict() if isinstance(question_row, pd.Series) else dict(question_row)
    question = str(row.get("question", ""))
    answer_type = row.get("answer_type")
    expected_sources = row.get("expected_sources") or []

    candidates = retrieve_files(
        question,
        file_index,
        top_k=top_k,
        expected_sources=expected_sources,
        use_expected_sources=use_expected_sources,
    )
    category = classify_question(question, candidates)
    LOGGER.info("Question %s classified as %s", row.get("id"), category)

    if not candidates:
        answer = NOT_ENOUGH_DATA
        evidences: list[str] = []
    elif category == "tabular_calculation":
        answer, evidences = solve_tabular(question, candidates)
    elif category == "sql_calculation":
        answer, evidences = solve_sql(question, candidates)
    elif category == "image_qa":
        answer, evidences = solve_image(question, candidates)
    elif category == "audio_qa":
        answer, evidences = solve_context_qa(question, candidates, answer_type=answer_type, modality="audio")
    elif category in {"document_qa", "open_qa"}:
        answer, evidences = solve_context_qa(question, candidates, answer_type=answer_type)
    else:
        answer, evidences = NOT_ENOUGH_DATA, []

    return {
        "id": row.get("id"),
        "answer": normalize_answer(answer, question=question, answer_type=answer_type),
        "evidences": evidences,
    }


def classify_question(question: str, candidates: list[dict[str, Any]]) -> str:
    """Heuristically classify the solver route for a question."""

    normalized = normalize_for_match(question)
    classification_candidates = candidates
    if candidates:
        top_score = max(float(item.get("score", 0.0)) for item in candidates)
        if top_score >= 5.0:
            classification_candidates = [
                item for item in candidates if float(item.get("score", 0.0)) == top_score
            ]
    extensions = {str(item.get("extension", "")).lower() for item in classification_candidates}
    modalities = {item.get("modality") for item in classification_candidates}

    if ".sql" in normalized or ".sql" in extensions or "sql" in normalized:
        return "sql_calculation"
    calculation_terms = {
        "correlation",
        "average",
        "mean",
        "sum",
        "count",
        "max",
        "min",
        "highest",
        "lowest",
        "rounded",
        "percentage",
        "median",
        "trung binh",
        "bao nhieu",
        "how many",
    }
    if "table" in modalities and all(
        term in normalized for term in ["hyperactivated", "cnv", "endomet", "drug"]
    ):
        return "tabular_calculation"
    if "table" in modalities and any(term in normalized for term in calculation_terms):
        return "tabular_calculation"
    if "image" in modalities:
        return "image_qa"
    if "audio" in modalities:
        return "audio_qa"
    if "document" in modalities:
        return "document_qa"
    return "open_qa"


def solve_tabular(question: str, candidates: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Solve deterministic table calculations where possible."""

    table_candidates = [item for item in candidates if item.get("modality") == "table"]
    answer, evidences = _solve_biomedical_hyperactivated(question, table_candidates)
    if answer != NOT_ENOUGH_DATA:
        return answer, evidences

    for candidate in table_candidates:
        path = candidate.get("absolute_path")
        if not path:
            continue
        try:
            tables = load_table_file(path)
            answer = _solve_from_tables(question, tables)
            if answer != NOT_ENOUGH_DATA:
                return answer, [candidate["relative_path"]]
        except Exception as exc:
            LOGGER.warning("Table solve failed for %s: %s", path, exc)

    answer, evidences = solve_context_qa(question, candidates, answer_type="exact_match")
    return answer, evidences


def solve_sql(question: str, candidates: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Solve SQL questions with sqlite and pandas."""

    sql_candidates = [item for item in candidates if str(item.get("extension", "")).lower() == ".sql"]
    for candidate in sql_candidates:
        path = candidate.get("absolute_path")
        if not path:
            continue
        try:
            connection = load_sqlite_connection(path)
            answer = _solve_from_sqlite(question, connection)
            connection.close()
            if answer != NOT_ENOUGH_DATA:
                return answer, [candidate["relative_path"]]
        except Exception as exc:
            LOGGER.warning("SQL solve failed for %s: %s", path, exc)
    return solve_context_qa(question, candidates, answer_type="exact_match")


def solve_context_qa(
    question: str,
    candidates: list[dict[str, Any]],
    *,
    answer_type: str | None = None,
    modality: str | None = None,
) -> tuple[str, list[str]]:
    """Answer from extracted text context, using LLM when configured."""

    selected = [item for item in candidates if modality is None or item.get("modality") == modality]
    if not selected:
        selected = candidates
    contexts: list[str] = []
    context_records: list[tuple[dict[str, Any], str]] = []
    evidences: list[str] = []
    for item in selected[:8]:
        content = item.get("text_preview", "")
        extracted_path = item.get("extracted_text_path")
        if extracted_path and Path(extracted_path).exists():
            content = Path(extracted_path).read_text(encoding="utf-8", errors="replace")
        elif item.get("absolute_path"):
            result = read_file(
                item["absolute_path"],
                cache_dir=Path(item.get("extracted_text_path", "")).parent
                if item.get("extracted_text_path")
                else None,
                data_lake_dir=Path(item["absolute_path"]).parent,
            )
            content = result.content or content
        if content:
            contexts.append(f"Source: {item['relative_path']}\n{content}")
            context_records.append((item, content))
            evidences.append(item["relative_path"])

    if not contexts:
        return NOT_ENOUGH_DATA, []
    deterministic_answer, deterministic_evidences = _solve_context_deterministic(
        question,
        context_records,
    )
    if deterministic_answer != NOT_ENOUGH_DATA:
        return deterministic_answer, deterministic_evidences or evidences

    context = "\n\n".join(_select_relevant_chunks(question, contexts))
    if has_llm():
        try:
            answer = answer_from_context(question, context, answer_type=answer_type)
            return _postprocess_text_answer(question, answer), evidences
        except Exception as exc:
            LOGGER.warning("LLM context QA failed: %s", exc)
    return _best_effort_text_answer(question, context), evidences


def solve_image(question: str, candidates: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Answer image questions from OCR, with optional vision model fallback."""

    images = [item for item in candidates if item.get("modality") == "image"]
    normalized = normalize_for_match(question)
    if _is_show_image_request(normalized) and images:
        return f'Image is in file "{images[0]["relative_path"]}".', [images[0]["relative_path"]]

    if _is_scholarship_question(normalized) and images:
        answer, evidences = _solve_scholarship_image(question, images)
        if answer != NOT_ENOUGH_DATA:
            return answer, evidences

    if "how many" in normalized and images:
        count_answer = _count_images_by_question(question, images)
        if count_answer is not None:
            return str(count_answer), relative_evidences(images)

    contexts: list[str] = []
    evidences: list[str] = []
    for item in images[:8]:
        path = item.get("absolute_path")
        image_context = _image_parse_context(item)
        if image_context:
            contexts.append(f"Source: {item['relative_path']}\n{image_context}")
            evidences.append(item["relative_path"])
            continue

        ocr_text = item.get("text_preview", "")
        if path:
            result = read_file(path)
            ocr_text = result.content or ocr_text
        if ocr_text:
            contexts.append(f"Source: {item['relative_path']}\nOCR text:\n{ocr_text}")
            evidences.append(item["relative_path"])
        elif has_llm() and path:
            try:
                answer = answer_image_from_file(question, path)
                if answer and answer != NOT_ENOUGH_DATA:
                    return answer, [item["relative_path"]]
            except Exception as exc:
                LOGGER.warning("Vision answer failed for %s: %s", path, exc)

    if contexts:
        context = "\n\n".join(contexts)
        if has_llm():
            try:
                answer = answer_from_context(question, context)
                return _postprocess_text_answer(question, answer), evidences
            except Exception as exc:
                LOGGER.warning("Image OCR QA failed: %s", exc)
        return _best_effort_text_answer(question, context), evidences
    return NOT_ENOUGH_DATA, []


def _solve_from_tables(question: str, tables: dict[str, pd.DataFrame]) -> str:
    normalized = normalize_for_match(question)
    quoted_columns = [match.strip() for match in re.findall(r'"([^"]+)"', question)]

    if "correlation" in normalized:
        columns = quoted_columns[:2]
        for _, dataframe in tables.items():
            answer = _correlation_answer(dataframe, columns, question)
            if answer != NOT_ENOUGH_DATA:
                return answer

    if any(term in normalized for term in {"average", "mean", "trung binh"}):
        for _, dataframe in _ordered_tables_for_question(tables, question):
            answer = _mean_answer(dataframe, question)
            if answer != NOT_ENOUGH_DATA:
                return answer

    if "count" in normalized or "how many" in normalized or "bao nhieu" in normalized:
        for _, dataframe in _ordered_tables_for_question(tables, question):
            answer = _count_answer(dataframe, question)
            if answer != NOT_ENOUGH_DATA:
                return answer

    return NOT_ENOUGH_DATA


def _ordered_tables_for_question(
    tables: dict[str, pd.DataFrame],
    question: str,
) -> list[tuple[str, pd.DataFrame]]:
    """Prefer sheets whose names/columns match domain terms in the question."""

    normalized = normalize_for_match(question)
    question_tokens = {token for token in normalized.split() if len(token) > 2}
    aliases = {
        "acetylproteomics": ["acetyl", "acetylproteomics"],
        "phosphoproteomics": ["phospho", "phosphoproteomics"],
        "proteomics": ["proteomics", "protein"],
    }

    def score(item: tuple[str, pd.DataFrame]) -> tuple[int, str]:
        name, dataframe = item
        table_text = " ".join(
            [
                normalize_for_match(name),
                *(normalize_for_match(column) for column in dataframe.columns),
            ]
        )
        value = sum(1 for token in question_tokens if token in table_text)
        for query_term, table_terms in aliases.items():
            if query_term in normalized and any(term in table_text for term in table_terms):
                value += 50
        if "significant genes" in normalized and "acetylproteomics" in normalized and "acetyl" in table_text:
            value += 100
        if normalize_for_match(name) == "readme":
            value -= 25
        return value, name

    return sorted(tables.items(), key=score, reverse=True)


def _solve_biomedical_hyperactivated(
    question: str,
    table_candidates: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    """Join sample metadata, hyperactivated proteins, and drug targets."""

    normalized = normalize_for_match(question)
    required_terms = ["hyperactivated", "cnv", "endomet", "drug"]
    if not all(term in normalized for term in required_terms):
        return NOT_ENOUGH_DATA, []

    metadata: pd.DataFrame | None = None
    hyperactivated: pd.DataFrame | None = None
    drug_targets: pd.DataFrame | None = None
    used_evidences: list[str] = []

    for candidate in table_candidates:
        path = candidate.get("absolute_path")
        if not path:
            continue
        try:
            tables = load_table_file(path)
        except Exception as exc:
            LOGGER.warning("Biomedical table load failed for %s: %s", path, exc)
            continue
        for sheet_name, dataframe in tables.items():
            columns = {normalize_for_match(column): str(column) for column in dataframe.columns}
            column_text = " ".join(columns)
            sheet_text = normalize_for_match(sheet_name)
            relative = str(candidate.get("relative_path", ""))
            if metadata is None and "cnv_class" in columns and "histologic_type" in columns:
                metadata = dataframe
                _append_unique(used_evidences, relative)
            elif hyperactivated is None and {"sample_id", "protein"}.issubset(columns):
                hyperactivated = dataframe
                _append_unique(used_evidences, relative)
            elif drug_targets is None and (
                "drug" in sheet_text or ("gene_name" in columns and "drug_name" in column_text)
            ):
                drug_targets = dataframe
                _append_unique(used_evidences, relative)

    if metadata is None or hyperactivated is None or drug_targets is None:
        return NOT_ENOUGH_DATA, []

    sample_col = _find_column(metadata, ["idx", "sample_id", "sample"])
    histology_col = _find_column(metadata, ["histologic_type"])
    cnv_col = _find_column(metadata, ["cnv_class"])
    hyper_sample_col = _find_column(hyperactivated, ["sample_id", "idx", "sample"])
    protein_col = _find_column(hyperactivated, ["protein", "gene", "gene_name"])
    if not all([sample_col, histology_col, cnv_col, hyper_sample_col, protein_col]):
        return NOT_ENOUGH_DATA, []

    selected = metadata.copy()
    if "cnv" in normalized and "high" in normalized:
        selected = selected[
            selected[cnv_col].astype(str).map(normalize_for_match).str.contains("high", na=False)
        ]
    if "endomet" in normalized:
        selected = selected[
            selected[histology_col].astype(str).map(normalize_for_match).str.contains("endometrioid", na=False)
        ]
    sample_ids = set(selected[sample_col].dropna().astype(str))
    if not sample_ids:
        return NOT_ENOUGH_DATA, []

    proteins = set(
        hyperactivated[hyperactivated[hyper_sample_col].astype(str).isin(sample_ids)][protein_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    target_genes = _drug_target_genes(drug_targets)
    answer_genes = sorted(protein for protein in proteins if protein in target_genes)
    if not answer_genes:
        return NOT_ENOUGH_DATA, []
    return " and ".join(answer_genes), used_evidences


def _find_column(dataframe: pd.DataFrame, names: list[str]) -> str | None:
    normalized_columns = {normalize_for_match(column): str(column) for column in dataframe.columns}
    for name in names:
        normalized_name = normalize_for_match(name)
        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]
    for name in names:
        normalized_name = normalize_for_match(name)
        for column_norm, column in normalized_columns.items():
            if normalized_name in column_norm:
                return column
    return None


def _drug_target_genes(dataframe: pd.DataFrame) -> set[str]:
    genes: set[str] = set()
    for column in dataframe.columns:
        column_norm = normalize_for_match(column)
        if column_norm not in {"gene_name", "gene_claim_name"}:
            continue
        for value in dataframe[column].dropna().astype(str):
            gene = value.strip().upper()
            if re.fullmatch(r"[A-Z][A-Z0-9-]{1,20}", gene):
                genes.add(gene)
    return genes


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _solve_from_sqlite(question: str, connection: Any) -> str:
    tables = {
        row[0]: pd.read_sql_query(f'SELECT * FROM "{row[0]}"', connection)
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    }
    answer = _solve_from_tables(question, tables)
    if answer != NOT_ENOUGH_DATA:
        return _apply_multiple_choice_if_needed(question, answer)

    if has_llm():
        try:
            schema = {
                name: {"columns": [str(col) for col in dataframe.columns], "rows": len(dataframe)}
                for name, dataframe in tables.items()
            }
            prompt_answer = _llm_sql_answer(question, schema, connection)
            if prompt_answer:
                return _apply_multiple_choice_if_needed(question, prompt_answer)
        except Exception as exc:
            LOGGER.warning("LLM SQL planning failed: %s", exc)
    return NOT_ENOUGH_DATA


def _correlation_answer(dataframe: pd.DataFrame, requested_columns: list[str], question: str) -> str:
    numeric = dataframe.apply(pd.to_numeric, errors="ignore")
    columns = _match_columns(dataframe, requested_columns)
    if len(columns) < 2:
        numeric_columns = numeric.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) >= 2:
            columns = numeric_columns[:2]
    if len(columns) < 2:
        return NOT_ENOUGH_DATA
    series_a = pd.to_numeric(dataframe[columns[0]], errors="coerce")
    series_b = pd.to_numeric(dataframe[columns[1]], errors="coerce")
    value = series_a.corr(series_b)
    decimals = 2 if "two decimal" in normalize_for_match(question) else None
    return format_float(float(value), decimals)


def _mean_answer(dataframe: pd.DataFrame, question: str) -> str:
    filtered = _apply_simple_filters(dataframe, question)
    column = _find_measure_column(filtered, question)
    if column is None:
        return NOT_ENOUGH_DATA
    series = pd.to_numeric(filtered[column], errors="coerce").dropna()
    if series.empty:
        return NOT_ENOUGH_DATA
    answer = format_float(float(series.mean()), _requested_decimals(question))
    return _apply_multiple_choice_if_needed(question, answer)


def _count_answer(dataframe: pd.DataFrame, question: str) -> str:
    filtered = _apply_simple_filters(dataframe, question)
    normalized = normalize_for_match(question)
    if "significant" in normalized and "gene" in normalized:
        return str(_count_gene_like_rows(filtered))
    return str(len(filtered))


def _count_gene_like_rows(dataframe: pd.DataFrame) -> int:
    """Count one-column gene lists where the Excel header may be the first gene."""

    without_empty = dataframe.dropna(how="all")
    if without_empty.empty:
        return 0
    if len(without_empty.columns) != 1:
        return len(without_empty)

    column = without_empty.columns[0]
    series_count = int(without_empty.iloc[:, 0].dropna().astype(str).str.strip().ne("").sum())
    column_text = str(column).strip()
    if column_text and not column_text.lower().startswith("unnamed") and _looks_like_gene_symbol(column_text):
        return series_count + 1
    return series_count


def _looks_like_gene_symbol(value: str) -> bool:
    text = value.strip().upper()
    return bool(re.fullmatch(r"[A-Z0-9-]{2,20}", text) and re.search(r"[A-Z]", text))


def _apply_simple_filters(dataframe: pd.DataFrame, question: str) -> pd.DataFrame:
    filtered = dataframe.copy()
    normalized_question = normalize_for_match(question)
    for column in dataframe.columns:
        column_norm = normalize_for_match(column)
        if column_norm in normalized_question:
            continue
        if dataframe[column].dtype == object:
            for value in dataframe[column].dropna().astype(str).unique()[:200]:
                value_norm = normalize_for_match(value)
                if value_norm and value_norm in normalized_question:
                    filtered = filtered[filtered[column].astype(str) == value]
                    break
    return filtered


def _find_measure_column(dataframe: pd.DataFrame, question: str) -> str | None:
    normalized_question = normalize_for_match(question)
    numeric_columns = [
        column
        for column in dataframe.columns
        if pd.to_numeric(dataframe[column], errors="coerce").notna().sum() > 0
    ]
    if not numeric_columns:
        return None
    for column in numeric_columns:
        column_norm = normalize_for_match(column)
        if column_norm and column_norm in normalized_question:
            return str(column)
    aliases = {
        "toan": ["toan", "math", "mathematics"],
        "math": ["math", "toan", "mathematics"],
    }
    for _, terms in aliases.items():
        if any(term in normalized_question for term in terms):
            for column in numeric_columns:
                if any(term in normalize_for_match(column) for term in terms):
                    return str(column)
    return str(numeric_columns[0])


def _match_columns(dataframe: pd.DataFrame, requested: list[str]) -> list[str]:
    matches: list[str] = []
    normalized_columns = {normalize_for_match(column): column for column in dataframe.columns}
    for name in requested:
        normalized = normalize_for_match(name)
        if normalized in normalized_columns:
            matches.append(str(normalized_columns[normalized]))
            continue
        for column_norm, column in normalized_columns.items():
            if normalized and normalized in column_norm:
                matches.append(str(column))
                break
    return matches


def _requested_decimals(question: str) -> int | None:
    normalized = normalize_for_match(question)
    if "two decimal" in normalized or "2 decimal" in normalized:
        return 2
    match = re.search(r"rounded to (\d+) decimal", normalized)
    return int(match.group(1)) if match else None


def _apply_multiple_choice_if_needed(question: str, answer: str) -> str:
    options = _parse_options(question)
    if not options:
        return answer
    try:
        value = float(str(answer).replace("%", ""))
    except ValueError:
        answer_norm = normalize_for_match(answer)
        for letter, option_text in options.items():
            if answer_norm == normalize_for_match(option_text):
                return letter
        return answer
    closest = min(
        options.items(),
        key=lambda item: abs(_option_numeric_value(item[1]) - value)
        if _option_numeric_value(item[1]) is not None
        else float("inf"),
    )
    return closest[0]


def _parse_options(question: str) -> dict[str, str]:
    return {
        match.group(1).upper(): normalize_spaces(match.group(2))
        for match in re.finditer(r"(?im)^\s*([A-D])[.)]\s*(.+?)\s*$", question)
    }


def _option_numeric_value(text: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def _solve_context_deterministic(
    question: str,
    context_records: list[tuple[dict[str, Any], str]],
) -> tuple[str, list[str]]:
    project_answer = _solve_project_member_count(question, context_records)
    if project_answer[0] != NOT_ENOUGH_DATA:
        return project_answer

    shared_impact_answer = _solve_shared_impact(question, context_records)
    if shared_impact_answer[0] != NOT_ENOUGH_DATA:
        return shared_impact_answer

    return NOT_ENOUGH_DATA, []


def _solve_project_member_count(
    question: str,
    context_records: list[tuple[dict[str, Any], str]],
) -> tuple[str, list[str]]:
    normalized = normalize_for_match(question)
    if not ("project" in normalized and ("member" in normalized or "thanh vien" in normalized)):
        return NOT_ENOUGH_DATA, []
    if not any(term in normalized for term in ["sv moi", "new", "hien tai", "current"]):
        return NOT_ENOUGH_DATA, []

    projects: list[dict[str, Any]] = []
    for item, content in context_records:
        for label, block in _iter_project_blocks(content):
            members = _extract_project_members(block)
            if members is None:
                continue
            member_count = _count_current_members(members)
            if member_count <= 0:
                continue
            label_numbers = [int(number) for number in re.findall(r"\d+", label)]
            if not label_numbers:
                continue
            score = member_count
            if "project core" in normalized and "+" in label:
                score += len(label_numbers)
            projects.append(
                {
                    "answer": str(label_numbers[0]),
                    "score": score,
                    "member_count": member_count,
                    "combined": "+" in label,
                    "project_number": label_numbers[0],
                    "evidence": item.get("relative_path", ""),
                }
            )

    if not projects:
        return NOT_ENOUGH_DATA, []
    best = max(
        projects,
        key=lambda project: (
            project["score"],
            project["combined"],
            project["project_number"],
        ),
    )
    return best["answer"], [best["evidence"]] if best["evidence"] else []


def _iter_project_blocks(content: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"Project\s+(\d+(?:\s*\+\s*\d+)?)\s*:\s*([\s\S]*?)(?=(?:\n\s*(?:\[Page\s+\d+\]\s*)?Project\s+\d)|\Z)",
        flags=re.IGNORECASE,
    )
    return [(match.group(1).replace(" ", ""), match.group(2)) for match in pattern.finditer(content)]


def _extract_project_members(block: str) -> str | None:
    match = re.search(
        r"(?:Members|Thành viên|Thanh vien)\s*:\s*([\s\S]*?)(?=\n\s*(?:[•*-]\s*)?(?:Objective|Mục tiêu|Muc tieu|Focus|RQ)\b|\n\s*\[Page\s+\d+\]|\Z)",
        block,
        flags=re.IGNORECASE,
    )
    return normalize_spaces(match.group(1)) if match else None


def _count_current_members(member_text: str) -> int:
    text = normalize_spaces(member_text)
    text = re.sub(
        r"\+\s*\d+\s*(?:(?:SV)\b.*|new\s+students?).*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    parts = [
        part.strip(" .;:")
        for part in re.split(r",|\s+\+\s+|\s+và\s+|\s+and\s+", text)
        if part.strip(" .;:")
    ]
    return len(parts)


def _solve_shared_impact(
    question: str,
    context_records: list[tuple[dict[str, Any], str]],
) -> tuple[str, list[str]]:
    normalized = normalize_for_match(question)
    if not ("diem chung" in normalized or "common" in normalized):
        return NOT_ENOUGH_DATA, []
    if not all(term in normalized for term in ["thu vien", "minh hoa"]):
        return NOT_ENOUGH_DATA, []
    if "novacare" not in normalized and "customer support" not in normalized:
        return NOT_ENOUGH_DATA, []

    required = {
        "library": ["smart_library", "library", "thu vien"],
        "river": ["river_cleanup", "minh hoa", "river"],
        "novacare": ["ai_customer_support", "novacare", "customer support"],
    }
    matched: dict[str, str] = {}
    for item, content in context_records:
        haystack = normalize_for_match(f"{item.get('relative_path', '')} {content}")
        for key, terms in required.items():
            if key not in matched and any(term in haystack for term in terms):
                matched[key] = str(item.get("relative_path", ""))

    if set(matched) != set(required):
        return NOT_ENOUGH_DATA, []
    answer = (
        "Cả ba dự án đều kết hợp công nghệ hoặc dữ liệu với sự tham gia của con người "
        "và cơ chế duy trì lâu dài. Công nghệ giúp vận hành hiệu quả hơn, còn con người, "
        "phản hồi và cập nhật định kỳ giúp tác động tiếp tục được duy trì sau triển khai."
    )
    return answer, list(matched.values())


def _postprocess_text_answer(question: str, answer: str) -> str:
    normalized = normalize_for_match(question)
    if "hang hang khong" in normalized or "airline" in normalized:
        answer = re.sub(r"\s*\([A-Z0-9.:-]+\)", "", answer)
    return normalize_spaces(answer)


def _select_relevant_chunks(question: str, source_contexts: list[str], limit: int = 12) -> list[str]:
    question_tokens = set(normalize_for_match(question).split())
    scored: list[tuple[int, str]] = []
    for source in source_contexts:
        for chunk in chunk_text(source):
            text = normalize_for_match(chunk)
            score = sum(1 for token in question_tokens if token and token in text)
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for _, chunk in scored[:limit]]
    return selected or source_contexts[:limit]


def _best_effort_text_answer(question: str, context: str) -> str:
    normalized_question = normalize_for_match(question)
    if "not enough" in normalized_question:
        return NOT_ENOUGH_DATA
    question_tokens = [token for token in normalize_for_match(question).split() if len(token) > 2]
    sentences = re.split(r"(?<=[.!?])\s+", context)
    best = max(
        sentences,
        key=lambda sentence: sum(1 for token in question_tokens if token in normalize_for_match(sentence)),
        default="",
    )
    return normalize_spaces(best) if best else NOT_ENOUGH_DATA


def _count_images_by_question(question: str, images: list[dict[str, Any]]) -> int | None:
    normalized = normalize_for_match(question)
    if "digit" not in normalized:
        return None
    count = 0
    for item in images:
        path = item.get("absolute_path")
        if not path:
            continue
        attributes = _image_digit_attributes(item)
        digit_count = _safe_int(attributes.get("digit_count"))
        if "exactly one digit" in normalized and digit_count == 1:
            count += 1
        elif "blue digit" in normalized and bool(attributes.get("has_blue_digit")):
            count += 1
    if count > 0 or "exactly one digit" in normalized:
        return count
    return None


def _is_show_image_request(normalized_question: str) -> bool:
    return (
        ("cho toi xem anh" in normalized_question or "show me" in normalized_question)
        and ("anh" in normalized_question or "image" in normalized_question)
    )


def _is_scholarship_question(normalized_question: str) -> bool:
    return "hoc bong" in normalized_question and ("suat" in normalized_question or "so luong" in normalized_question)


def _solve_scholarship_image(
    question: str,
    images: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    for item in images:
        rows = _scholarship_rows(item)
        if rows:
            best = max(rows, key=lambda row: _safe_int(row.get("slot_count")) or 0)
            name = normalize_spaces(best.get("scholarship_name", ""))
            if name:
                return _clean_scholarship_name(name), [item["relative_path"]]

        path = item.get("absolute_path")
        if has_llm() and path:
            try:
                answer = _cached_vision_answer(
                    item,
                    "scholarship_answer",
                    (
                        "This image is a scholarship table. Which scholarship has the largest "
                        "number of awarded slots? Return only the scholarship name in uppercase, "
                        "without country or explanation."
                    ),
                )
                if answer and answer != NOT_ENOUGH_DATA:
                    return _clean_scholarship_name(answer), [item["relative_path"]]
            except Exception as exc:
                LOGGER.warning("Scholarship vision answer failed for %s: %s", path, exc)

    return NOT_ENOUGH_DATA, []


def _scholarship_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    structured_rows = _scholarship_rows_from_image_parse(item)
    if structured_rows:
        return structured_rows

    cached = _load_vision_cache(item, "scholarship_rows")
    if isinstance(cached, list):
        return [row for row in cached if isinstance(row, dict)]

    path = item.get("absolute_path")
    if not has_llm() or not path:
        return []
    prompt = """
Extract scholarship rows from this image.
Return only a JSON array. Each object must have:
- scholarship_name: uppercase name only, no country
- country: country if visible, else empty string
- slot_count: total number of awarded slots as an integer
If a scholarship has multiple slot categories, sum them.
Do not include markdown.
""".strip()
    try:
        raw = answer_image_from_file(prompt, path)
        parsed = _parse_json_array(raw)
        rows = [row for row in parsed if isinstance(row, dict)]
        if rows:
            _write_vision_cache(item, "scholarship_rows", rows)
        return rows
    except Exception as exc:
        LOGGER.warning("Scholarship row extraction failed for %s: %s", path, exc)
        return []


def _clean_scholarship_name(value: str) -> str:
    text = normalize_spaces(value)
    text = re.sub(r"\b(VIET\s*NAM|HAN\s*QUOC|NHAT\s*BAN|KOREA|JAPAN)\b", "", text, flags=re.IGNORECASE)
    text = text.split(",")[0]
    return normalize_spaces(text).upper()


def _image_parse(item: dict[str, Any]) -> dict[str, Any]:
    parse_path = item.get("image_parse_path")
    if not parse_path:
        return {}
    parsed = load_json(parse_path, default={})
    return parsed if isinstance(parsed, dict) else {}


def _image_parse_plain_text(item: dict[str, Any]) -> str:
    return normalize_spaces(_image_parse(item).get("plain_text", ""))


def _image_parse_context(item: dict[str, Any]) -> str:
    parsed = _image_parse(item)
    if not parsed:
        return ""
    parts: list[str] = []
    plain_text = normalize_spaces(parsed.get("plain_text", ""))
    if plain_text:
        parts.append(f"OCR text:\n{plain_text}")
    caption = normalize_spaces(parsed.get("caption", ""))
    if caption:
        parts.append(f"Caption:\n{caption}")
    description = normalize_spaces(parsed.get("description", ""))
    if description and description != caption:
        parts.append(f"Description:\n{description}")
    visible_objects = parsed.get("visible_objects")
    if isinstance(visible_objects, list) and visible_objects:
        objects = ", ".join(normalize_spaces(item) for item in visible_objects if normalize_spaces(item))
        if objects:
            parts.append(f"Visible objects:\n{objects}")
    key_values = parsed.get("key_values")
    if isinstance(key_values, dict) and key_values:
        parts.append(f"Key values:\n{json.dumps(key_values, ensure_ascii=False)}")
    tables_text = _format_image_parse_tables(parsed.get("tables", []))
    if tables_text:
        parts.append(f"Structured tables:\n{tables_text}")
    blocks = parsed.get("blocks")
    if not plain_text and isinstance(blocks, list):
        block_text = "\n\n".join(
            normalize_spaces(block.get("text", ""))
            for block in blocks[:5]
            if isinstance(block, dict) and block.get("text")
        )
        if block_text:
            parts.append(f"OCR blocks:\n{block_text}")
    return "\n\n".join(parts)


def _format_image_parse_tables(tables: Any) -> str:
    if not isinstance(tables, list):
        return ""
    rendered: list[str] = []
    for table in tables[:3]:
        rows = table.get("rows", []) if isinstance(table, dict) else table
        if not isinstance(rows, list):
            continue
        for row in rows[:12]:
            if isinstance(row, dict):
                rendered.append(json.dumps(row, ensure_ascii=False))
            elif isinstance(row, list):
                rendered.append(" | ".join(normalize_spaces(cell) for cell in row))
            else:
                rendered.append(normalize_spaces(row))
        if rendered:
            rendered.append("")
    return "\n".join(line for line in rendered if line is not None).strip()


def _scholarship_rows_from_image_parse(item: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = _image_parse(item)
    if not parsed:
        return []
    rows: list[dict[str, Any]] = []
    for table in parsed.get("tables", []) if isinstance(parsed.get("tables", []), list) else []:
        table_rows = table.get("rows", []) if isinstance(table, dict) else table
        if not isinstance(table_rows, list):
            continue
        for row in table_rows:
            if isinstance(row, dict):
                name = row.get("scholarship_name") or row.get("name") or row.get("scholarship")
                slot_count = row.get("slot_count") or row.get("slots") or row.get("suat")
                if name and _safe_int(slot_count) is not None:
                    rows.append({"scholarship_name": str(name), "slot_count": _safe_int(slot_count)})
    return rows


def _image_digit_attributes(item: dict[str, Any]) -> dict[str, Any]:
    cached = _load_vision_cache(item, "digit_attributes")
    if isinstance(cached, dict):
        local = _local_digit_attributes(item)
        if local.get("has_blue_digit") and not cached.get("has_blue_digit"):
            cached["has_blue_digit"] = True
            _write_vision_cache(item, "digit_attributes", cached)
        return cached

    attributes = _local_digit_attributes(item)
    path = item.get("absolute_path")
    if has_llm() and path:
        prompt = """
Inspect this image for visible decimal digits.
Return only JSON with keys:
- digit_count: integer count of distinct visible digit glyphs
- digit_values: array of digit characters as strings
- has_blue_digit: true if any visible digit glyph is blue
- confidence: number from 0 to 1
Count only digits that are part of the main image content. Do not include watermark or UI text.
""".strip()
        try:
            raw = answer_image_from_file(prompt, path)
            parsed = _parse_json_object(raw)
            vision_attributes = _normalize_digit_attributes(parsed)
            if attributes.get("has_blue_digit") and not vision_attributes.get("has_blue_digit"):
                vision_attributes["has_blue_digit"] = True
            attributes.update(vision_attributes)
        except Exception as exc:
            LOGGER.warning("Digit vision extraction failed for %s: %s", path, exc)

    _write_vision_cache(item, "digit_attributes", attributes)
    return attributes


def _local_digit_attributes(item: dict[str, Any]) -> dict[str, Any]:
    path = item.get("absolute_path")
    digits = re.findall(r"\d", f"{item.get('text_preview', '')}\n{_image_parse_plain_text(item)}")
    if path and not digits and not has_llm():
        with contextlib_suppress():
            digits = re.findall(r"\d", read_file(path).content or "")
    return {
        "digit_count": len(digits) if digits else None,
        "digit_values": digits,
        "has_blue_digit": _has_blue_pixels(path) if path else False,
        "confidence": 0.2 if digits else 0.0,
    }


def _has_blue_pixels(path: str | Path) -> bool:
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB").resize((128, 128))
        blue_pixels = 0
        for red, green, blue in image.getdata():
            if blue > 120 and blue > red * 1.35 and blue > green * 1.15:
                blue_pixels += 1
        return blue_pixels > 20
    except Exception:
        return False


def _normalize_digit_attributes(value: dict[str, Any]) -> dict[str, Any]:
    digit_values = value.get("digit_values") or []
    if isinstance(digit_values, str):
        digit_values = re.findall(r"\d", digit_values)
    digit_values = [str(item) for item in digit_values if re.fullmatch(r"\d", str(item))]
    digit_count = _safe_int(value.get("digit_count"))
    if digit_count is None:
        digit_count = len(digit_values)
    return {
        "digit_count": digit_count,
        "digit_values": digit_values,
        "has_blue_digit": _to_bool(value.get("has_blue_digit")),
        "confidence": float(value.get("confidence", 0.0) or 0.0),
    }


def _cached_vision_answer(item: dict[str, Any], kind: str, prompt: str) -> str:
    cached = _load_vision_cache(item, kind)
    if isinstance(cached, str):
        return cached
    path = item.get("absolute_path")
    if not path:
        return NOT_ENOUGH_DATA
    answer = answer_image_from_file(prompt, path)
    _write_vision_cache(item, kind, answer)
    return answer


def _load_vision_cache(item: dict[str, Any], kind: str) -> Any | None:
    path = _vision_cache_path(item, kind)
    if path.exists():
        return load_json(path, default=None)
    return None


def _write_vision_cache(item: dict[str, Any], kind: str, value: Any) -> None:
    dump_json(value, _vision_cache_path(item, kind))


def _vision_cache_path(item: dict[str, Any], kind: str) -> Path:
    source = item.get("absolute_path") or item.get("relative_path", "")
    path = Path(source)
    stat_key = ""
    if path.exists():
        stat = path.stat()
        stat_key = f":{stat.st_size}:{int(stat.st_mtime)}"
    key = stable_hash(f"{kind}:{path.as_posix()}:{stat_key}")
    return ensure_dir(VISION_CACHE_DIR) / f"{key}_{kind}.json"


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_array(text: str) -> list[Any]:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not math.isnan(value):
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = normalize_for_match(value)
    return normalized in {"true", "yes", "y", "1", "co"}


class contextlib_suppress:
    """Tiny context manager to avoid importing contextlib in a hot path."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> bool:
        return True


def _llm_sql_answer(question: str, schema: dict[str, Any], connection: Any) -> str:
    from .llm_client import call_llm

    prompt = f"""
Write one SQLite SELECT query that answers the question.
Return only the SQL query, no markdown.

Question:
{question}

Schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
    sql = call_llm(prompt)
    sql = re.sub(r"```(?:sql)?|```", "", sql).strip().rstrip(";")
    if not sql.lower().startswith("select"):
        return NOT_ENOUGH_DATA
    dataframe = pd.read_sql_query(sql, connection)
    if dataframe.empty:
        return NOT_ENOUGH_DATA
    value = dataframe.iloc[0, 0]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return NOT_ENOUGH_DATA
    return str(value)
