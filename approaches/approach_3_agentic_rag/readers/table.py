"""Buoc 3a: table reader with code-first computation.

For computation questions the LLM writes ONE pandas expression over the
retrieved tables; the expression is executed locally so arithmetic is exact
instead of hallucinated. When a question needs data from more than one
retrieved table file (e.g. "proteins targeted by drugs in file B, filtered by
significance in file A"), every table-modality candidate is loaded into its
own named DataFrame variable so a single expression can join across them.
The expression is screened against a blocklist and evaluated with a builtins
whitelist (len, sum, min, max...; no imports/exec/file I/O). Any failure falls
back to schema + sample rows so the reasoning step still has usable context.
"""

from __future__ import annotations

import builtins as _builtins
import io
import json
import logging
import re
from typing import Any

import numpy as np
import pandas as pd

from ..config import Approach3Config
from ..core.models import QuestionProfile
from ..shared_src.file_readers import extract_candidate_text, load_table_file
from ..shared_src.llm_client import call_llm, has_llm
from ..shared_src.utils import has_word, normalize_for_match, normalize_spaces, truncate_text

LOGGER = logging.getLogger(__name__)

_FORBIDDEN = re.compile(
    r"__|\bimport\b|\bopen\b|\bexec\b|\beval\b|\bcompile\b|\bglobals\b|\blocals\b"
    r"|\bgetattr\b|\bsetattr\b|\bdelattr\b|\bos\b|\bsys\b|\bsubprocess\b"
    r"|read_csv|read_excel|to_csv|to_excel|to_pickle|read_pickle|\bquery\s*\("
)

_CODEGEN_PROMPT = """
You translate one analytics question into ONE Python expression over pandas
DataFrame(s) (pandas as pd, numpy as np are available).

Available tables (each is a separate variable; use its exact name):
{schema}

Rules:
- Return ONLY JSON: {{"expression": "..."}}
- Single expression, no statements, no imports, no file or network access.
- Use only columns listed in the schema above. Match column names exactly.
- If the question needs data from more than one table, join/filter across the
  variables shown above in ONE expression (e.g. df_a[df_a['gene'].isin(df_b['gene'])]).
- Prefer exact filters (== on values shown in samples) and pandas aggregations.
- String/categorical columns often have leading/trailing whitespace that
  isn't obvious from how the sample values are displayed (e.g. a column may
  store " >50K" with a leading space, not ">50K"). For any `==` filter on a
  string column, strip both sides defensively:
  `df['col'].str.strip() == 'value'` instead of `df['col'] == 'value'`, so an
  invisible whitespace mismatch does not silently produce zero results.
- If the question asks WHICH entities (genes, proteins, names, categories...)
  satisfy a condition - not a row-level record - end the expression with
  something that returns the distinct entity values (e.g. `['gene'].unique()`
  or `.drop_duplicates()`), not every matching row. A question asking "which
  genes" wants gene names, not one row per site/measurement of those genes.
- If a table's own description (e.g. a README/index sheet) already names or
  defines exactly what the question asks for, use that table's row count or
  values as-is. Do NOT invent an additional filter/threshold (a p-value, FDR,
  fold-change cutoff, etc.) that is not stated in the question, even if
  another table looks more detailed or "more precise" - a plain row count
  from the table the question is literally naming is correct as given.
- If the question cannot be answered from these tables, return {{"expression": ""}}.

Question:
{question}
""".strip()

_SHORTLIST_PROMPT = """
A question needs to be answered using SOME of the tables below. Each entry
shows only a variable name and its column names (no data). Pick every
variable that is plausibly needed - when in doubt, include it.

Tables:
{toc}

Return ONLY JSON: {{"relevant": ["var_name", ...]}}

Question:
{question}
""".strip()

# Full schema (columns + sample rows) for every sheet of a multi-file/sheet
# join can run past this many characters (observed: 30 sheets across 8
# biomedical xlsx files -> 170k chars), silently truncating away the one
# sheet the join actually needs and making the LLM give up with an empty
# expression. Shortlisting relevant sheets first (from a cheap columns-only
# table of contents that never gets this large) keeps the real schema build
# small enough to never hit this limit.
_SCHEMA_CHAR_BUDGET = 16000


def table_context_text(
    candidate: dict[str, Any],
    *,
    config: Approach3Config,
) -> str:
    """Schema plus sample rows: the always-available table context."""

    tables = _load_tables(candidate)
    if not tables:
        return ""
    parts: list[str] = []
    for sheet_name, frame in tables.items():
        parts.append(_describe_table(sheet_name, frame, sample_rows=config.table_sample_rows))
    return "\n\n".join(parts)


def compute_table_answer(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
    *,
    config: Approach3Config,
) -> dict[str, Any] | None:
    """Generate and execute one pandas expression across all table candidates.

    Every (candidate, sheet) pair gets its own namespace variable so a single
    LLM-written expression can join across files. When only one table loads,
    it is also aliased as `df` so simple single-table expressions still work.
    """

    if not (config.use_table_compute and has_llm()) or not candidates:
        return None

    all_frames: dict[str, pd.DataFrame] = {}
    var_sources: dict[str, str] = {}
    var_sheet_names: dict[str, str] = {}
    used_names: set[str] = set()

    for candidate in candidates:
        tables = _load_tables(candidate)
        relative_path = str(candidate.get("relative_path", candidate.get("filename", "table")))
        base_var = _unique_identifier(relative_path, used_names)
        for sheet_name, frame in tables.items():
            var = base_var if len(tables) == 1 else _unique_identifier(
                f"{base_var}_{sheet_name}", used_names
            )
            all_frames[var] = frame
            var_sources[var] = relative_path
            var_sheet_names[var] = sheet_name

    if not all_frames:
        return None

    selected_vars = (
        _shortlist_relevant_tables(profile.question, all_frames, config=config)
        if len(all_frames) > 1
        else set(all_frames)
    )
    namespace: dict[str, pd.DataFrame] = {var: frame for var, frame in all_frames.items() if var in selected_vars}
    if not namespace:
        namespace = dict(all_frames)  # shortlist empty/failed - fall back to everything

    schema_parts = [
        f"{var} (from {var_sources[var]}):\n"
        + _describe_table(var_sheet_names[var], namespace[var], sample_rows=config.table_sample_rows)
        for var in namespace
    ]
    if len(namespace) == 1:
        namespace.setdefault("df", next(iter(namespace.values())))

    prompt = _CODEGEN_PROMPT.format(
        question=profile.question,
        schema=truncate_text("\n\n".join(schema_parts), _SCHEMA_CHAR_BUDGET),
    )
    try:
        raw = call_llm(
            prompt,
            model=config.answer_model,
            temperature=0,
            system="Return compact valid JSON only.",
        )
        match = re.search(r"\{[\s\S]*\}", raw or "")
        parsed = json.loads(match.group(0), strict=False) if match else {}
    except Exception as exc:
        LOGGER.warning("Table codegen failed for %s: %s", list(var_sources.values()), exc)
        return None

    expression = str(parsed.get("expression", "") or "").strip()
    if not expression:
        return None

    try:
        result = safe_eval_namespace(expression, namespace)
    except Exception as exc:
        LOGGER.warning("Table expression failed (%s): %s", expression, exc)
        return None
    return {
        "expression": expression,
        "result": _format_result(result),
        "sources": list(dict.fromkeys(var_sources.values())),
    }


def _shortlist_relevant_tables(
    question: str,
    frames: dict[str, pd.DataFrame],
    *,
    config: Approach3Config,
) -> set[str]:
    """Pick the variables plausibly needed to answer the question.

    Full schema (columns + sample rows) for every sheet across many
    multi-sheet files can run past the codegen prompt's character budget,
    silently truncating away the one sheet a join actually needs. This
    builds a compact table of contents - small enough to never hit that
    budget even with dozens of sheets - and asks a cheap model to shortlist
    before the expensive full schema is built. Errors or an empty pick fall
    back to "everything" (the caller's previous behavior) rather than
    narrowing to nothing.

    Small sheets (e.g. a README/index sheet mapping other sheet names to
    descriptions, or a generic single-column "value" sheet) get a few sample
    rows included even in this compact view - their columns alone
    ("Sheet, Description" or just "value") carry no signal, and the row
    *content* is exactly what would tell the shortlist step that this sheet
    documents/contains the answer (observed: a README row literally reading
    "D-SE-acetyl -> Significant genes by acetylproteomics" was invisible to
    the shortlist with columns-only, so the sheet it names got excluded).
    """

    toc_parts = []
    for var, frame in frames.items():
        columns = ", ".join(str(c) for c in frame.columns)
        if len(frame) <= 20:
            sample = frame.head(5).to_string(index=False)
            toc_parts.append(f"{var}: {columns}\n{sample}")
        else:
            toc_parts.append(f"{var}: {columns}")
    toc = "\n".join(toc_parts)
    prompt = _SHORTLIST_PROMPT.format(toc=truncate_text(toc, _SCHEMA_CHAR_BUDGET), question=question)
    try:
        raw = call_llm(
            prompt,
            model=config.analysis_model,
            temperature=0,
            system="Return compact valid JSON only.",
        )
        match = re.search(r"\{[\s\S]*\}", raw or "")
        parsed = json.loads(match.group(0), strict=False) if match else {}
    except Exception as exc:
        LOGGER.warning("Table shortlist failed: %s", exc)
        return set()

    relevant = parsed.get("relevant")
    if not isinstance(relevant, list):
        return set()
    return {str(var) for var in relevant if str(var) in frames}


_ROSTER_GROUP_TERMS = ("project", "du an")
_ROSTER_SIZE_TERMS = ("thanh vien", "member")
_ROSTER_MAX_TERMS = ("most", "highest", "largest")
_ROSTER_MIN_TERMS = ("fewest", "least", "smallest", "lowest")
_ROSTER_EXCLUDE_TERMS = ("sv moi", "new student", "moi tuyen")

_ROSTER_EXTRACT_PROMPT = """
Check first: does this document actually contain a listing of projects/teams,
each with a name/number and a roster of current member names? Many documents
(course slides, quizzes, unrelated reports) do NOT - if this one doesn't,
return {{"projects": []}} and nothing else. Do not invent projects or members
that are not genuinely listed in the document.

Otherwise extract EVERY project as JSON. Some projects also list a separate
count of NEW members (e.g. "+ 2 SV moi", "+1 new student") who should NOT be
counted as current members. For each project, decide "is_core" = true only
if the document's own structure marks it as part of the main/core program
(e.g. listed under a "core program" heading distinct from a "beyond"/extra
section) - false for side/bonus projects. If the document has no such
grouping, set "is_core" to true for every project.

Return ONLY JSON: {{"projects": [{{"project_no": "...", "title": "...",
"is_core": true or false, "members": ["name", ...], "new_member_count": 0}}]}}

Document:
{text}
""".strip()


def _roster_superlative_direction(normalized: str) -> str | None:
    """"max" for "most/highest members", "min" for "fewest/least members", else None.

    English superlatives are fixed phrases ("most", "fewest"...); Vietnamese
    uses the discontinuous "nhieu ... nhat" (most) / "it ... nhat" (fewest)
    construction, so the quantity word and "nhat" are checked separately
    rather than as one adjacent phrase.
    """

    if any(has_word(normalized, term) for term in _ROSTER_MAX_TERMS):
        return "max"
    if any(has_word(normalized, term) for term in _ROSTER_MIN_TERMS):
        return "min"
    if has_word(normalized, "nhat"):
        if has_word(normalized, "it"):
            return "min"
        if has_word(normalized, "nhieu"):
            return "max"
    return None


def is_roster_max_question(question: str) -> bool:
    """True for "which project has the most/fewest current members (excluding
    new students)?"-style questions.

    A single free-text read of a multi-project roster is unreliable: the
    model can pick the wrong project, fold in the "new student" count that
    the question explicitly asks to exclude, or answer the wrong direction
    (max vs min) if asked "fewest" instead of "most". This routes such
    questions to structured per-project extraction + code-side min/max, the
    same split used by compute_scholarship_answer for crowded image tables.
    """

    normalized = normalize_for_match(question)
    has_group = any(has_word(normalized, term) for term in _ROSTER_GROUP_TERMS)
    has_size = any(has_word(normalized, term) for term in _ROSTER_SIZE_TERMS)
    has_superlative = _roster_superlative_direction(normalized) is not None
    has_exclude = any(has_word(normalized, term) for term in _ROSTER_EXCLUDE_TERMS)
    return has_group and has_size and has_superlative and has_exclude


def compute_roster_answer(
    profile: QuestionProfile,
    document_candidates: list[dict[str, Any]],
    *,
    config: Approach3Config,
) -> dict[str, Any] | None:
    """Extract every project's roster from candidate documents, pick the
    min/max current-member count in code (excluding named "new member" counts),
    matching whichever direction ("most" vs "fewest") the question asks for.

    Returns None (caller falls back to the normal per-document text block)
    when the question isn't this shape, no candidate yields usable rosters,
    or no project is marked core.
    """

    direction = _roster_superlative_direction(normalize_for_match(profile.question))
    if not is_roster_max_question(profile.question) or direction is None:
        return None
    if not (config.use_table_compute and has_llm()) or not document_candidates:
        return None

    all_projects: list[dict[str, Any]] = []
    sources: list[str] = []
    for candidate in document_candidates:
        text = extract_candidate_text(candidate)
        if not text.strip():
            continue
        relative_path = str(candidate.get("relative_path", ""))
        try:
            projects = _call_roster_extract(text, model=config.answer_model)
        except Exception as exc:
            LOGGER.warning("Roster extraction failed for %s: %s", relative_path, exc)
            continue
        if projects:
            sources.append(relative_path)
        for project in projects:
            project["_source"] = relative_path
            all_projects.append(project)

    core_projects = [p for p in all_projects if p.get("is_core") is True]
    candidate_projects = core_projects or all_projects
    if not candidate_projects:
        return None

    picker = max if direction == "max" else min
    winner = picker(candidate_projects, key=lambda p: p.get("_current_count", 0))
    if winner.get("_current_count", 0) <= 0:
        return None

    return {
        "answer": str(winner["_current_count"]),
        "direction": direction,
        "winner": winner,
        "projects": all_projects,
        "sources": sources,
    }


def _call_roster_extract(text: str, *, model: str) -> list[dict[str, Any]]:
    raw = call_llm(
        _ROSTER_EXTRACT_PROMPT.format(text=truncate_text(text, 16000)),
        model=model,
        temperature=0,
        system="Return compact valid JSON only.",
    )
    match = re.search(r"\{[\s\S]*\}", raw or "")
    parsed = json.loads(match.group(0), strict=False) if match else {}
    projects = parsed.get("projects")
    if not isinstance(projects, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for project in projects:
        if not isinstance(project, dict):
            continue
        members = project.get("members")
        member_names = [str(m) for m in members] if isinstance(members, list) else []
        cleaned.append(
            {
                "project_no": project.get("project_no", ""),
                "title": normalize_spaces(project.get("title", "")),
                "is_core": bool(project.get("is_core")) if isinstance(project.get("is_core"), bool) else None,
                "members": member_names,
                "_current_count": len(member_names),
            }
        )
    return cleaned


_SAFE_BUILTINS = {
    name: getattr(_builtins, name)
    for name in (
        "len", "sum", "min", "max", "abs", "round", "sorted", "reversed",
        "list", "dict", "set", "tuple", "str", "int", "float", "bool",
        "enumerate", "zip", "range", "any", "all", "map", "filter",
    )
}


def safe_eval_namespace(expression: str, frames: dict[str, pd.DataFrame]) -> Any:
    """Evaluate one screened pandas expression against named DataFrames.

    Blocks dangerous names via _FORBIDDEN (imports, file/network I/O, exec/eval,
    dunder access) but still allows everyday read-only builtins (len, sum, min,
    max, round...) that pandas expressions routinely need.
    """

    if _FORBIDDEN.search(expression):
        raise ValueError(f"Expression rejected by safety screen: {expression}")
    namespace = {"pd": pd, "np": np, **frames}
    return eval(expression, {"__builtins__": _SAFE_BUILTINS}, namespace)  # noqa: S307


def safe_eval_expression(expression: str, frame: pd.DataFrame) -> Any:
    """Evaluate one screened pandas expression against a single DataFrame `df`."""

    return safe_eval_namespace(expression, {"df": frame})


def _unique_identifier(text: str, used: set[str]) -> str:
    base = re.sub(r"\W", "_", text).strip("_") or "t"
    if base[0].isdigit():
        base = f"t_{base}"
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _load_tables(candidate: dict[str, Any]) -> dict[str, pd.DataFrame]:
    absolute = candidate.get("absolute_path")
    if not absolute:
        return {}
    try:
        tables = load_table_file(absolute)
    except Exception as exc:
        LOGGER.warning("Could not load tables from %s: %s", absolute, exc)
        return {}
    return {str(name): frame for name, frame in (tables or {}).items() if isinstance(frame, pd.DataFrame)}


def _describe_table(sheet_name: str, frame: pd.DataFrame, *, sample_rows: int) -> str:
    buffer = io.StringIO()
    buffer.write(f"Sheet: {sheet_name}\n")
    buffer.write(f"Rows: {len(frame)}\n")
    buffer.write("Columns: " + ", ".join(f"{col} ({dtype})" for col, dtype in frame.dtypes.items()) + "\n")
    buffer.write("Sample rows:\n")
    buffer.write(frame.head(sample_rows).to_string(index=False))
    return buffer.getvalue()


def _format_result(result: Any) -> str:
    if isinstance(result, pd.DataFrame):
        return result.head(50).to_string(index=False)
    if isinstance(result, pd.Series):
        return result.head(50).to_string()
    if isinstance(result, float):
        return repr(result)
    return str(result)
