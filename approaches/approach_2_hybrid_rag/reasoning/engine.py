"""Answer generation for approach 2."""

from __future__ import annotations

import json
import re
from typing import Any

from ..shared_src.formatter import normalize_answer
from ..shared_src.llm_client import answer_image_from_file, call_llm, has_llm
from ..shared_src.solvers import solve_context_qa as solve_context_with_existing_reader
from ..shared_src.solvers import solve_image as solve_image_with_existing_reader
from ..shared_src.solvers import solve_sql as solve_sql_with_existing_reader
from ..shared_src.utils import NOT_ENOUGH_DATA, normalize_for_match, normalize_spaces, truncate_text

from .context_builder import build_contexts
from .table_reasoner import try_answer_tables
from ..core.models import AnswerResult, ContextItem, QuestionProfile


def answer_question(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
) -> AnswerResult:
    """Answer one question from retrieved candidates."""

    if profile.modality_hint == "image":
        image_answer = _try_existing_image_answer(profile, candidates)
        if image_answer.answer != NOT_ENOUGH_DATA:
            return image_answer

    sql_answer = _try_existing_sql_answer(profile, candidates)
    if sql_answer.answer != NOT_ENOUGH_DATA:
        return sql_answer

    context_heuristic_answer = _try_existing_context_answer(profile, candidates)
    if context_heuristic_answer.answer != NOT_ENOUGH_DATA:
        return context_heuristic_answer

    if profile.modality_hint in {"table", "cross_file", "auto"}:
        table_answer, table_evidences, table_debug = try_answer_tables(profile, candidates)
        if table_answer != NOT_ENOUGH_DATA:
            return AnswerResult(table_answer, table_evidences, "deterministic_table", table_debug)

    image_answer = _try_direct_image_answer(profile, candidates)
    if image_answer.answer != NOT_ENOUGH_DATA:
        return image_answer

    contexts = build_contexts(candidates)
    if contexts and has_llm():
        llm_answer = _answer_with_llm(profile, contexts)
        if llm_answer.answer != NOT_ENOUGH_DATA:
            return llm_answer

    if contexts:
        return _extractive_fallback(profile, contexts)

    return AnswerResult(NOT_ENOUGH_DATA, [], "no_context", {})


def _try_existing_image_answer(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
) -> AnswerResult:
    images = [item for item in candidates if item.get("modality") == "image"]
    if not images:
        return AnswerResult(NOT_ENOUGH_DATA, [], "skip_existing_image", {})
    try:
        answer, evidences = solve_image_with_existing_reader(profile.question, images)
    except Exception as exc:
        return AnswerResult(NOT_ENOUGH_DATA, [], "existing_image_error", {"error": str(exc)})
    answer = normalize_spaces(answer)
    if not answer or normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA):
        return AnswerResult(NOT_ENOUGH_DATA, [], "existing_image_empty", {})
    return AnswerResult(answer, evidences, "existing_image_reader", {"image_count": len(images)})


def _try_existing_sql_answer(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
) -> AnswerResult:
    if not any(str(item.get("extension", "")).lower() == ".sql" for item in candidates):
        return AnswerResult(NOT_ENOUGH_DATA, [], "skip_existing_sql", {})
    try:
        answer, evidences = solve_sql_with_existing_reader(profile.question, candidates)
    except Exception as exc:
        return AnswerResult(NOT_ENOUGH_DATA, [], "existing_sql_error", {"error": str(exc)})
    answer = normalize_spaces(answer)
    if not answer or normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA):
        return AnswerResult(NOT_ENOUGH_DATA, [], "existing_sql_empty", {})
    return AnswerResult(answer, evidences, "existing_sql_reader", {})


def _try_existing_context_answer(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
) -> AnswerResult:
    normalized = normalize_for_match(profile.question)
    should_try = profile.modality_hint == "audio" or any(
        term in normalized for term in ["project core", "thanh vien", "member", "diem chung", "common"]
    )
    if not should_try:
        return AnswerResult(NOT_ENOUGH_DATA, [], "skip_existing_context", {})
    try:
        answer, evidences = solve_context_with_existing_reader(
            profile.question,
            candidates,
            answer_type=profile.answer_type,
            modality="audio" if profile.modality_hint == "audio" else None,
        )
    except Exception as exc:
        return AnswerResult(NOT_ENOUGH_DATA, [], "existing_context_error", {"error": str(exc)})
    answer = normalize_spaces(answer)
    if not answer or normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA):
        return AnswerResult(NOT_ENOUGH_DATA, [], "existing_context_empty", {})
    return AnswerResult(answer, evidences, "existing_context_reader", {})


def _try_direct_image_answer(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
) -> AnswerResult:
    if profile.modality_hint != "image" or not has_llm():
        return AnswerResult(NOT_ENOUGH_DATA, [], "skip_image_direct", {})
    images = [item for item in candidates if item.get("modality") == "image" and item.get("absolute_path")]
    if not images:
        return AnswerResult(NOT_ENOUGH_DATA, [], "skip_image_direct", {})
    try:
        answer = answer_image_from_file(profile.question, images[0]["absolute_path"])
    except Exception as exc:
        return AnswerResult(NOT_ENOUGH_DATA, [], "image_direct_error", {"error": str(exc)})
    answer = normalize_spaces(answer)
    if not answer or normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA):
        return AnswerResult(NOT_ENOUGH_DATA, [], "image_direct_empty", {})
    return AnswerResult(answer, [images[0]["relative_path"]], "direct_vision", {})


def _answer_with_llm(
    profile: QuestionProfile,
    contexts: list[ContextItem],
) -> AnswerResult:
    context_blocks = "\n\n---\n\n".join(item.to_prompt_block(limit=9000) for item in contexts)
    allowed_sources = [item.relative_path for item in contexts]
    prompt = f"""
Use only the supplied sources to answer the challenge question.
Return valid JSON only:
{{"answer": "...", "evidences": ["relative/path.ext"]}}

Rules:
- Use only evidence paths from the allowed source list.
- If the sources are insufficient, use answer "{NOT_ENOUGH_DATA}" and evidences [].
- For exact-match answers, return only the final value in the answer field.
- For Yes/No questions, answer exactly "Yes" or "No".
- Preserve requested units, rounding, percent signs, casing, and date format.

Question profile:
{json.dumps(profile.to_dict(), ensure_ascii=False)}

Allowed sources:
{json.dumps(allowed_sources, ensure_ascii=False)}

Sources:
{truncate_text(context_blocks, 65000)}
""".strip()
    try:
        raw = call_llm(
            prompt,
            system="You are a data-lake QA system. Return JSON only and never use outside knowledge.",
            temperature=0,
        )
        parsed = _parse_json_object(raw)
    except Exception as exc:
        return AnswerResult(NOT_ENOUGH_DATA, [], "llm_error", {"error": str(exc)})

    answer = normalize_spaces(parsed.get("answer", ""))
    evidences = [str(item) for item in parsed.get("evidences", []) if str(item) in allowed_sources]
    if not answer:
        answer = NOT_ENOUGH_DATA
    if normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA):
        evidences = []
    return AnswerResult(answer, evidences, "llm_context", {"raw": parsed})


def _extractive_fallback(
    profile: QuestionProfile,
    contexts: list[ContextItem],
) -> AnswerResult:
    question_tokens = [token for token in normalize_for_match(profile.question).split() if len(token) > 2]
    best_sentence = ""
    best_score = 0
    best_source = ""

    for item in contexts:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", item.text)
        for sentence in sentences:
            normalized = normalize_for_match(sentence)
            score = sum(1 for token in question_tokens if token in normalized)
            if score > best_score:
                best_score = score
                best_sentence = normalize_spaces(sentence)
                best_source = item.relative_path

    if not best_sentence or best_score == 0:
        return AnswerResult(NOT_ENOUGH_DATA, [], "extractive_no_match", {})

    answer = best_sentence
    if profile.answer_type.lower() == "exact_match" and len(answer) > 240:
        answer = NOT_ENOUGH_DATA
        best_source = ""
    return AnswerResult(answer, [best_source] if best_source else [], "extractive_fallback", {"score": best_score})


def finalize_answer(profile: QuestionProfile, result: AnswerResult) -> AnswerResult:
    """Normalize answer text after reasoning."""

    answer = normalize_answer(
        result.answer,
        question=profile.question,
        answer_type=profile.answer_type,
    )
    evidences = [] if normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA) else result.evidences
    return AnswerResult(answer, list(dict.fromkeys(evidences)), result.strategy, result.debug)


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
