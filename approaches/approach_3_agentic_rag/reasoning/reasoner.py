"""Buoc 4: LLM reasoning over assembled context blocks.

One JSON-only call produces answer + evidences. Computation and cross-file
questions get a chain-of-thought slot inside the JSON ("reasoning" comes
first) so the model thinks before answering while output stays parseable.
Three guards from the pipeline document are enforced: context-only answers,
exact "Not enough data to answer." when context is insufficient, and evidence
paths restricted to the supplied sources. Without an API key an extractive
fallback keeps the pipeline functional.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import Approach3Config
from ..core.models import AnswerResult, ContextBlock, QuestionProfile
from ..shared_src.formatter import normalize_answer
from ..shared_src.llm_client import call_llm, has_llm
from ..shared_src.utils import NOT_ENOUGH_DATA, normalize_for_match, normalize_spaces, truncate_text

LOGGER = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a professional data-lake QA system. Use ONLY the supplied sources; "
    "never use outside knowledge. Return valid JSON only."
)

_ANSWER_PROMPT = """
Answer the question using only the sources below.

Return ONLY a JSON object:
{json_shape}

Rules:
- evidences must be relative paths taken from the allowed source list, and must
  list every file actually used for the answer.
- If the sources are insufficient or irrelevant, set answer exactly to
  "{not_enough}" and evidences to [].
- For exact-match questions return only the final value (no explanation). If
  the question specifies exactly what form the answer should take (e.g. "trả
  lời bằng đúng tên tiêu đề", "return only the ordinal number"), return
  ONLY that literal value - no added labels, numbers, or prefixes citing
  where it came from (e.g. answer "Data Engineering R&D", not "Project 2:
  Data Engineering R&D").
- For Yes/No questions answer exactly "Yes" or "No".
- If the question lists lettered options (A/B/C/D...), this is multiple
  choice: compute/derive the value, match it to the option it equals, and
  answer with ONLY that letter (e.g. "C") - never the raw computed value
  itself, even if it looks like a clean number. This takes priority over the
  digit-form rule below.
- For a numeric exact-match answer that is NOT multiple choice, always write
  it in digit form (e.g. "6"), even if the source text spells the number out
  in words (e.g. "sáu người", "six people") - never copy the spelled-out word
  as the answer.
- If a source's text starts with "Computed ..." (e.g. "Computed
  deterministically", "Computed by checking each item individually",
  "Computed by extracting every ... row"), that result is authoritative - use
  it directly, do not recount, re-derive, or override it from other raw
  per-file text (OCR, captions, free-text vision answers).
- Respect these format requirements: {format_instructions}
- Answer in the same language as the question unless told otherwise.

Question ({answer_type}):
{question}

Allowed sources:
{allowed_sources}

Sources:
{context}
""".strip()

_JSON_SIMPLE = '{"answer": "...", "evidences": ["relative/path.ext"]}'
_JSON_COT = (
    '{"reasoning": "think step by step here first", '
    '"answer": "...", "evidences": ["relative/path.ext"]}'
)


def answer_question(
    profile: QuestionProfile,
    blocks: list[ContextBlock],
    *,
    config: Approach3Config,
) -> AnswerResult:
    """Produce the final answer for one question."""

    if not blocks:
        return AnswerResult(NOT_ENOUGH_DATA, [], "no_context", {})

    if config.use_llm and has_llm():
        result = _answer_with_llm(profile, blocks, config=config)
        if result.strategy == "llm_error":
            # Transient API/parsing hiccups happen; one retry recovers most of
            # them without masking a real, reproducible failure.
            LOGGER.warning("LLM reasoning failed (%s); retrying once.", result.debug.get("error"))
            result = _answer_with_llm(profile, blocks, config=config)
        if result.strategy != "llm_error":
            return result
        LOGGER.warning("LLM reasoning failed twice (%s); using extractive fallback.", result.debug.get("error"))
        fallback = _extractive_fallback(profile, blocks)
        # Keep the real failure reason visible in debug output instead of
        # silently discarding it - "extractive_too_long"/"{}" alone hid why
        # the LLM path was abandoned in the first place.
        fallback.debug["llm_error"] = result.debug.get("error")
        return fallback

    return _extractive_fallback(profile, blocks)


def finalize_answer(
    profile: QuestionProfile,
    result: AnswerResult,
    *,
    valid_paths: set[str],
) -> AnswerResult:
    """Normalize formatting and keep only manifest-valid evidence paths."""

    answer = normalize_answer(
        result.answer,
        question=profile.question,
        answer_type=profile.answer_type,
    )
    evidences = [path for path in result.evidences if path in valid_paths]
    if normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA):
        answer = NOT_ENOUGH_DATA
        evidences = []
    return AnswerResult(answer, list(dict.fromkeys(evidences)), result.strategy, result.debug)


def _answer_with_llm(
    profile: QuestionProfile,
    blocks: list[ContextBlock],
    *,
    config: Approach3Config,
) -> AnswerResult:
    use_cot = profile.requires_computation or profile.needs_multiple_sources
    allowed_sources = list(dict.fromkeys(block.relative_path for block in blocks))
    context = "\n\n---\n\n".join(block.to_prompt_block() for block in blocks)

    prompt = _ANSWER_PROMPT.format(
        json_shape=_JSON_COT if use_cot else _JSON_SIMPLE,
        not_enough=NOT_ENOUGH_DATA,
        format_instructions=json.dumps(profile.format_instructions, ensure_ascii=False),
        answer_type=profile.answer_type or "unknown",
        question=profile.question,
        allowed_sources=json.dumps(allowed_sources, ensure_ascii=False),
        context=truncate_text(context, config.max_context_chars),
    )
    try:
        raw = call_llm(prompt, model=config.answer_model, temperature=0, system=_SYSTEM_PROMPT)
        parsed = _parse_json_object(raw)
    except Exception as exc:
        return AnswerResult(NOT_ENOUGH_DATA, [], "llm_error", {"error": str(exc)})
    if not parsed:
        return AnswerResult(NOT_ENOUGH_DATA, [], "llm_error", {"error": "unparseable response"})

    answer = normalize_spaces(parsed.get("answer", ""))
    evidences = [
        str(item) for item in parsed.get("evidences", []) if str(item) in allowed_sources
    ]
    if not answer:
        answer = NOT_ENOUGH_DATA
    if normalize_for_match(answer) == normalize_for_match(NOT_ENOUGH_DATA):
        evidences = []
    debug: dict[str, Any] = {"used_cot": use_cot}
    if use_cot and parsed.get("reasoning"):
        debug["reasoning"] = truncate_text(str(parsed["reasoning"]), 2000)
    return AnswerResult(answer, evidences, "llm_reasoning", debug)


def _extractive_fallback(
    profile: QuestionProfile,
    blocks: list[ContextBlock],
) -> AnswerResult:
    """Pick the sentence sharing the most tokens with the question."""

    question_tokens = [
        token for token in normalize_for_match(profile.question).split() if len(token) > 2
    ]
    best_sentence, best_score, best_source = "", 0, ""

    for block in blocks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", block.text):
            normalized = normalize_for_match(sentence)
            score = sum(1 for token in question_tokens if token in normalized)
            if score > best_score:
                best_sentence = normalize_spaces(sentence)
                best_score = score
                best_source = block.relative_path

    if not best_sentence or best_score == 0:
        return AnswerResult(NOT_ENOUGH_DATA, [], "extractive_no_match", {})
    if profile.answer_type.lower() == "exact_match" and len(best_sentence) > 240:
        return AnswerResult(NOT_ENOUGH_DATA, [], "extractive_too_long", {})
    return AnswerResult(
        truncate_text(best_sentence, 600),
        [best_source] if best_source else [],
        "extractive_fallback",
        {"score": best_score},
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return {}
    try:
        # strict=False tolerates raw control characters (literal newlines) inside
        # string values - the CoT "reasoning" field often contains multi-line
        # bullet points that models emit unescaped, which would otherwise make
        # an entirely well-formed answer fail to parse.
        parsed = json.loads(match.group(0), strict=False)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
