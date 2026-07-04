"""Answer formatting and validation helpers."""

from __future__ import annotations

import math
import re
from typing import Any

from .utils import NOT_ENOUGH_DATA, normalize_for_match, normalize_spaces


def normalize_answer(
    answer: Any,
    *,
    question: str = "",
    answer_type: str | None = None,
) -> str:
    """Normalize a raw answer for submission."""

    text = normalize_spaces(answer)
    if not text:
        return NOT_ENOUGH_DATA
    if normalize_for_match(text) == normalize_for_match(NOT_ENOUGH_DATA):
        return NOT_ENOUGH_DATA

    multiple_choice = normalize_multiple_choice(text, question)
    if multiple_choice is not None:
        return multiple_choice

    yes_no = normalize_yes_no(text, question)
    if yes_no is not None:
        return yes_no

    numeric = normalize_numeric(text, question)
    if numeric is not None:
        return numeric

    if "uppercase" in question.lower() or "chu hoa" in normalize_for_match(question):
        text = text.upper()

    if answer_type and answer_type.lower() == "exact_match":
        text = strip_common_answer_prefix(text)
    return text


def normalize_yes_no(answer: str, question: str = "") -> str | None:
    """Normalize binary answers only when the signal is clear."""

    lowered = normalize_for_match(answer)
    compact = re.sub(r"[^a-z]+", "", lowered)
    question_lowered = normalize_for_match(question)
    if compact in {"yes", "y", "true", "co", "dung"}:
        return "Yes"
    if compact in {"no", "n", "false", "khong", "sai"}:
        return "No"
    if question_lowered.startswith(("did ", "do ", "does ", "is ", "are ", "was ", "were ")):
        if re.search(r"\byes\b", lowered):
            return "Yes"
        if re.search(r"\bno\b", lowered):
            return "No"
    return None


def normalize_multiple_choice(answer: str, question: str = "") -> str | None:
    """Normalize A/B/C/D answers for multiple-choice questions."""

    question_has_options = bool(re.search(r"(?im)^\s*[A-D][.)]\s+", question))
    stripped = answer.strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped
    if question_has_options:
        match = re.search(r"\b([A-D])\b", stripped)
        if match:
            return match.group(1)
    return None


def normalize_numeric(answer: str, question: str = "") -> str | None:
    """Format numeric answers according to wording in the question."""

    number_match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer)
    if not number_match:
        return None
    raw = number_match.group(0).replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    question_lowered = normalize_for_match(question)
    numeric_intent = any(
        term in question_lowered
        for term in {
            "how many",
            "bao nhieu",
            "correlation",
            "average",
            "mean",
            "sum",
            "count",
            "percentage",
            "rounded",
            "number",
            "total",
        }
    )
    leftover = answer.replace(number_match.group(0), "")
    if re.search(r"[A-Za-z\u00C0-\u1EF9]", leftover) and not numeric_intent:
        return None

    decimals = _rounding_decimals(question_lowered)
    if decimals is not None:
        formatted = f"{value:.{decimals}f}"
    elif value.is_integer() and "." not in raw:
        formatted = str(int(value))
    elif value.is_integer() and "integer" in question_lowered:
        formatted = str(int(value))
    else:
        formatted = str(value).rstrip("0").rstrip(".")

    if "percentage" in question_lowered or "percent" in question_lowered:
        wants_symbol = "%" in question or "include %" in question_lowered
        if wants_symbol and "%" not in formatted:
            formatted += "%"
    return formatted


def strip_common_answer_prefix(answer: str) -> str:
    """Remove common LLM answer prefixes from exact answers."""

    return re.sub(
        r"^(final answer|answer|dap an|tra loi)\s*[:：]\s*",
        "",
        answer.strip(),
        flags=re.IGNORECASE,
    )


def exact_match(predicted: Any, expected: Any) -> bool:
    """Normalized exact-match comparison."""

    return normalize_for_match(predicted) == normalize_for_match(expected)


def format_float(value: float, decimals: int | None = None) -> str:
    """Format a float without spurious trailing zeros unless decimals is fixed."""

    if value is None or math.isnan(value):
        return NOT_ENOUGH_DATA
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.10g}"


def _rounding_decimals(question_lowered: str) -> int | None:
    if "two decimal" in question_lowered or "2 decimal" in question_lowered:
        return 2
    if "three decimal" in question_lowered or "3 decimal" in question_lowered:
        return 3
    if "one decimal" in question_lowered or "1 decimal" in question_lowered:
        return 1
    match = re.search(r"rounded to (\d+) decimal", question_lowered)
    if match:
        return int(match.group(1))
    return None
