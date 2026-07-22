"""Buoc 3c: on-demand vision QA over image candidates, cached on disk.

Three distinct vision paths live here:
- `vision_answer`: answers the question AS-IS against one image. Correct for
  "which scholarship does this image describe?"-style questions where the
  retrieved image genuinely is the single subject of the question.
- `count_matching_images`: for "how many images in <folder> satisfy X?"
  questions, asking each image the ORIGINAL aggregate question is meaningless
  (an image can't say how many OTHER images match). The question is rewritten
  once into a per-image Yes/No criterion, each image is asked that instead,
  and the match count is computed in code - mirroring readers/table.py's
  code-first philosophy: LLM judges one item, code does the counting.
- `compute_scholarship_answer`: for "which scholarship has the most slots?"
  questions, a single free-text vision answer is unreliable when one image
  packs a dozen scholarship rows into a crowded table (the model, or the
  noisy OCR text sitting next to it in context, latches onto the wrong row).
  Each candidate image is asked to extract every row as structured JSON, and
  code picks the row with the maximum slot count - the same "LLM reads one
  item, code aggregates" split as count_matching_images.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from ..config import Approach3Config
from ..core.models import QuestionProfile
from ..shared_src.llm_client import OPENROUTER_BASE_URL, call_llm, has_llm
from ..shared_src.utils import (
    NOT_ENOUGH_DATA,
    dump_json,
    ensure_dir,
    has_word,
    load_json,
    normalize_for_match,
    normalize_spaces,
    stable_hash,
)

LOGGER = logging.getLogger(__name__)

_REWRITE_PROMPT = """
Rewrite this question about MULTIPLE files in a folder into a Yes/No question
about a SINGLE file, preserving the exact filtering criterion (colors, counts,
content, etc.) from the original.

Return ONLY JSON: {{"per_item_question": "..."}}

Original question (about a whole folder):
{question}

Example:
Original: "How many images in 'number_image' contain a blue digit?"
Rewritten: "Does this image show a blue digit?"
""".strip()

_STRUCTURED_VISION_PROMPT = """
Answer this Yes/No question about the image using what is visible.

If the question asks about a digit/number's color and the digit is styled as
a solid-color icon or badge (e.g. a circle, square, or other flat shape with
the digit cut out in a contrasting color, like a white numeral inside a
colored circle), the badge/icon's dominant color IS the digit's color for
this purpose - even though the glyph strokes themselves are a different,
contrasting color. This is not optional or a matter of interpretation: you
must answer as if the digit itself were that color.
Example: a white "7" cut out of a solid blue circle badge -> answer as if
asked about a blue digit: "Does this image contain/show a blue digit?" -> Yes.

{caption_hint}Return ONLY JSON: {{"matches": true or false, "reason": "short justification"}}

Question: {question}
""".strip()

_CAPTION_HINT = (
    'An automated caption of this image (from indexing) says: "{caption}". '
    "Use it as a hint but trust the actual image if they disagree.\n"
)

_SCHOLARSHIP_TERMS = ("hoc bong", "scholarship")
_SLOT_TERMS = ("suat", "so luong", "slot")

_SCHOLARSHIP_EXTRACT_PROMPT = """
Check first: does this image contain a table/poster listing multiple named
scholarships, where EACH scholarship has an explicit COUNT of awarded slots
(labeled things like "So luong", "Suat", "SV", "HS", or similar - a number of
people/awards, NOT a currency amount, percentage, or tuition value)? If the
image is a single scholarship announcement, an unrelated topic, or has no
such per-row slot count, return {{"rows": []}} and nothing else.

Otherwise extract EVERY scholarship row you can read, even if the layout is
crowded or the text is small. Skip any row whose only numbers are money,
dates, or percentages rather than a slot count.

Some tables split rows into two groups: scholarships funded by the state
budget ("trong ngan sach") versus scholarships funded externally - by a
company, country, or organization ("ngoai ngan sach", often the row also
names a country like "Nhat Ban" or "Han Quoc"). Set "off_budget" to true only
for rows in the externally-funded group; set it to false for state-budget
rows. If the image has no such grouping, set "off_budget" to true for every
row.

{caption_hint}Return ONLY JSON: {{"rows": [{{"scholarship_name": "...",
"country": "... or empty string", "off_budget": true or false,
"slot_components": [<slot-count numbers found for this row, e.g. separate
counts per category - never money/dates/percentages>],
"confidence": "high|medium|low"}}]}}
""".strip()


def is_scholarship_slot_question(question: str) -> bool:
    """True for "which scholarship has the most slots?"-style questions.

    A single free-text vision answer is unreliable when one image packs a
    dozen scholarship rows into a crowded table - the model (or noisy OCR
    text sitting next to it in context) latches onto the wrong row. This
    routes such questions to structured per-row extraction + code-side max
    instead, mirroring the count_matching_images code-first split.
    """

    normalized = normalize_for_match(question)
    has_scholarship = any(has_word(normalized, term) for term in _SCHOLARSHIP_TERMS)
    has_slot = any(has_word(normalized, term) for term in _SLOT_TERMS)
    return has_scholarship and has_slot


def vision_answer(
    profile: QuestionProfile,
    candidate: dict[str, Any],
    *,
    config: Approach3Config,
    cache_dir: str | Path,
) -> str | None:
    """Ask the vision model about one image; results are cached per (image, question)."""

    if not (config.use_vision and has_llm()):
        return None
    absolute = candidate.get("absolute_path")
    if not absolute or not Path(absolute).exists():
        return None

    cache_root = ensure_dir(cache_dir)
    caption = _candidate_caption(candidate)
    key = stable_hash(
        f"{config.vision_model}::{candidate.get('relative_path')}::{profile.question}::{caption}",
        length=24,
    )
    cache_path = cache_root / f"{key}.json"
    cached = load_json(cache_path, default=None)
    if isinstance(cached, dict) and "answer" in cached:
        return str(cached["answer"])

    try:
        answer = _call_vision(
            profile.question, Path(absolute), model=config.vision_model, caption=caption
        )
    except Exception as exc:
        LOGGER.warning("Vision QA failed for %s: %s", candidate.get("relative_path"), exc)
        return None
    dump_json(
        {
            "question": profile.question,
            "image": candidate.get("relative_path"),
            "model": config.vision_model,
            "answer": answer,
        },
        cache_path,
    )
    return answer


def count_matching_images(
    profile: QuestionProfile,
    image_candidates: list[dict[str, Any]],
    *,
    config: Approach3Config,
    cache_dir: str | Path,
) -> dict[str, Any] | None:
    """Count how many images satisfy the question's criterion, computed in code.

    Returns None (caller falls back to the per-image caption-only blocks) when
    vision is disabled, there are no images, or the question rewrite fails.
    """

    if not (config.use_vision_count_compute and has_llm()) or not image_candidates:
        return None

    per_item_question = _rewrite_per_item_question(profile.question, model=config.analysis_model)
    if not per_item_question:
        return None

    cache_root = ensure_dir(cache_dir)
    matched_files: list[str] = []
    evaluated = 0
    for candidate in image_candidates:
        absolute = candidate.get("absolute_path")
        relative_path = str(candidate.get("relative_path", ""))
        if not absolute or not Path(absolute).exists():
            continue

        caption = _candidate_caption(candidate)
        # Caption is part of the key so a newly-added/changed index caption
        # re-judges instead of serving a stale caption-less verdict.
        key = stable_hash(
            f"{config.vision_model}::count_v3::{relative_path}::{per_item_question}::{caption}",
            length=24,
        )
        cache_path = cache_root / f"{key}.json"
        cached = load_json(cache_path, default=None)
        if isinstance(cached, dict) and "matches" in cached:
            result = cached
        else:
            try:
                result = _call_vision_structured(
                    per_item_question, Path(absolute), model=config.vision_model, caption=caption
                )
            except Exception as exc:
                LOGGER.warning("Structured vision QA failed for %s: %s", relative_path, exc)
                continue
            dump_json(
                {"image": relative_path, "criterion": per_item_question, **result},
                cache_path,
            )

        if result.get("matches") is None:
            continue
        evaluated += 1
        if result["matches"] is True:
            matched_files.append(relative_path)

    if evaluated == 0:
        return None
    return {
        "criterion": per_item_question,
        "total": evaluated,
        "matched_count": len(matched_files),
        "matched_files": matched_files,
    }


def compute_scholarship_answer(
    profile: QuestionProfile,
    image_candidates: list[dict[str, Any]],
    *,
    config: Approach3Config,
    cache_dir: str | Path,
) -> dict[str, Any] | None:
    """Extract scholarship rows from every candidate image, pick the max in code.

    Returns None (caller falls back to the normal per-image vision/OCR blocks)
    when the question isn't this shape, vision is disabled, or no candidate
    yields any usable row - a crowded, unrelated, or unreadable image should
    never fabricate a winner.
    """

    if not is_scholarship_slot_question(profile.question):
        return None
    if not (config.use_vision and has_llm()) or not image_candidates:
        return None

    cache_root = ensure_dir(cache_dir)
    all_rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for candidate in image_candidates:
        absolute = candidate.get("absolute_path")
        relative_path = str(candidate.get("relative_path", ""))
        if not absolute or not Path(absolute).exists():
            continue

        caption = _candidate_caption(candidate)
        key = stable_hash(
            f"{config.vision_model}::scholarship_rows_v2::{relative_path}::{caption}",
            length=24,
        )
        cache_path = cache_root / f"{key}.json"
        cached = load_json(cache_path, default=None)
        if isinstance(cached, dict) and "rows" in cached:
            rows = cached["rows"]
        else:
            try:
                rows = _call_vision_rows(Path(absolute), model=config.vision_model, caption=caption)
            except Exception as exc:
                LOGGER.warning("Scholarship row extraction failed for %s: %s", relative_path, exc)
                continue
            dump_json({"image": relative_path, "rows": rows}, cache_path)

        if rows:
            sources.append(relative_path)
        for row in rows:
            row["_source"] = relative_path
            all_rows.append(row)

    off_budget_rows = [row for row in all_rows if row.get("off_budget") is True]
    candidate_rows = off_budget_rows or all_rows
    if not candidate_rows:
        return None

    winner = max(candidate_rows, key=lambda row: row.get("_slot_count", 0.0))
    if winner.get("_slot_count", 0.0) <= 0 or not winner.get("scholarship_name"):
        return None

    return {
        "answer": _clean_scholarship_name(str(winner.get("scholarship_name", ""))),
        "winner": winner,
        "rows": all_rows,
        "sources": sources,
    }


def _clean_scholarship_name(name: str) -> str:
    # Rows are formatted "NAME, Country" (e.g. "SHINNYO, Nhật Bản") - the
    # question asks for the name only, uppercased, with no country.
    name = name.split(",")[0]
    return normalize_spaces(name).upper()


def _call_vision_rows(path: Path, *, model: str, caption: str = "") -> list[dict[str, Any]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    caption_hint = _CAPTION_HINT.format(caption=caption) if caption else ""

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return compact valid JSON only."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _SCHOLARSHIP_EXTRACT_PROMPT.format(caption_hint=caption_hint),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            },
        ],
        temperature=0,
    )
    parsed = _parse_json_object(response.choices[0].message.content or "")
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        components = row.get("slot_components")
        numbers = _extract_numbers(components) if isinstance(components, list) else []
        cleaned.append(
            {
                "scholarship_name": normalize_spaces(row.get("scholarship_name", "")),
                "country": normalize_spaces(row.get("country", "")),
                "off_budget": bool(row.get("off_budget")) if isinstance(row.get("off_budget"), bool) else None,
                "slot_components": numbers,
                "_slot_count": sum(numbers),
                "confidence": row.get("confidence", ""),
            }
        )
    return cleaned


def _extract_numbers(values: list[Any]) -> list[float]:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            numbers.append(float(value))
            continue
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        if match:
            numbers.append(float(match.group(0)))
    return numbers


def _rewrite_per_item_question(question: str, *, model: str) -> str | None:
    try:
        raw = call_llm(
            _REWRITE_PROMPT.format(question=question),
            model=model,
            temperature=0,
            system="Return compact valid JSON only.",
        )
        parsed = _parse_json_object(raw)
    except Exception as exc:
        LOGGER.warning("Per-item question rewrite failed: %s", exc)
        return None
    rewritten = normalize_spaces(parsed.get("per_item_question", ""))
    return rewritten or None


def _candidate_caption(candidate: dict[str, Any]) -> str:
    """Bước 0 indexed caption/description for one image, if any."""

    parts = [candidate.get("image_caption"), candidate.get("image_description")]
    return normalize_spaces(" ".join(str(p) for p in parts if p))


def _call_vision_structured(
    question: str, path: Path, *, model: str, caption: str = ""
) -> dict[str, Any]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    caption_hint = _CAPTION_HINT.format(caption=caption) if caption else ""

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return compact valid JSON only."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _STRUCTURED_VISION_PROMPT.format(
                            question=question, caption_hint=caption_hint
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            },
        ],
        temperature=0,
    )
    parsed = _parse_json_object(response.choices[0].message.content or "")
    matches = parsed.get("matches")
    return {
        "matches": bool(matches) if isinstance(matches, bool) else None,
        "reason": normalize_spaces(parsed.get("reason", "")),
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0), strict=False)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _call_vision(question: str, path: Path, *, model: str, caption: str = "") -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    caption_line = f"\nIndexed caption of this image (a hint): {caption}" if caption else ""

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using the image (a caption hint may be provided). Be concise "
                    f"and literal. If the image is insufficient, answer exactly: {NOT_ENOUGH_DATA}"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Question: {question}{caption_line}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            },
        ],
        temperature=0,
    )
    return normalize_spaces(response.choices[0].message.content or "")
