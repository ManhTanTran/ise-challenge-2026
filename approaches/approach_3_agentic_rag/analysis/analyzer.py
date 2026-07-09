"""Buoc 1: question analysis.

Deterministic regex detections (explicit filenames, wildcard patterns, format
requirements) always run first because they are cheap and exact. On top of
that, one small LLM call classifies language/modality/keywords; its output is
merged into the profile and cached on disk so reruns are free. Without an API
key the heuristic profile alone is used.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..config import Approach3Config
from ..core.models import QuestionProfile
from ..shared_src.llm_client import call_llm, has_llm
from ..shared_src.utils import (
    dump_json,
    has_word,
    load_json,
    normalize_for_match,
    normalize_spaces,
    stable_hash,
)

LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_EXTENSIONS = (".m4a", ".mp3", ".wav")
TABLE_EXTENSIONS = (".csv", ".xlsx", ".xls", ".sql")
DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".pptx", ".ppt", ".txt", ".md", ".html")

STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "what", "which",
    "where", "when", "how", "many", "much", "does", "did", "are", "is",
    "was", "were", "please", "answer", "file", "files",
    "hay", "cho", "toi", "cua", "cac", "nhung", "trong", "nao", "bao",
    "nhieu", "gi", "la", "va", "den", "tren",
}

COMPUTATION_TERMS = {
    "average", "mean", "sum", "total", "count", "how many", "bao nhieu",
    "percentage", "percent", "correlation", "median", "highest", "lowest",
    "max", "min", "rounded", "trung binh", "so luong", "ty le", "tong",
}

MULTI_SOURCE_TERMS = {
    "compare", "across", "all files", "which file", "shared", "common",
    "diem chung", "so sanh", "tat ca", "cross file", "both", "ca hai",
}

_ANALYSIS_PROMPT = """
Analyze one natural-language question for a data-lake QA system.
Return ONLY a JSON object with these keys:
- language: ISO code like "vi", "en", "zh"
- modality_hint: one of "table", "document", "image", "audio", "cross_file", "auto"
- keywords: up to 12 short search keywords/entities from the question (keep original language, add English translations for non-English domain terms)
- requires_computation: true/false (needs arithmetic such as mean/count/filter)
- needs_multiple_sources: true/false (must combine several files)
- format_instructions: object with any of: decimals (int), uppercase (bool), binary (bool, Yes/No), percentage (bool), unit (string), other (string)

Do NOT answer the question itself.

Question:
{question}
""".strip()


def load_analysis_cache(work_dir: str | Path) -> dict[str, Any]:
    return load_json(Path(work_dir) / "analysis_cache.json", default={}) or {}


def save_analysis_cache(work_dir: str | Path, cache: dict[str, Any]) -> None:
    dump_json(cache, Path(work_dir) / "analysis_cache.json")


def analyze_question(
    row: dict[str, Any],
    *,
    config: Approach3Config,
    cache: dict[str, Any] | None = None,
) -> QuestionProfile:
    """Build the routing profile for one question (Buoc 1)."""

    question = str(row.get("question", ""))
    answer_type = str(row.get("answer_type", "") or "")

    profile = _heuristic_profile(row.get("id"), question, answer_type)

    if config.use_llm_analysis and has_llm():
        key = stable_hash(f"{config.analysis_model}::{question}", length=24)
        parsed = (cache or {}).get(key)
        if parsed is None:
            parsed = _llm_analysis(question, model=config.analysis_model)
            if cache is not None and parsed:
                cache[key] = parsed
        if parsed:
            profile = _merge_llm_analysis(profile, parsed)
            profile.analysis_source = "llm+heuristic"
    return profile


def _heuristic_profile(question_id: Any, question: str, answer_type: str) -> QuestionProfile:
    quoted = _quoted_phrases(question)
    file_hints, wildcards = _explicit_file_hints(question)
    profile = QuestionProfile(
        question_id=question_id,
        question=question,
        answer_type=answer_type,
        language=_detect_language(question),
        quoted_phrases=quoted,
        explicit_file_hints=file_hints,
        wildcard_patterns=wildcards,
        requires_computation=_contains_any(question, COMPUTATION_TERMS),
        needs_multiple_sources=_contains_any(question, MULTI_SOURCE_TERMS),
    )
    profile.modality_hint = _modality_hint(question, profile)
    profile.keywords = _keywords(question, profile)
    profile.format_instructions = _format_instructions(question, answer_type)
    return profile


def _llm_analysis(question: str, *, model: str) -> dict[str, Any]:
    try:
        raw = call_llm(
            _ANALYSIS_PROMPT.format(question=question),
            model=model,
            temperature=0,
            system="Return compact valid JSON only. Never answer the question.",
        )
        match = re.search(r"\{[\s\S]*\}", raw or "")
        if not match:
            return {}
        parsed = json.loads(match.group(0), strict=False)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as exc:
        LOGGER.warning("LLM question analysis failed: %s", exc)
        return {}


def _merge_llm_analysis(profile: QuestionProfile, parsed: dict[str, Any]) -> QuestionProfile:
    if isinstance(parsed.get("language"), str) and parsed["language"]:
        profile.language = parsed["language"][:5]
    hint = str(parsed.get("modality_hint", "") or "")
    if hint in {"table", "document", "image", "audio", "cross_file", "auto"}:
        # Regex file hints beat the LLM guess: an explicit ".csv" in the
        # question is stronger evidence than semantic classification.
        if not profile.explicit_file_hints or hint != "auto":
            profile.modality_hint = hint or profile.modality_hint
    if isinstance(parsed.get("keywords"), list):
        merged = profile.keywords + [normalize_spaces(item) for item in parsed["keywords"]]
        profile.keywords = list(dict.fromkeys(item for item in merged if item))[:30]
    if isinstance(parsed.get("requires_computation"), bool):
        profile.requires_computation = profile.requires_computation or parsed["requires_computation"]
    if isinstance(parsed.get("needs_multiple_sources"), bool):
        profile.needs_multiple_sources = (
            profile.needs_multiple_sources or parsed["needs_multiple_sources"]
        )
    if isinstance(parsed.get("format_instructions"), dict):
        merged_format = dict(parsed["format_instructions"])
        merged_format.update(profile.format_instructions)  # regex wins on conflicts
        profile.format_instructions = merged_format
    return profile


def _detect_language(question: str) -> str:
    if re.search(r"[一-鿿]", question):
        return "zh"
    if re.search(
        r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
        question.lower(),
    ):
        return "vi"
    normalized = normalize_for_match(question)
    if any(f" {term} " in f" {normalized} " for term in ["hay", "cho toi", "bao nhieu", "la gi"]):
        return "vi"
    return "en"


def _quoted_phrases(question: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"[\"“”']([^\"“”']+)[\"“”']", question)
        if match.group(1).strip()
    ]


def _explicit_file_hints(question: str) -> tuple[list[str], list[str]]:
    extensions = IMAGE_EXTENSIONS + AUDIO_EXTENSIONS + TABLE_EXTENSIONS + DOCUMENT_EXTENSIONS
    extension_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in extensions)
    hints: list[str] = []
    wildcards: list[str] = []
    # Non-space tokens only: filenames containing spaces are still caught via
    # quoted phrases and the filename-in-question check during retrieval.
    for match in re.finditer(
        rf"([A-Za-z0-9_\-\[\]./\\一-鿿]+\.({extension_pattern}))\b",
        question,
        flags=re.IGNORECASE,
    ):
        hints.append(normalize_spaces(match.group(1)).replace("\\", "/"))
    for phrase in _quoted_phrases(question):
        if re.search(rf"\.({extension_pattern})$", phrase, flags=re.IGNORECASE):
            hints.append(phrase.replace("\\", "/"))
    for match in re.finditer(r"([A-Za-z0-9_\-\[\]\(\)&./\\]+/\*)", question):
        wildcards.append(normalize_spaces(match.group(1)).replace("\\", "/"))
    return list(dict.fromkeys(hints)), list(dict.fromkeys(wildcards))


def _contains_any(question: str, terms: set[str]) -> bool:
    normalized = normalize_for_match(question)
    return any(has_word(normalized, term) for term in terms)


def _modality_hint(question: str, profile: QuestionProfile) -> str:
    normalized = normalize_for_match(question)
    hints = " ".join(profile.explicit_file_hints + profile.wildcard_patterns).lower()
    if any(ext in hints for ext in IMAGE_EXTENSIONS) or any(
        has_word(normalized, term) for term in ["image", "picture", "photo", "ocr", "digit", "anh", "hinh"]
    ):
        return "image"
    if any(ext in hints for ext in AUDIO_EXTENSIONS) or any(
        has_word(normalized, term) for term in ["audio", "voice", "transcript", "recording", "ghi am"]
    ):
        return "audio"
    if any(ext in hints for ext in TABLE_EXTENSIONS) or profile.requires_computation or any(
        has_word(normalized, term) for term in ["csv", "xlsx", "table", "sheet", "sql", "bang"]
    ):
        return "table"
    if profile.needs_multiple_sources:
        return "cross_file"
    if any(ext in hints for ext in DOCUMENT_EXTENSIONS):
        return "document"
    return "auto"


def _keywords(question: str, profile: QuestionProfile) -> list[str]:
    normalized = normalize_for_match(question)
    tokens = re.findall(r"[一-鿿]+|[a-z0-9_./*\-]+", normalized)
    keywords: list[str] = []
    for phrase in profile.quoted_phrases + profile.explicit_file_hints + profile.wildcard_patterns:
        if phrase and phrase not in keywords:
            keywords.append(phrase)
    for token in tokens:
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in keywords:
            keywords.append(token)
    return keywords[:30]


def _format_instructions(question: str, answer_type: str) -> dict[str, Any]:
    normalized = normalize_for_match(question)
    instructions: dict[str, Any] = {"answer_type": answer_type or "unknown"}
    if "yes/no" in normalized or re.match(r"^(did|do|does|is|are|was|were)\b", normalized):
        instructions["binary"] = True
    if "uppercase" in normalized or "chu hoa" in normalized:
        instructions["uppercase"] = True
    if "percent" in normalized or "percentage" in normalized:
        instructions["percentage"] = True
        instructions["include_percent_sign"] = "%" in question
    match = re.search(r"(?:rounded to|round to|lam tron.{0,10})\s*(\d+)\s*(?:decimal|chu so)", normalized)
    if match:
        instructions["decimals"] = int(match.group(1))
    else:
        for words, value in (("one", 1), ("two", 2), ("three", 3)):
            if f"{words} decimal" in normalized or f"{value} decimal" in normalized:
                instructions["decimals"] = value
                break
    return instructions
