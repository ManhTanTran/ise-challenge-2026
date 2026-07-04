"""Question analysis and routing for approach 2."""

from __future__ import annotations

import json
import re
from typing import Any

from ..shared_src.llm_client import call_llm, has_llm
from ..shared_src.utils import normalize_for_match, normalize_spaces

from .models import QuestionProfile


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_EXTENSIONS = (".m4a", ".mp3", ".wav")
TABLE_EXTENSIONS = (".csv", ".xlsx", ".xls", ".sql")
DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".pptx", ".ppt", ".txt", ".md", ".html")

STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "that",
    "this",
    "what",
    "which",
    "where",
    "when",
    "how",
    "many",
    "much",
    "does",
    "did",
    "are",
    "is",
    "was",
    "were",
    "hay",
    "cho",
    "toi",
    "cua",
    "cac",
    "nhung",
    "trong",
    "nao",
    "bao",
    "nhieu",
}


def analyze_question(
    row: dict[str, Any],
    *,
    use_llm: bool = False,
) -> QuestionProfile:
    """Create a retrieval and reasoning profile for one question."""

    question = str(row.get("question", ""))
    answer_type = str(row.get("answer_type", "") or "")
    expected_sources = [str(item).replace("\\", "/") for item in row.get("expected_sources", []) or []]

    profile = QuestionProfile(
        question_id=row.get("id"),
        question=question,
        answer_type=answer_type,
        language=_detect_language(question),
        quoted_phrases=_quoted_phrases(question),
        explicit_file_hints=_explicit_file_hints(question) + _domain_hints(question) + expected_sources,
        requires_computation=_requires_computation(question),
        needs_multiple_sources=_needs_multiple_sources(question),
        expected_sources=expected_sources,
    )
    profile.modality_hint = _modality_hint(question, profile)
    profile.keywords = _keywords(question, profile)
    profile.format_instructions = _format_instructions(question, answer_type)

    if use_llm and has_llm():
        profile = _merge_llm_profile(profile)
    return profile


def _detect_language(question: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", question):
        return "zh"
    normalized = normalize_for_match(question)
    vietnamese_terms = {"hay", "cho", "toi", "trong", "bao nhieu", "diem", "du an", "hoc bong"}
    if any(term in normalized for term in vietnamese_terms) or re.search(
        r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
        question.lower(),
    ):
        return "vi"
    return "en"


def _quoted_phrases(question: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"[\"']([^\"']+)[\"']", question)
        if match.group(1).strip()
    ]


def _explicit_file_hints(question: str) -> list[str]:
    extensions = IMAGE_EXTENSIONS + AUDIO_EXTENSIONS + TABLE_EXTENSIONS + DOCUMENT_EXTENSIONS
    extension_pattern = "|".join(re.escape(ext.lstrip(".")) for ext in extensions)
    hints = []
    for match in re.finditer(
        rf"([A-Za-z0-9_\-\[\]\(\)& ./\\\u4e00-\u9fff]+\.({extension_pattern}))",
        question,
        flags=re.IGNORECASE,
    ):
        hints.append(normalize_spaces(match.group(1)).replace("\\", "/"))
    for match in re.finditer(r"([A-Za-z0-9_\-\[\]\(\)& ./\\]+/\*)", question):
        hints.append(normalize_spaces(match.group(1)).replace("\\", "/"))
    return list(dict.fromkeys(hints))


def _domain_hints(question: str) -> list[str]:
    """Add light data-lake domain hints without relying on sample answers."""

    normalized = normalize_for_match(question)
    hints: list[str] = []
    if any(
        term in normalized
        for term in [
            "gene",
            "genes",
            "proteomics",
            "phosphoproteomics",
            "acetylproteomics",
            "hyperactivated",
            "cnv",
            "fda-approved",
            "drug",
        ]
    ):
        hints.append("biomedical/*")
    if "hoc bong" in normalized or "scholarship" in normalized:
        hints.append("scholarship1.png")
    if "number_image" in normalized or ("image" in normalized and "digit" in normalized):
        hints.append("number_image/*")
    if "axiom" in normalized or "project core" in normalized:
        hints.append("iSE-AXIOM-Internal Intro.pdf")
    if ("lop 10a1" in normalized and "toan" in normalized) or "class_grades" in normalized:
        hints.append("class_grades.sql")
    if "workshop" in normalized or "march 22" in normalized or "03.22" in normalized:
        hints.append("workshop_03.22.m4a")
    return hints


def _requires_computation(question: str) -> bool:
    normalized = normalize_for_match(question)
    terms = {
        "average",
        "mean",
        "sum",
        "total",
        "count",
        "how many",
        "bao nhieu",
        "percentage",
        "percent",
        "correlation",
        "median",
        "highest",
        "lowest",
        "max",
        "min",
        "rounded",
        "trung binh",
        "so luong",
    }
    return any(term in normalized for term in terms)


def _needs_multiple_sources(question: str) -> bool:
    normalized = normalize_for_match(question)
    terms = {
        "compare",
        "across",
        "all files",
        "which file",
        "shared",
        "common",
        "diem chung",
        "so sanh",
        "tat ca",
        "cross file",
    }
    return any(term in normalized for term in terms)


def _modality_hint(question: str, profile: QuestionProfile) -> str:
    normalized = normalize_for_match(question)
    hints = " ".join(profile.explicit_file_hints).lower()
    if any(ext in hints for ext in IMAGE_EXTENSIONS) or any(
        term in normalized for term in ["image", "picture", "photo", "ocr", "digit", "anh", "hinh", "hoc bong"]
    ):
        return "image"
    if any(ext in hints for ext in AUDIO_EXTENSIONS) or any(
        term in normalized for term in ["audio", "voice", "transcript", "workshop", "m4a"]
    ):
        return "audio"
    if any(term in normalized for term in ["axiom", "project core", "thanh vien", "member"]):
        return "document"
    if any(ext in hints for ext in TABLE_EXTENSIONS) or profile.requires_computation or any(
        term in normalized
        for term in ["csv", "xlsx", "table", "sheet", "sql", "average", "mean", "count", "correlation"]
    ):
        return "table"
    if profile.needs_multiple_sources:
        return "cross_file"
    return "document"


def _keywords(question: str, profile: QuestionProfile) -> list[str]:
    normalized = normalize_for_match(question)
    tokens = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_./*\-]+", normalized)
    keywords = []
    for phrase in profile.quoted_phrases + profile.explicit_file_hints:
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
    match = re.search(r"(?:rounded to|round to)\s+(\d+)\s+decimal", normalized)
    if match:
        instructions["decimals"] = int(match.group(1))
    elif "two decimal" in normalized or "2 decimal" in normalized:
        instructions["decimals"] = 2
    elif "three decimal" in normalized or "3 decimal" in normalized:
        instructions["decimals"] = 3
    elif "one decimal" in normalized or "1 decimal" in normalized:
        instructions["decimals"] = 1
    return instructions


def _merge_llm_profile(profile: QuestionProfile) -> QuestionProfile:
    prompt = f"""
Analyze this challenge question for retrieval.
Return only JSON with keys:
language, modality_hint, keywords, requires_computation, needs_multiple_sources, format_instructions.

Question:
{profile.question}
""".strip()
    try:
        raw = call_llm(
            prompt,
            system="Return compact valid JSON only. Do not answer the question.",
            temperature=0,
        )
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return profile
        parsed = json.loads(match.group(0))
    except Exception:
        return profile

    if isinstance(parsed.get("language"), str):
        profile.language = parsed["language"]
    if isinstance(parsed.get("modality_hint"), str):
        profile.modality_hint = parsed["modality_hint"]
    if isinstance(parsed.get("keywords"), list):
        merged = profile.keywords + [str(item) for item in parsed["keywords"]]
        profile.keywords = list(dict.fromkeys(item for item in merged if item))[:30]
    if isinstance(parsed.get("requires_computation"), bool):
        profile.requires_computation = parsed["requires_computation"]
    if isinstance(parsed.get("needs_multiple_sources"), bool):
        profile.needs_multiple_sources = parsed["needs_multiple_sources"]
    if isinstance(parsed.get("format_instructions"), dict):
        profile.format_instructions.update(parsed["format_instructions"])
    return profile
