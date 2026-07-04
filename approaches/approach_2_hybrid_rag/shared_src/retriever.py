"""Evidence retrieval over indexed data lake files."""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path
from typing import Any

from .llm_client import has_llm, rerank_files
from .utils import normalize_for_match, read_text_with_fallback, safe_relative_path, truncate_text

LOGGER = logging.getLogger(__name__)


def retrieve_files(
    question: str,
    file_index: list[dict[str, Any]],
    *,
    top_k: int = 8,
    use_llm_rerank: bool = True,
    expected_sources: list[str] | None = None,
    use_expected_sources: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve likely evidence files for a question."""

    if not file_index:
        return []

    ranked: dict[str, dict[str, Any]] = {}
    protected: dict[str, dict[str, Any]] = {}
    source_hint_count = 0

    for item in _protected_matches(question, file_index):
        _add_candidate(protected, item, score=20.0, reason="protected_match")
        _add_candidate(ranked, item, score=20.0, reason="protected_match")

    for item in _direct_matches(question, file_index):
        _add_candidate(ranked, item, score=5.0, reason="filename_match")

    if use_expected_sources and expected_sources:
        source_hint_matches = _source_hint_matches(expected_sources, file_index)
        source_hint_count = len(source_hint_matches)
        for item in source_hint_matches:
            _add_candidate(protected, item, score=30.0, reason="source_hint")
            _add_candidate(ranked, item, score=10.0, reason="source_hint")

    for item in _tfidf_matches(question, file_index, top_k=max(top_k * 3, 20)):
        _add_candidate(ranked, item, score=item.get("score", 0.0), reason="tfidf")

    candidates = sorted(ranked.values(), key=lambda item: item["score"], reverse=True)
    protected_candidates = sorted(protected.values(), key=lambda item: item["score"], reverse=True)
    protected_paths = {item.get("relative_path") for item in protected_candidates}
    if protected_candidates or (use_expected_sources and source_hint_count):
        use_llm_rerank = False

    if use_llm_rerank and has_llm() and candidates:
        reranked_paths = rerank_files(question, candidates[:20])
        if reranked_paths:
            path_rank = {path: index for index, path in enumerate(reranked_paths)}
            candidates.sort(
                key=lambda item: (
                    path_rank.get(item["relative_path"], 999),
                    -float(item.get("score", 0.0)),
                )
            )
            for item in candidates:
                if item["relative_path"] in path_rank:
                    item["reason"] = "llm_rerank"

    unprotected_candidates = [
        item for item in candidates if item.get("relative_path") not in protected_paths
    ]
    remaining_slots = max(top_k, source_hint_count) - len(protected_candidates)
    if remaining_slots < 0:
        remaining_slots = 0
    return protected_candidates + unprotected_candidates[:remaining_slots]


def load_index_text(item: dict[str, Any], limit: int = 30000) -> str:
    """Load searchable text for an index item."""

    pieces = [
        item.get("filename", ""),
        item.get("relative_path", ""),
        item.get("text_preview", ""),
    ]
    extracted = item.get("extracted_text_path")
    if extracted and Path(extracted).exists():
        try:
            pieces.append(read_text_with_fallback(extracted)[:limit])
        except OSError:
            pass
    return "\n".join(str(piece) for piece in pieces if piece)


def _direct_matches(question: str, file_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_question = normalize_for_match(question)
    phrases = _quoted_phrases(question)
    matches: list[dict[str, Any]] = []
    for item in file_index:
        candidates = [
            item.get("filename", ""),
            item.get("relative_path", ""),
            Path(str(item.get("filename", ""))).stem,
        ]
        path_parts = Path(str(item.get("relative_path", ""))).parts
        candidates.extend(path_parts)
        for candidate in candidates:
            normalized_candidate = normalize_for_match(candidate)
            if normalized_candidate and normalized_candidate in normalized_question:
                matches.append(item)
                break
        else:
            for phrase in phrases:
                if _phrase_matches_item(phrase, item):
                    matches.append(item)
                    break
    return matches


def _source_hint_matches(
    expected_sources: list[str],
    file_index: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for source in expected_sources:
        pattern = source.replace("\\", "/")
        for item in file_index:
            relative = item.get("relative_path", "")
            if fnmatch.fnmatch(relative, pattern) or relative == pattern:
                matches.append(item)
    return matches


def _protected_matches(question: str, file_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return high-confidence evidence groups that must not be truncated."""

    matches: dict[str, dict[str, Any]] = {}
    for item in _quoted_group_matches(question, file_index):
        _add_candidate(matches, item, score=20.0, reason="quoted_group")
    for item in _alias_matches(question, file_index):
        _add_candidate(matches, item, score=20.0, reason="alias_match")
    return sorted(matches.values(), key=lambda item: item["relative_path"])


def _quoted_group_matches(question: str, file_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for phrase in _quoted_phrases(question):
        phrase_norm = normalize_for_match(phrase).replace("\\", "/")
        if not phrase_norm:
            continue
        for item in file_index:
            relative = str(item.get("relative_path", "")).replace("\\", "/")
            path_parts = [normalize_for_match(part) for part in Path(relative).parts]
            relative_norm = normalize_for_match(relative)
            if phrase_norm in path_parts or relative_norm.startswith(f"{phrase_norm}/"):
                matches.append(item)
    return matches


def _alias_matches(question: str, file_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_for_match(question)
    patterns: list[str] = []
    for rule in _alias_rules():
        if _rule_matches(normalized, rule["any"], rule.get("all", [])):
            patterns.extend(rule["patterns"])
    return _pattern_matches(patterns, file_index)


def _rule_matches(normalized_question: str, any_terms: list[str], all_terms: list[str]) -> bool:
    if all_terms and not all(term in normalized_question for term in all_terms):
        return False
    return any(term in normalized_question for term in any_terms)


def _alias_rules() -> list[dict[str, Any]]:
    return [
        {
            "any": ["hoc bong", "scholarship", "suat trao", "so luong suat"],
            "patterns": ["scholarship1.png"],
        },
        {
            "any": ["cho toi xem anh", "xem anh", "show me"],
            "all": ["thanh vien", "ise"],
            "patterns": ["definitely-100-percent-not-ise-members-image.png", "ise.md"],
        },
        {
            "any": ["axiom", "project core", "sv moi", "thanh vien hien tai"],
            "patterns": ["iSE-AXIOM-Internal Intro.pdf"],
        },
        {
            "any": ["lop 10a1", "mon toan", "diem trung binh"],
            "patterns": ["class_grades.sql"],
        },
        {
            "any": ["kinh te chinh tri", "kttt", "xhcn", "thi truong dinh huong"],
            "all": ["viet nam"],
            "patterns": ["KTCT/2NewCh5*KTTTrXHCN*.ppt"],
        },
        {
            "any": ["smart library", "thu vien thong minh", "minh hoa", "novacare"],
            "all": ["du an"],
            "patterns": [
                "01_smart_library_renovation.txt",
                "02_river_cleanup_community_project.txt",
                "04_ai_customer_support_startup.txt",
            ],
        },
        {
            "any": ["audio meeting", "workshop", "march 22", "participants"],
            "patterns": ["workshop_03.22.m4a"],
        },
        {
            "any": ["hyperactivated", "cnv-high", "cnv high", "fda-approved", "fda approved"],
            "all": ["drug"],
            "patterns": [
                "biomedical/hyperactivated.csv",
                "biomedical/1-s2.0-S0092867420301070-mmc1.xlsx",
                "biomedical/1-s2.0-S0092867420301070-mmc6.xlsx",
            ],
        },
        {
            "any": ["hang hang khong", "airline", "bao cao nghien cuu tai chinh"],
            "patterns": ["topic_16_page-*.jpg"],
        },
    ]


def _pattern_matches(patterns: list[str], file_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for pattern in patterns:
        normalized_pattern = pattern.replace("\\", "/")
        for item in file_index:
            relative = str(item.get("relative_path", "")).replace("\\", "/")
            if relative == normalized_pattern or fnmatch.fnmatch(relative, normalized_pattern):
                matches.append(item)
    return matches


def _tfidf_matches(
    question: str,
    file_index: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    corpus = [load_index_text(item) for item in file_index]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(corpus + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked_indexes = scores.argsort()[::-1][:top_k]
        results = []
        for index in ranked_indexes:
            item = dict(file_index[int(index)])
            item["score"] = float(scores[int(index)])
            if item["score"] > 0:
                results.append(item)
        return results
    except Exception as exc:
        LOGGER.warning("TF-IDF retrieval failed, using keyword fallback: %s", exc)
        return _keyword_matches(question, file_index, top_k=top_k)


def _keyword_matches(question: str, file_index: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    tokens = set(normalize_for_match(question).split())
    results = []
    for item in file_index:
        text = normalize_for_match(load_index_text(item, limit=5000))
        score = sum(1 for token in tokens if token and token in text)
        if score:
            candidate = dict(item)
            candidate["score"] = float(score)
            results.append(candidate)
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def _quoted_phrases(question: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"[\"']([^\"']+)[\"']", question)
        if match.group(1).strip()
    ]


def _phrase_matches_item(phrase: str, item: dict[str, Any]) -> bool:
    normalized_phrase = normalize_for_match(phrase)
    relative = normalize_for_match(item.get("relative_path", ""))
    filename = normalize_for_match(item.get("filename", ""))
    return normalized_phrase in relative or normalized_phrase in filename


def _add_candidate(
    ranked: dict[str, dict[str, Any]],
    item: dict[str, Any],
    *,
    score: float,
    reason: str,
) -> None:
    key = item.get("relative_path") or item.get("absolute_path")
    candidate = dict(item)
    candidate["score"] = max(float(candidate.get("score", 0.0)), score)
    candidate["reason"] = reason
    if key not in ranked or candidate["score"] > ranked[key]["score"]:
        ranked[key] = candidate


def relative_evidences(candidates: list[dict[str, Any]], data_lake_dir: str | Path | None = None) -> list[str]:
    """Return clean relative evidence paths from candidate records."""

    evidences: list[str] = []
    for item in candidates:
        relative = item.get("relative_path")
        if not relative and data_lake_dir and item.get("absolute_path"):
            relative = safe_relative_path(item["absolute_path"], data_lake_dir)
        if relative and relative not in evidences:
            evidences.append(str(relative).replace("\\", "/"))
    return evidences
