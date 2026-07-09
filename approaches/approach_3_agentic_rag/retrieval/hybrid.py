"""Buoc 2: hybrid retrieval.

Merge rule follows the pipeline document: semantic vector similarity and BM25
are normalized per query and combined as 0.6 x vector + 0.4 x bm25 (weights
configurable). Explicit filename/wildcard mentions bypass the scoring with an
absolute score so they always survive ranking. When nothing clears the
relevance threshold the question is routed to "Not enough data".
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Approach3Config
from ..core.models import QuestionProfile
from ..indexing.bm25 import BM25Index, tokenize
from ..indexing.vector_index import VectorIndex
from ..shared_src.utils import normalize_for_match

DIRECT_MATCH_SCORE = 10.0
DISTINCTIVE_KEYWORD_BOOST = 3.0
MAX_CHUNKS_PER_FILE = 4


def retrieve(
    profile: QuestionProfile,
    manifest: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    *,
    config: Approach3Config,
) -> list[dict[str, Any]]:
    """Return ranked candidate files with their best supporting chunks."""

    if not manifest:
        return []
    manifest_by_path = {str(item.get("relative_path", "")): item for item in manifest}

    candidates: dict[str, dict[str, Any]] = {}
    direct = _direct_file_matches(profile, manifest)
    for path, reason in direct.items():
        _merge_candidate(candidates, manifest_by_path[path], DIRECT_MATCH_SCORE, reason, [])

    # Cross-file questions need every relevant file, and a topically-relevant
    # file can rank low semantically (e.g. the "NovaCare" project doc ranked
    # #129/549 chunks - well past the normal 64-chunk window - because the
    # abstract "sustainable impact" query isn't close to it). Widen both the
    # scan window and the returned set so scattered evidence isn't dropped.
    effective_top_k = config.cross_file_top_k if profile.needs_multiple_sources else config.top_k

    query = profile.search_query()
    vector_scores = _normalize(vector_index.scores(query))
    bm25_scores = _normalize(bm25_index.scores(query))
    best_semantic = 0.0
    if len(vector_scores) == len(chunks) and len(chunks) > 0:
        combined = config.vector_weight * vector_scores + config.bm25_weight * bm25_scores
        best_semantic = float(np.max(combined)) if len(combined) else 0.0
        scan_limit = len(chunks) if profile.needs_multiple_sources else max(60, config.top_k * 8)
        order = np.argsort(-combined)[:scan_limit]
        for index in order:
            score = float(combined[index])
            if score <= 0:
                break
            chunk = chunks[index]
            item = manifest_by_path.get(str(chunk.get("relative_path", "")))
            if not item:
                continue
            reason = _dominant_signal(vector_scores[index], bm25_scores[index])
            score += _modality_bonus(profile, item, config.modality_boost)
            scored_chunk = dict(chunk)
            scored_chunk["_score"] = score
            _merge_candidate(candidates, item, score, reason, [scored_chunk])

    # Cross-file entity rescue: a question naming a specific entity ("NovaCare")
    # must surface the one file that contains it, even when that file ranks low
    # on the broad query. Boost files whose content holds a distinctive (rare)
    # question keyword - the classic high-IDF signal that combined scoring dilutes.
    if profile.needs_multiple_sources:
        for path in _distinctive_content_matches(profile, chunks):
            if path in candidates:
                candidates[path]["score"] += DISTINCTIVE_KEYWORD_BOOST
                candidates[path]["reason"] = "rare_keyword"

    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    for candidate in ranked:
        candidate["chunks"] = sorted(
            candidate.get("chunks", []),
            key=lambda chunk: float(chunk.get("_score", 0.0)),
            reverse=True,
        )[:MAX_CHUNKS_PER_FILE]

    pattern_set = _full_pattern_set(profile, ranked)
    if pattern_set:
        return pattern_set

    # Buoc 2 guard: nothing explicit and nothing semantically close ->
    # let the pipeline answer "Not enough data" without spending LLM calls.
    if not direct and best_semantic < config.min_relevance:
        return []
    return ranked[:effective_top_k]


def _distinctive_content_matches(
    profile: QuestionProfile,
    chunks: list[dict[str, Any]],
    *,
    max_files: int = 3,
    min_len: int = 4,
) -> set[str]:
    """Files whose content contains a rare question keyword (a proper noun like
    "NovaCare" that appears in only a handful of files). Stopwords are already
    dropped by the analyzer; here we keep single tokens long enough to be
    specific and rare enough (<= max_files) to be a reliable entity signal."""

    keywords = {
        normalize_for_match(kw)
        for kw in profile.keywords
        if len(normalize_for_match(kw)) >= min_len and " " not in normalize_for_match(kw)
    }
    if not keywords:
        return set()

    file_tokens: dict[str, set[str]] = {}
    for chunk in chunks:
        path = str(chunk.get("relative_path", ""))
        file_tokens.setdefault(path, set()).update(tokenize(str(chunk.get("text", ""))))

    boosted: set[str] = set()
    for keyword in keywords:
        holders = [path for path, tokens in file_tokens.items() if keyword in tokens]
        if 1 <= len(holders) <= max_files:
            boosted.update(holders)
    return boosted


def _direct_file_matches(
    profile: QuestionProfile,
    manifest: list[dict[str, Any]],
) -> dict[str, str]:
    normalized_question = normalize_for_match(profile.question).replace("\\", "/")
    hints = profile.explicit_file_hints + profile.quoted_phrases
    patterns = [normalize_for_match(p).replace("\\", "/") for p in profile.wildcard_patterns]
    matches: dict[str, str] = {}

    for item in manifest:
        relative = str(item.get("relative_path", "")).replace("\\", "/")
        filename = str(item.get("filename", ""))
        stem = Path(filename).stem
        relative_norm = normalize_for_match(relative).replace("\\", "/")
        filename_norm = normalize_for_match(filename)
        stem_norm = normalize_for_match(stem)

        matched = False
        for pattern in patterns:
            if pattern and (
                fnmatch.fnmatch(relative_norm, pattern)
                or fnmatch.fnmatch(relative_norm, f"*/{pattern}")
            ):
                matches[relative] = "explicit_pattern"
                matched = True
                break
        if matched:
            continue

        for hint in hints:
            hint_norm = normalize_for_match(hint).replace("\\", "/")
            if not hint_norm:
                continue
            if hint_norm in {relative_norm, filename_norm, stem_norm}:
                matches[relative] = "explicit_file"
                matched = True
                break
            if len(hint_norm) >= 5 and (hint_norm in relative_norm or hint_norm in filename_norm):
                matches[relative] = "explicit_text"
                matched = True
                break
        if matched:
            continue

        if stem_norm and len(stem_norm) >= 5 and stem_norm in normalized_question:
            matches[relative] = "filename_in_question"
        elif filename_norm and filename_norm in normalized_question:
            matches[relative] = "filename_in_question"

    return matches


_DIRECT_MATCH_REASONS = {"explicit_pattern", "explicit_text", "explicit_file"}


def is_folder_count_question(profile: QuestionProfile) -> bool:
    """True when the question counts/filters across every file in a named folder.

    Not just literal wildcard syntax ("number_image/*") - a quoted bare folder
    name ("in 'number_image'") matches every file under it via the same
    direct-match substring path (reason="explicit_text"). Shared by retrieval
    (bypass the top_k cutoff so no matching file is silently dropped) and by
    the image reader (rewrite the per-image vision question and count in code
    instead of asking one image "how many images...").
    """

    has_folder_hint = bool(
        profile.wildcard_patterns or profile.quoted_phrases or profile.explicit_file_hints
    )
    if not has_folder_hint:
        return False
    normalized = normalize_for_match(profile.question)
    return any(term in normalized for term in ["how many", "bao nhieu", "count", "all", "tat ca"])


def _full_pattern_set(
    profile: QuestionProfile,
    ranked: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Counting/listing questions over a named folder need every matching file.

    Either bypasses the normal top_k cutoff, or files are silently dropped
    before they're ever read, undercounting the answer (e.g. "how many
    images ... contain a blue digit?").
    """

    if not is_folder_count_question(profile):
        return None
    protected = [item for item in ranked if item.get("reason") in _DIRECT_MATCH_REASONS]
    if len(protected) < 2:
        # A single specific file match doesn't need full-set treatment - the
        # normal top_k window already covers it.
        return None
    if profile.modality_hint == "image":
        image_only = [item for item in protected if item.get("modality") == "image"]
        protected = image_only or protected
    return protected or None


def _dominant_signal(vector_score: float, bm25_score: float) -> str:
    if vector_score > bm25_score * 1.3:
        return "vector"
    if bm25_score > vector_score * 1.3:
        return "bm25"
    return "hybrid"


def _modality_bonus(profile: QuestionProfile, item: dict[str, Any], boost: float) -> float:
    modality = str(item.get("modality", ""))
    hint = profile.modality_hint
    if hint == modality:
        return boost
    if hint == "cross_file":
        return boost / 3
    if hint == "table" and str(item.get("extension", "")).lower() == ".sql":
        return boost
    return 0.0


def _merge_candidate(
    candidates: dict[str, dict[str, Any]],
    item: dict[str, Any],
    score: float,
    reason: str,
    chunks: list[dict[str, Any]],
) -> None:
    path = str(item.get("relative_path", ""))
    if not path:
        return
    record = candidates.setdefault(
        path,
        {**item, "score": 0.0, "reason": reason, "reasons": [], "chunks": []},
    )
    record["reasons"].append(reason)
    # Max-pooling across a file's matching chunks, not sum: summing structurally
    # favors long documents (more chunks -> more small matches added together)
    # over a short file whose single chunk is a precise, complete match - e.g.
    # a 61-chunk PDF sharing common words outscored a 1-chunk SQL dump that
    # actually contained the exact answer data.
    if score >= record["score"]:
        record["score"] = float(score)
        record["reason"] = reason
    seen = {chunk.get("chunk_id") for chunk in record["chunks"]}
    for chunk in chunks:
        if chunk.get("chunk_id") not in seen:
            record["chunks"].append(chunk)


def _normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    maximum = float(np.max(values))
    if maximum <= 0:
        return np.zeros(len(values))
    return values / maximum
