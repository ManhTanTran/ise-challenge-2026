"""Hybrid retrieval: direct filename, BM25, and TF-IDF vector signals."""

from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..shared_src.utils import normalize_for_match

from ..core.models import QuestionProfile


def retrieve(
    profile: QuestionProfile,
    manifest: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Return ranked file candidates with selected chunks."""

    if not manifest:
        return []

    manifest_by_path = {str(item.get("relative_path", "")): item for item in manifest}
    file_scores: dict[str, dict[str, Any]] = {}

    direct_paths = _direct_file_matches(profile, manifest)
    for path, reason in direct_paths.items():
        item = manifest_by_path.get(path)
        if item:
            _add_file_score(file_scores, item, 5.0, reason, [])

    chunk_scores = _hybrid_chunk_scores(profile, chunks)
    for chunk, score, reason in chunk_scores:
        item = manifest_by_path.get(str(chunk.get("relative_path", "")))
        if not item:
            continue
        score += _modality_bonus(profile, item)
        _add_file_score(file_scores, item, score, reason, [chunk])

    ranked = sorted(file_scores.values(), key=lambda item: item["score"], reverse=True)
    for candidate in ranked:
        candidate["chunks"] = sorted(
            candidate.get("chunks", []),
            key=lambda chunk: float(chunk.get("_score", 0.0)),
            reverse=True,
        )[:4]
    if _needs_full_pattern_set(profile):
        protected = [
            item
            for item in ranked
            if "explicit_pattern" in item.get("reasons", [])
            and (profile.modality_hint != "image" or item.get("modality") == "image")
        ]
        return protected or ranked
    return ranked[:top_k]


def _direct_file_matches(
    profile: QuestionProfile,
    manifest: list[dict[str, Any]],
) -> dict[str, str]:
    normalized_question = normalize_for_match(profile.question).replace("\\", "/")
    hints = profile.explicit_file_hints + profile.quoted_phrases
    matches: dict[str, str] = {}

    for item in manifest:
        relative = str(item.get("relative_path", "")).replace("\\", "/")
        filename = str(item.get("filename", ""))
        stem = Path(filename).stem
        relative_norm = normalize_for_match(relative).replace("\\", "/")
        filename_norm = normalize_for_match(filename)
        stem_norm = normalize_for_match(stem)

        for hint in hints:
            normalized_hint = normalize_for_match(hint).replace("\\", "/")
            if not normalized_hint:
                continue
            wildcard = "*" in normalized_hint
            if wildcard and fnmatch.fnmatch(relative_norm, normalized_hint):
                matches[relative] = "explicit_pattern"
                break
            if normalized_hint in {relative_norm, filename_norm, stem_norm}:
                matches[relative] = "explicit_file"
                break
            if normalized_hint in relative_norm or normalized_hint in filename_norm:
                matches[relative] = "explicit_text"
                break
        else:
            if stem_norm and len(stem_norm) >= 5 and stem_norm in normalized_question:
                matches[relative] = "filename_in_question"
            elif filename_norm and filename_norm in normalized_question:
                matches[relative] = "filename_in_question"

    return matches


def _hybrid_chunk_scores(
    profile: QuestionProfile,
    chunks: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], float, str]]:
    if not chunks:
        return []
    query = " ".join([profile.question, *profile.keywords])
    bm25 = _bm25_scores(query, chunks)
    vector = _tfidf_scores(query, chunks)
    bm25 = _normalize_scores(bm25)
    vector = _normalize_scores(vector)

    results: list[tuple[dict[str, Any], float, str]] = []
    for index, chunk in enumerate(chunks):
        score = 0.45 * bm25[index] + 0.55 * vector[index]
        if score <= 0:
            continue
        candidate_chunk = dict(chunk)
        candidate_chunk["_score"] = float(score)
        reason = "hybrid"
        if bm25[index] > vector[index] * 1.3:
            reason = "bm25"
        elif vector[index] > bm25[index] * 1.3:
            reason = "vector"
        results.append((candidate_chunk, float(score), reason))

    return sorted(results, key=lambda item: item[1], reverse=True)[: max(50, len(results))]


def _bm25_scores(query: str, chunks: list[dict[str, Any]]) -> np.ndarray:
    query_tokens = _tokens(query)
    docs = [_tokens(str(chunk.get("text", ""))) for chunk in chunks]
    if not query_tokens or not docs:
        return np.zeros(len(chunks))

    doc_freq: Counter[str] = Counter()
    for doc in docs:
        doc_freq.update(set(doc))
    avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
    k1 = 1.5
    b = 0.75
    scores = []
    for doc in docs:
        counts = Counter(doc)
        dl = len(doc) or 1
        score = 0.0
        for token in query_tokens:
            if token not in counts:
                continue
            idf = math.log(1 + (len(docs) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            tf = counts[token]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avgdl, 1e-6)))
        scores.append(score)
    return np.asarray(scores, dtype=float)


def _tfidf_scores(query: str, chunks: list[dict[str, Any]]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [str(chunk.get("text", "")) for chunk in chunks]
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(corpus + [query])
        return cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except Exception:
        return np.zeros(len(chunks))


def _tokens(text: str) -> list[str]:
    normalized = normalize_for_match(text)
    raw = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_./*\-]+", normalized)
    return [token for token in raw if len(token) >= 2]


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    maximum = float(np.max(values))
    if maximum <= 0:
        return np.zeros(len(values))
    return values / maximum


def _modality_bonus(profile: QuestionProfile, item: dict[str, Any]) -> float:
    modality = str(item.get("modality", ""))
    hint = profile.modality_hint
    if hint == modality:
        return 0.35
    if hint == "cross_file":
        return 0.1
    if hint == "table" and str(item.get("extension", "")).lower() == ".sql":
        return 0.35
    return 0.0


def _add_file_score(
    file_scores: dict[str, dict[str, Any]],
    item: dict[str, Any],
    score: float,
    reason: str,
    chunks: list[dict[str, Any]],
) -> None:
    path = str(item.get("relative_path", ""))
    if not path:
        return
    record = file_scores.setdefault(
        path,
        {
            **item,
            "score": 0.0,
            "reason": reason,
            "reasons": [],
            "chunks": [],
        },
    )
    record["score"] += float(score)
    record["reasons"].append(reason)
    if score >= record.get("best_score", 0.0):
        record["best_score"] = float(score)
        record["reason"] = reason
    existing_chunk_ids = {chunk.get("chunk_id") for chunk in record["chunks"]}
    for chunk in chunks:
        if chunk.get("chunk_id") not in existing_chunk_ids:
            record["chunks"].append(chunk)


def _needs_full_pattern_set(profile: QuestionProfile) -> bool:
    normalized = normalize_for_match(profile.question)
    return (
        "how many" in normalized
        and "image" in normalized
        and any("*" in hint for hint in profile.explicit_file_hints)
    )
