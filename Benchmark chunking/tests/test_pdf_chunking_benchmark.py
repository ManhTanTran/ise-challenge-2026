from __future__ import annotations

import numpy as np

from approaches.approach_3_agentic_rag.tools.benchmark_pdf_chunking import (
    _parse_extensions,
    semantic_chunk_text,
    split_sentences,
)


def _topic_embed(sentences: list[str]) -> np.ndarray:
    vectors = []
    for sentence in sentences:
        if "táo" in sentence.lower():
            vectors.append([1.0, 0.0])
        elif "xe" in sentence.lower():
            vectors.append([0.0, 1.0])
        else:
            vectors.append([0.1, 0.9])
    return np.asarray(vectors, dtype=np.float32)


def test_split_sentences_uses_punctuation_and_newlines() -> None:
    text = "Câu đầu. Câu thứ hai!\nMột tiêu đề\nNội dung cuối?"
    assert split_sentences(text) == [
        "Câu đầu.",
        "Câu thứ hai!",
        "Một tiêu đề",
        "Nội dung cuối?",
    ]


def test_semantic_chunking_splits_low_similarity_topics() -> None:
    text = "Táo có màu đỏ. Táo mọc trên cây. Xe chạy trên đường. Xe dùng động cơ."
    chunks, diagnostics = semantic_chunk_text(
        text,
        embed_fn=_topic_embed,
        max_chars=200,
        min_chars=0,
        percentile=50,
    )
    assert chunks == ["Táo có màu đỏ. Táo mọc trên cây.", "Xe chạy trên đường. Xe dùng động cơ."]
    assert diagnostics.semantic_boundaries == 1


def test_semantic_chunking_enforces_max_chars() -> None:
    text = " ".join(["Táo rất ngon."] * 20)
    chunks, diagnostics = semantic_chunk_text(
        text,
        embed_fn=_topic_embed,
        max_chars=55,
        min_chars=0,
        percentile=25,
    )
    assert len(chunks) > 1
    assert max(map(len, chunks)) <= 55
    assert diagnostics.hard_splits > 0


def test_long_sentence_is_split_before_embedding() -> None:
    embedded_inputs: list[str] = []

    def recording_embed(sentences: list[str]) -> np.ndarray:
        embedded_inputs.extend(sentences)
        return _topic_embed(sentences)

    semantic_chunk_text(
        "Táo" * 100,
        embed_fn=recording_embed,
        max_chars=40,
        min_chars=0,
        percentile=25,
    )
    assert embedded_inputs
    assert max(map(len, embedded_inputs)) <= 40


def test_tiny_chunk_merges_toward_more_similar_neighbor() -> None:
    text = "Táo đỏ. Lạ. Xe chạy trên đường. Xe dùng động cơ."
    chunks, diagnostics = semantic_chunk_text(
        text,
        embed_fn=_topic_embed,
        max_chars=200,
        min_chars=15,
        percentile=50,
    )
    assert all(len(chunk) >= 15 for chunk in chunks)
    assert diagnostics.tiny_merges >= 1


def test_parse_extensions_normalizes_dots_and_case() -> None:
    assert _parse_extensions("PDF, .DocX,txt") == {".pdf", ".docx", ".txt"}
