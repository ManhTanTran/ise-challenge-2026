from pathlib import Path

from approaches.approach_3_agentic_rag.analysis.analyzer import analyze_question
from approaches.approach_3_agentic_rag.config import Approach3Config
from approaches.approach_3_agentic_rag.indexing.bm25 import BM25Index
from approaches.approach_3_agentic_rag.indexing.vector_index import VectorIndex
from approaches.approach_3_agentic_rag.retrieval.hybrid import retrieve


def _offline_config() -> Approach3Config:
    config = Approach3Config()
    config.use_llm_analysis = False
    return config


def _make_lake():
    manifest = [
        {
            "file_id": "f1",
            "filename": "Credit.csv",
            "relative_path": "finance/Credit.csv",
            "extension": ".csv",
            "modality": "table",
        },
        {
            "file_id": "f2",
            "filename": "scholarship1.png",
            "relative_path": "images/scholarship1.png",
            "extension": ".png",
            "modality": "image",
        },
        {
            "file_id": "f3",
            "filename": "notes.txt",
            "relative_path": "docs/notes.txt",
            "extension": ".txt",
            "modality": "document",
        },
    ]
    chunks = [
        {
            "chunk_id": "f1::0",
            "file_id": "f1",
            "relative_path": "finance/Credit.csv",
            "filename": "Credit.csv",
            "extension": ".csv",
            "modality": "table",
            "text": "Filename: Credit.csv Columns: age, balance, income credit card customers",
            "chunk_index": 0,
        },
        {
            "chunk_id": "f2::0",
            "file_id": "f2",
            "relative_path": "images/scholarship1.png",
            "filename": "scholarship1.png",
            "extension": ".png",
            "modality": "image",
            "text": "Filename: scholarship1.png scholarship announcement for students",
            "chunk_index": 0,
        },
        {
            "chunk_id": "f3::0",
            "file_id": "f3",
            "relative_path": "docs/notes.txt",
            "filename": "notes.txt",
            "extension": ".txt",
            "modality": "document",
            "text": "Filename: notes.txt meeting notes about the AXIOM project roadmap",
            "chunk_index": 0,
        },
    ]
    return manifest, chunks


def _indexes(chunks, tmp_path: Path):
    vector_index = VectorIndex.load_or_build(
        chunks, tmp_path / "vector_index", model_name="unused", rebuild=True
    )
    return vector_index, BM25Index(chunks)


def test_explicit_filename_outranks_semantic(tmp_path: Path):
    manifest, chunks = _make_lake()
    vector_index, bm25_index = _indexes(chunks, tmp_path)
    profile = analyze_question(
        {"id": 1, "question": "What is the average balance in Credit.csv?"},
        config=_offline_config(),
    )
    candidates = retrieve(
        profile, manifest, chunks, vector_index, bm25_index, config=_offline_config()
    )
    assert candidates
    assert candidates[0]["relative_path"] == "finance/Credit.csv"
    assert candidates[0]["reason"].startswith("explicit") or candidates[0]["reason"] == "filename_in_question"


def test_semantic_retrieval_finds_topic_file(tmp_path: Path):
    manifest, chunks = _make_lake()
    vector_index, bm25_index = _indexes(chunks, tmp_path)
    profile = analyze_question(
        {"id": 2, "question": "What does the meeting say about the AXIOM roadmap?"},
        config=_offline_config(),
    )
    candidates = retrieve(
        profile, manifest, chunks, vector_index, bm25_index, config=_offline_config()
    )
    assert candidates
    assert candidates[0]["relative_path"] == "docs/notes.txt"


def test_irrelevant_question_returns_empty(tmp_path: Path):
    manifest, chunks = _make_lake()
    vector_index, bm25_index = _indexes(chunks, tmp_path)
    config = _offline_config()
    config.min_relevance = 0.99
    profile = analyze_question(
        {"id": 3, "question": "zzz qqq xxyyzz unrelated gibberish"},
        config=config,
    )
    candidates = retrieve(profile, manifest, chunks, vector_index, bm25_index, config=config)
    assert candidates == []


def test_distinctive_content_match_finds_rare_entity():
    from approaches.approach_3_agentic_rag.core.models import QuestionProfile
    from approaches.approach_3_agentic_rag.retrieval.hybrid import _distinctive_content_matches

    chunks = [
        {"relative_path": "a.txt", "text": "NovaCare is a health startup project with support agents"},
    ] + [
        # "project" is common (appears in many files) -> not distinctive.
        {"relative_path": f"f{i}.txt", "text": "this is another project document"}
        for i in range(6)
    ]
    profile = QuestionProfile(
        question_id=1,
        question="commonality across the NovaCare project?",
        keywords=["novacare", "project"],
        needs_multiple_sources=True,
    )
    boosted = _distinctive_content_matches(profile, chunks)
    # "novacare" is in exactly one file -> boosted; "project" spans 7 files -> not.
    assert boosted == {"a.txt"}


def test_vector_index_persists_and_reloads(tmp_path: Path):
    _, chunks = _make_lake()
    first = VectorIndex.load_or_build(
        chunks, tmp_path / "vi", model_name="unused", rebuild=True
    )
    second = VectorIndex.load_or_build(
        chunks, tmp_path / "vi", model_name="unused", rebuild=False
    )
    assert second.meta["fingerprint"] == first.meta["fingerprint"]
    scores = second.scores("scholarship announcement")
    assert len(scores) == len(chunks)
    assert scores.argmax() == 1
