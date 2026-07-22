from approaches.approach_3_agentic_rag.config import Approach3Config
from approaches.approach_3_agentic_rag.core.models import AnswerResult, ContextBlock, QuestionProfile
from approaches.approach_3_agentic_rag.reasoning.reasoner import (
    _parse_json_object,
    answer_question,
    finalize_answer,
)
from approaches.approach_3_agentic_rag.shared_src.utils import NOT_ENOUGH_DATA


def _offline_config() -> Approach3Config:
    config = Approach3Config()
    config.use_llm = False
    return config


def _profile(question: str) -> QuestionProfile:
    return QuestionProfile(question_id=1, question=question, answer_type="exact_match")


def test_no_context_returns_not_enough_data():
    result = answer_question(_profile("Anything?"), [], config=_offline_config())
    assert result.answer == NOT_ENOUGH_DATA
    assert result.evidences == []


def test_extractive_fallback_picks_matching_sentence():
    blocks = [
        ContextBlock(
            relative_path="docs/notes.txt",
            modality="document",
            text="The AXIOM project started in 2024. Unrelated sentence here.",
        )
    ]
    result = answer_question(
        _profile("When did the AXIOM project start?"), blocks, config=_offline_config()
    )
    assert "AXIOM" in result.answer
    assert result.evidences == ["docs/notes.txt"]


def test_finalize_filters_unknown_evidence_paths():
    profile = _profile("Question?")
    raw = AnswerResult("42", ["docs/real.txt", "docs/hallucinated.txt"], "llm_reasoning")
    final = finalize_answer(profile, raw, valid_paths={"docs/real.txt"})
    assert final.evidences == ["docs/real.txt"]


def test_finalize_clears_evidences_for_not_enough_data():
    profile = _profile("Question?")
    raw = AnswerResult(NOT_ENOUGH_DATA, ["docs/real.txt"], "llm_reasoning")
    final = finalize_answer(profile, raw, valid_paths={"docs/real.txt"})
    assert final.answer == NOT_ENOUGH_DATA
    assert final.evidences == []


def test_parse_json_object_tolerates_unescaped_newlines_in_strings():
    # Models sometimes emit multi-line bullet-point "reasoning" text with raw
    # newlines instead of "\n" escapes - strict JSON parsing rejects this even
    # though the object is otherwise well-formed (observed on Q12's cross-file
    # "common thread" question, which uses the CoT reasoning field).
    raw = '```json\n{\n "reasoning": "Line one.\nLine two.\n*   Bullet.",\n "answer": "X",\n "evidences": []\n}\n```'
    parsed = _parse_json_object(raw)
    assert parsed.get("answer") == "X"
    assert "Line two" in parsed.get("reasoning", "")
