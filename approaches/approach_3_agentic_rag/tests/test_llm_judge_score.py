import pandas as pd

from approaches.approach_3_agentic_rag.tools import llm_judge_score as judge_module


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "question": "How many X?",
                "predicted_answer": "5",
                "groundtruth": "5",
                "answer_type": "exact_match",
                "is_correct": True,
                "error_type": "",
            },
            {
                "id": 2,
                "question": "Which genes are hyperactivated?",
                "predicted_answer": "CDK12 and SMARCA4 are the hyperactivated genes.",
                "groundtruth": "CDK12 and SMARCA4",
                "answer_type": "llm_judge",
                "is_correct": "",
                "error_type": "semantic_or_unjudged",
            },
            {
                "id": 3,
                "question": "What is the capital city mentioned?",
                "predicted_answer": "Paris",
                "groundtruth": "Tokyo",
                "answer_type": "llm_judge",
                "is_correct": "",
                "error_type": "semantic_or_unjudged",
            },
        ]
    )


def test_score_llm_judge_rows_leaves_exact_match_untouched(monkeypatch):
    monkeypatch.setattr(judge_module, "has_llm", lambda: True)
    monkeypatch.setattr(judge_module, "judge_one", lambda *a, **k: {"judge_verdict": "correct", "judge_reason": "x"})

    result = judge_module.score_llm_judge_rows(_frame(), model="test-model")
    exact_row = result[result["id"] == 1].iloc[0]
    assert exact_row["is_correct"] is True or exact_row["is_correct"] == True  # noqa: E712
    assert exact_row["judge_verdict"] == ""  # never touched


def test_score_llm_judge_rows_fills_verdict_for_llm_judge_rows(monkeypatch):
    monkeypatch.setattr(judge_module, "has_llm", lambda: True)

    def fake_judge(question, predicted, groundtruth, *, model):
        # Row 2's predicted contains the groundtruth genes -> correct.
        # Row 3's predicted contradicts groundtruth -> incorrect.
        if "CDK12" in predicted:
            return {"judge_verdict": "correct", "judge_reason": "matches"}
        return {"judge_verdict": "incorrect", "judge_reason": "wrong city"}

    monkeypatch.setattr(judge_module, "judge_one", fake_judge)
    result = judge_module.score_llm_judge_rows(_frame(), model="test-model")

    row2 = result[result["id"] == 2].iloc[0]
    row3 = result[result["id"] == 3].iloc[0]
    assert row2["is_correct"] is True
    assert row2["error_type"] == ""
    assert row3["is_correct"] is False
    assert row3["error_type"] == "semantic_mismatch"


def test_score_llm_judge_rows_skips_when_no_llm(monkeypatch):
    monkeypatch.setattr(judge_module, "has_llm", lambda: False)
    result = judge_module.score_llm_judge_rows(_frame(), model="test-model")
    llm_rows = result[result["answer_type"] == "llm_judge"]
    assert (llm_rows["judge_verdict"] == "").all()


def test_is_correct_handles_bool_string_and_blank():
    assert judge_module._is_correct(True) is True
    assert judge_module._is_correct("True") is True
    assert judge_module._is_correct(False) is False
    assert judge_module._is_correct("") is False
    assert judge_module._is_correct(float("nan")) is False
