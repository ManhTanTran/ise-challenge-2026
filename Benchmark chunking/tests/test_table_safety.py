import pandas as pd
import pytest

from approaches.approach_3_agentic_rag.config import Approach3Config
from approaches.approach_3_agentic_rag.core.models import QuestionProfile
from approaches.approach_3_agentic_rag.readers import table as table_module
from approaches.approach_3_agentic_rag.readers.table import safe_eval_expression


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame({"age": [20, 30, 40], "city": ["HN", "HCM", "HN"]})


def test_valid_aggregation(frame: pd.DataFrame):
    assert safe_eval_expression("df['age'].mean()", frame) == 30


def test_valid_filter_count(frame: pd.DataFrame):
    assert safe_eval_expression("(df['city'] == 'HN').sum()", frame) == 2


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('dir')",
        "open('secret.txt')",
        "df.to_csv('x.csv')",
        "pd.read_csv('x.csv')",
        "eval('1+1')",
        "getattr(df, 'to_csv')('x.csv')",
    ],
)
def test_dangerous_expressions_rejected(frame: pd.DataFrame, expression: str):
    with pytest.raises(ValueError):
        safe_eval_expression(expression, frame)


def _candidate(relative_path: str) -> dict:
    return {"relative_path": relative_path, "absolute_path": f"/fake/{relative_path}", "modality": "table"}


def _profile(question: str) -> QuestionProfile:
    return QuestionProfile(question_id=1, question=question, requires_computation=True)


def _config() -> Approach3Config:
    config = Approach3Config()
    config.answer_model = "test-answer-model"
    config.analysis_model = "test-analysis-model"
    return config


def test_compute_table_answer_shortlists_before_codegen(monkeypatch):
    # Two tables: only "relevant" is actually needed. A naive implementation
    # would put both full schemas in the codegen prompt; the shortlist step
    # should narrow it down to just the relevant one first.
    tables_by_path = {
        "relevant.csv": {"Sheet1": pd.DataFrame({"gene": ["CDK12", "SMARCA4"], "score": [1, 2]})},
        "irrelevant.csv": {"Sheet1": pd.DataFrame({"unrelated_col": [1, 2, 3]})},
    }
    monkeypatch.setattr(table_module, "_load_tables", lambda candidate: tables_by_path[candidate["relative_path"]])
    monkeypatch.setattr(table_module, "has_llm", lambda: True)

    seen_prompts = []

    def fake_call_llm(prompt, *, model, temperature, system):
        seen_prompts.append(prompt)
        if "\"relevant\"" in prompt:  # the shortlist prompt's marker text
            return '{"relevant": ["relevant_csv"]}'
        return "{\"expression\": \"relevant_csv['gene'].tolist()\"}"

    monkeypatch.setattr(table_module, "call_llm", fake_call_llm)

    result = table_module.compute_table_answer(
        _profile("Which genes are relevant?"),
        [_candidate("relevant.csv"), _candidate("irrelevant.csv")],
        config=_config(),
    )
    assert result is not None
    assert "CDK12" in result["result"]
    # The codegen prompt (second call) must not contain the irrelevant table's
    # column, proving the shortlist actually narrowed the schema.
    codegen_prompt = seen_prompts[-1]
    assert "unrelated_col" not in codegen_prompt
    assert "gene" in codegen_prompt


def test_compute_table_answer_falls_back_to_all_tables_when_shortlist_fails(monkeypatch):
    tables_by_path = {
        "a.csv": {"Sheet1": pd.DataFrame({"x": [1, 2, 3]})},
        "b.csv": {"Sheet1": pd.DataFrame({"y": [10, 20]})},
    }
    monkeypatch.setattr(table_module, "_load_tables", lambda candidate: tables_by_path[candidate["relative_path"]])
    monkeypatch.setattr(table_module, "has_llm", lambda: True)

    def fake_call_llm(prompt, *, model, temperature, system):
        if "\"relevant\"" in prompt:
            raise RuntimeError("shortlist API error")
        return "{\"expression\": \"a_csv['x'].sum() + b_csv['y'].sum()\"}"

    monkeypatch.setattr(table_module, "call_llm", fake_call_llm)

    result = table_module.compute_table_answer(
        _profile("Sum everything"),
        [_candidate("a.csv"), _candidate("b.csv")],
        config=_config(),
    )
    # Shortlist failing must not narrow the namespace to nothing - both
    # tables should still be available to the codegen expression.
    assert result is not None
    assert result["result"] == "36"


def test_compute_table_answer_skips_shortlist_for_single_table(monkeypatch):
    monkeypatch.setattr(
        table_module, "_load_tables", lambda candidate: {"Sheet1": pd.DataFrame({"x": [1, 2, 3]})}
    )
    monkeypatch.setattr(table_module, "has_llm", lambda: True)

    calls = []

    def fake_call_llm(prompt, *, model, temperature, system):
        calls.append(prompt)
        return "{\"expression\": \"df['x'].sum()\"}"

    monkeypatch.setattr(table_module, "call_llm", fake_call_llm)

    result = table_module.compute_table_answer(
        _profile("Sum x"), [_candidate("only.csv")], config=_config()
    )
    assert result is not None
    assert result["result"] == "6"
    assert len(calls) == 1  # no separate shortlist call for a single table
