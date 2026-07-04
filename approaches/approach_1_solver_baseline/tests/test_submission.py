import json
from pathlib import Path

import pandas as pd

from src.submission import load_questions, validate_submission


def test_validate_submission_evidences_json(tmp_path: Path):
    data_lake = tmp_path / "lake"
    data_lake.mkdir()
    (data_lake / "a.txt").write_text("evidence", encoding="utf-8")
    submission = pd.DataFrame(
        [{"id": 1, "answer": "Yes", "evidences": json.dumps(["a.txt"])}],
        columns=["id", "answer", "evidences"],
    )
    validate_submission(submission, data_lake)


def test_load_questions_without_groundtruth(tmp_path: Path):
    question_path = tmp_path / "questions.csv"
    pd.DataFrame(
        {
            "STT": [1],
            "Question": ["How many rows?"],
            "Answer Type": ["exact_match"],
        }
    ).to_csv(question_path, index=False)
    questions = load_questions(question_path)
    assert list(questions.columns) == ["id", "question", "answer_type", "expected_sources"]
    assert questions.loc[0, "id"] == 1
