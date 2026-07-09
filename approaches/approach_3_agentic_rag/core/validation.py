"""Submission validation helpers for approach 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUBMISSION_COLUMNS = ["id", "answer", "evidences"]


def validate_submission(dataframe: pd.DataFrame, data_lake_dir: str | Path) -> None:
    """Validate shape and evidence paths before writing."""

    if list(dataframe.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be exactly {SUBMISSION_COLUMNS}.")
    if dataframe["id"].isna().any():
        raise ValueError("Submission contains missing ids.")
    if dataframe["answer"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Submission contains empty answers.")

    data_root = Path(data_lake_dir)
    for row_number, value in enumerate(dataframe["evidences"], start=1):
        for evidence in _parse_evidences(value, row_number):
            if not (data_root / evidence).exists():
                raise ValueError(f"Row {row_number} evidence file does not exist: {evidence}")


def evidence_json(evidences: list[str]) -> str:
    return json.dumps(list(dict.fromkeys(evidences)), ensure_ascii=False)


def _parse_evidences(value: Any, row_number: int) -> list[str]:
    try:
        evidences = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"Row {row_number} evidences is not valid JSON.") from exc
    if not isinstance(evidences, list):
        raise ValueError(f"Row {row_number} evidences must be a JSON list.")
    return [str(item) for item in evidences]
