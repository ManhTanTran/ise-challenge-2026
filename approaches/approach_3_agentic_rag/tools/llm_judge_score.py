"""Score llm_judge questions with an LLM semantic judge.

`shared_src/submission.py::write_error_analysis` only computes `is_correct`
for exact_match rows - llm_judge rows get `error_type = "semantic_or_unjudged"`
and `is_correct` is left blank. That means every llm_judge question (protein
sites, Chinese strategy summary, project-comparison, etc.) previously had no
automated pass/fail at all; "did this get better?" could only be answered by
re-reading each answer by eye after every change. This adds the missing
measurement: one LLM call per llm_judge row judges whether the predicted
answer captures the same core meaning as groundtruth (ignoring
wording/formatting/language differences), producing a real is_correct + a
short reason - the same repeatable signal exact_match rows already have.

Usage (repo root), after a run_pipeline call has written error_analysis.csv:

    python -X utf8 -m approaches.approach_3_agentic_rag.tools.llm_judge_score \
        --error-analysis "approaches/approach_3_agentic_rag/outputs/run_v10/error_analysis.csv"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import get_config
from ..shared_src.llm_client import call_llm, has_llm

_JUDGE_PROMPT = """
You are grading one QA answer. Judge whether the PREDICTED answer captures
the same core meaning/facts as the GROUNDTRUTH - ignore differences in
wording, formatting, verbosity, level of detail, or language, as long as the
essential content matches. A predicted answer that includes the correct
facts plus extra correct detail should be judged correct. A predicted answer
missing key facts, contradicting groundtruth, or substituting different
facts should be judged incorrect.

Question: {question}

Groundtruth: {groundtruth}

Predicted: {predicted}

Return ONLY JSON: {{"verdict": "correct" or "incorrect", "reason": "short justification"}}
""".strip()


def judge_one(question: str, predicted: str, groundtruth: str, *, model: str) -> dict[str, str]:
    raw = call_llm(
        _JUDGE_PROMPT.format(question=question, groundtruth=groundtruth, predicted=predicted),
        model=model,
        temperature=0,
        system="Return compact valid JSON only.",
    )
    match = re.search(r"\{[\s\S]*\}", raw or "")
    parsed = json.loads(match.group(0), strict=False) if match else {}
    verdict = str(parsed.get("verdict", "")).strip().lower()
    return {
        "judge_verdict": "correct" if verdict == "correct" else "incorrect",
        "judge_reason": str(parsed.get("reason", "")),
    }


def score_llm_judge_rows(frame: pd.DataFrame, *, model: str) -> pd.DataFrame:
    """Fill in is_correct/error_type for every llm_judge row via an LLM judge.

    exact_match rows are left untouched - write_error_analysis already scored
    those with a strict string comparison, which is the right tool for them.
    """

    frame = frame.copy()
    for column in ("judge_verdict", "judge_reason"):
        if column not in frame.columns:
            frame[column] = ""

    for index, row in frame.iterrows():
        if str(row.get("answer_type", "")).strip().lower() != "llm_judge":
            continue
        if not has_llm():
            continue
        result = judge_one(
            str(row.get("question", "")),
            str(row.get("predicted_answer", "")),
            str(row.get("groundtruth", "")),
            model=model,
        )
        frame.loc[index, "judge_verdict"] = result["judge_verdict"]
        frame.loc[index, "judge_reason"] = result["judge_reason"]
        frame.loc[index, "is_correct"] = result["judge_verdict"] == "correct"
        frame.loc[index, "error_type"] = "" if result["judge_verdict"] == "correct" else "semantic_mismatch"
    return frame


def _is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score llm_judge rows in an error_analysis.csv with an LLM judge.")
    parser.add_argument("--error-analysis", required=True, help="Path to error_analysis.csv.")
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the judged CSV. Defaults to <input>_judged.csv.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = get_config()
    input_path = Path(args.error_analysis)
    frame = pd.read_csv(input_path)
    judged = score_llm_judge_rows(frame, model=config.analysis_model)

    output = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_judged.csv")
    judged.to_csv(output, index=False)

    exact = judged[judged["answer_type"].astype(str).str.lower() == "exact_match"]
    llm_judge = judged[judged["answer_type"].astype(str).str.lower() == "llm_judge"]
    exact_correct = sum(_is_correct(v) for v in exact["is_correct"])
    llm_correct = sum(_is_correct(v) for v in llm_judge["is_correct"])
    total_correct = exact_correct + llm_correct
    total = len(judged)

    print(f"exact_match: {exact_correct}/{len(exact)}")
    print(f"llm_judge:   {llm_correct}/{len(llm_judge)}")
    print(f"overall:     {total_correct}/{total} ({total_correct / total * 100:.1f}%)" if total else "overall: 0/0")
    wrong = judged[~judged["is_correct"].apply(_is_correct)]
    print("Wrong:", ", ".join(f"Q{i}" for i in wrong["id"]) if len(wrong) else "-")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
