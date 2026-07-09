"""Log one full-pipeline run's exact-match score into the results leaderboard.

Run this after every `run_pipeline` execution that has groundtruth (i.e. wrote
an `error_analysis.csv`), so `results/submission_runs.md` always shows whether
the latest code change actually improved the score - not just whether it ran.

Usage (repo root):

    python -X utf8 -m approaches.approach_3_agentic_rag.tools.log_submission_result \
        --error-analysis "approaches/approach_3_agentic_rag/outputs/run_v3/error_analysis.csv" \
        --tag "v3-maxpool+whisper+multitable" \
        --notes "before broadened full-pattern-set + retry fix"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .results_tracker import build_submission_record, log_run, SUBMISSION_COLUMNS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log a submission run's exact-match score.")
    parser.add_argument("--error-analysis", required=True, help="Path to error_analysis.csv.")
    parser.add_argument("--tag", required=True, help="Short label for this run in the leaderboard.")
    parser.add_argument("--notes", default="", help="What changed since the last logged run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    record = build_submission_record(
        tag=args.tag,
        error_analysis_path=args.error_analysis,
        notes=args.notes,
    )
    board = log_run(
        RESULTS_DIR,
        "submission_runs",
        record,
        sort_keys=("exact_pct",),
        columns=SUBMISSION_COLUMNS,
    )
    print(f"Exact match: {record['exact_match']} ({record['exact_pct']*100:.1f}%)")
    print(f"Wrong: {record['wrong_ids']}")
    print(f"Leaderboard: {board}")


if __name__ == "__main__":
    main()
