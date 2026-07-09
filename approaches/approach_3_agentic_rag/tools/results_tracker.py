"""Accumulate experiment results into a self-updating leaderboard.

Each eval run appends one JSON line to `<board>.jsonl` (full history, nothing
overwritten) and regenerates `<board>.md`, a table sorted best-first so you can
see at a glance which config wins. Used by eval_retrieval; reusable for any
metric that produces a summary dict.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..shared_src.utils import ensure_dir

RETRIEVAL_COLUMNS = [
    ("tag", "tag"),
    ("backend", "backend"),
    ("weights", "weights"),
    ("top_k", "top_k"),
    ("analysis", "analysis"),
    ("fully_retrieved", "fully"),
    ("recall_full", "recall_full"),
    ("recall@8", "recall@8"),
    ("mrr", "mrr"),
    ("unanswerable_ok", "unans_ok"),
    ("misses", "misses"),
    ("timestamp", "when"),
]

SUBMISSION_COLUMNS = [
    ("tag", "tag"),
    ("exact_match", "exact_match"),
    ("exact_pct", "exact_pct"),
    ("wrong_ids", "wrong_ids"),
    ("notes", "notes"),
    ("timestamp", "when"),
]


def log_run(
    results_dir: str | Path,
    board_name: str,
    record: dict[str, Any],
    *,
    sort_keys: tuple[str, ...] = ("fully_retrieved", "recall_full"),
    columns: list[tuple[str, str]] = RETRIEVAL_COLUMNS,
) -> Path:
    """Append one result record and regenerate the ranked markdown board."""

    directory = ensure_dir(results_dir)
    record = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), **record}

    jsonl_path = directory / f"{board_name}.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    records = _read_jsonl(jsonl_path)
    records.sort(key=lambda row: tuple(-_as_float(row.get(key)) for key in sort_keys))
    _write_markdown(directory / f"{board_name}.md", board_name, records, columns=columns)
    return directory / f"{board_name}.md"


def build_retrieval_record(
    *,
    tag: str,
    backend: str,
    analysis: str,
    vector_weight: float,
    bm25_weight: float,
    top_k: int,
    summary: dict[str, Any],
    misses: list[Any],
) -> dict[str, Any]:
    """Flatten an eval_retrieval summary into a leaderboard row."""

    return {
        "tag": tag,
        "backend": backend,
        "weights": f"{vector_weight:g}/{bm25_weight:g}",
        "top_k": top_k,
        "analysis": analysis,
        "fully_retrieved": round(float(summary.get("fully_covered", 0.0)), 3),
        "recall_full": round(float(summary.get("recall_full", 0.0)), 3),
        "recall@8": round(float(summary.get("recall", {}).get("@8", 0.0)), 3),
        "mrr": round(float(summary.get("mrr", 0.0)), 3),
        "unanswerable_ok": f"{summary.get('unanswerable_correct', 0)}/{summary.get('unanswerable', 0)}",
        "misses": ",".join(f"Q{m}" for m in misses) if misses else "-",
    }


def build_submission_record(
    *,
    tag: str,
    error_analysis_path: str | Path,
    notes: str = "",
) -> dict[str, Any]:
    """Compute exact-match score from one run's error_analysis.csv."""

    import pandas as pd

    frame = pd.read_csv(error_analysis_path)
    exact = frame[frame["answer_type"].astype(str).str.lower() == "exact_match"]
    is_correct = exact["is_correct"].apply(lambda v: bool(v) if pd.notna(v) else False)
    correct = int(is_correct.sum())
    total = int(len(exact))
    wrong_ids = [str(row["id"]) for (_, row), ok in zip(exact.iterrows(), is_correct) if not ok]
    return {
        "tag": tag,
        "exact_match": f"{correct}/{total}",
        "exact_pct": round(correct / total, 3) if total else 0.0,
        "wrong_ids": ",".join(f"Q{i}" for i in wrong_ids) if wrong_ids else "-",
        "notes": notes,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_markdown(
    path: Path,
    board_name: str,
    records: list[dict[str, Any]],
    *,
    columns: list[tuple[str, str]],
) -> None:
    header = [label for _, label in columns]
    lines = [
        f"# Leaderboard: {board_name}",
        "",
        f"{len(records)} runs, best first. Regenerated automatically; edit "
        "`.jsonl` (not this file) to change history.",
        "",
        "| rank | " + " | ".join(header) + " |",
        "|" + "---|" * (len(header) + 1),
    ]
    for index, record in enumerate(records, start=1):
        cells = [str(index)] + [str(record.get(key, "")) for key, _ in columns]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
