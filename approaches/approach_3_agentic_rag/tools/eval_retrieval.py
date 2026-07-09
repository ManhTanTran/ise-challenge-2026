"""Grade Buoc 2 retrieval recall against the sample "Data Sources" column.

Retrieval is the make-or-break step: if the correct file never enters the
top-K, no reasoning can recover the answer. The sample question file carries a
"Data Sources" column = the ground-truth evidence file(s) per question. For
each question this tool runs analysis + retrieval and measures whether those
files are retrieved.

Metrics:
- Recall@K = (expected files found in top-K) / (expected files), averaged over
  answerable questions.
- Fully-covered@K = fraction of questions where ALL expected files are in top-K.
- MRR = mean reciprocal rank of the first correct file.
- Unanswerable ([] sources) are graded separately: retrieval SHOULD return
  nothing above the relevance threshold.

Runs fully offline (no API cost). Uses heuristic question analysis by default;
pass --use-llm-analysis to include the Buoc 1 LLM call. Without
sentence-transformers installed this measures the TF-IDF fallback recall.

Usage (repo root):

    python -X utf8 -m approaches.approach_3_agentic_rag.tools.eval_retrieval \
        --questions "data/sample_data_lake/0.Sample_Data.xlsx" \
        --data-lake "data/sample_data_lake/Data-Lake" \
        --file-index "approaches/approach_1_solver_baseline/outputs/runs/parse_20260630_095706/file_index.json"
"""

from __future__ import annotations

import argparse
import fnmatch
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from ..analysis.analyzer import analyze_question
from ..config import Approach3Config, get_config
from ..indexing.build import build_indexes
from ..retrieval.hybrid import retrieve
from ..shared_src.submission import load_questions
from ..shared_src.utils import ensure_dir
from .results_tracker import build_retrieval_record, log_run

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

K_VALUES = (1, 3, 5, 8)
EVAL_TOP_K = 12


def _norm(path: str) -> str:
    return str(path).replace("\\", "/").strip().lower()


def expand_expected(expected: list[str], manifest_paths: list[str]) -> set[str]:
    """Resolve expected sources (incl. wildcards) to concrete manifest paths."""

    resolved: set[str] = set()
    normalized_manifest = {_norm(path): path for path in manifest_paths}
    manifest_basename = {_norm(Path(path).name): path for path in manifest_paths}
    for entry in expected:
        target = _norm(entry)
        if not target:
            continue
        if "*" in target:
            for norm_path, original in normalized_manifest.items():
                if fnmatch.fnmatch(norm_path, target) or fnmatch.fnmatch(norm_path, f"*/{target}"):
                    resolved.add(original)
            continue
        if target in normalized_manifest:
            resolved.add(normalized_manifest[target])
        elif _norm(Path(entry).name) in manifest_basename:
            resolved.add(manifest_basename[_norm(Path(entry).name)])
        else:
            # Expected file not in manifest at all -> keep it so recall reflects the miss.
            resolved.add(entry)
    return resolved


def evaluate(
    questions: pd.DataFrame,
    manifest: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    vector_index: Any,
    bm25_index: Any,
    *,
    config: Approach3Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_paths = [str(item.get("relative_path", "")) for item in manifest]
    # Pure-ranking config: drop the "Not enough data" threshold and widen top-K
    # so recall reflects ranking quality, not the answerability cutoff.
    rank_config = replace(config, min_relevance=0.0, top_k=EVAL_TOP_K)
    threshold_config = config

    rows: list[dict[str, Any]] = []
    for _, row in questions.iterrows():
        expected_raw = list(row.get("expected_sources") or [])
        profile = analyze_question(row.to_dict(), config=config)
        ranked = retrieve(profile, manifest, chunks, vector_index, bm25_index, config=rank_config)
        ranked_paths = [str(item.get("relative_path", "")) for item in ranked]
        ranked_norm = [_norm(path) for path in ranked_paths]

        if not expected_raw:
            # Unanswerable: rerun with the real threshold; empty result is correct.
            thresholded = retrieve(
                profile, manifest, chunks, vector_index, bm25_index, config=threshold_config
            )
            rows.append(
                {
                    "id": row.get("id"),
                    "kind": "unanswerable",
                    "expected": 0,
                    "found_total": 0,
                    "recall_full": None,
                    **{f"recall@{k}": None for k in K_VALUES},
                    "first_rank": None,
                    "returned_empty": len(thresholded) == 0,
                    "top_retrieved": ", ".join(ranked_paths[:3]),
                }
            )
            continue

        expected = expand_expected(expected_raw, manifest_paths)
        expected_norm = {_norm(path) for path in expected}
        recalls = {}
        for k in K_VALUES:
            hit = sum(1 for path in ranked_norm[:k] if path in expected_norm)
            recalls[f"recall@{k}"] = hit / len(expected_norm)
        first_rank = next(
            (i + 1 for i, path in enumerate(ranked_norm) if path in expected_norm), None
        )
        # recall_full counts against the entire returned candidate list, so
        # wildcard/folder questions that legitimately need >K files are not
        # penalized by the fixed K window.
        found_all = sum(1 for path in set(ranked_norm) if path in expected_norm)
        recall_full = found_all / len(expected_norm)
        rows.append(
            {
                "id": row.get("id"),
                "kind": "answerable",
                "expected": len(expected_norm),
                "found_total": found_all,
                "recall_full": recall_full,
                **recalls,
                "first_rank": first_rank,
                "returned_empty": None,
                "top_retrieved": ", ".join(ranked_paths[:5]),
            }
        )

    frame = pd.DataFrame(rows)
    answerable = frame[frame["kind"] == "answerable"]
    unanswerable = frame[frame["kind"] == "unanswerable"]
    summary = {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall_full": (
            round(float(answerable["recall_full"].mean()), 3) if len(answerable) else 0.0
        ),
        "recall": {
            f"@{k}": round(float(answerable[f"recall@{k}"].mean()), 3) if len(answerable) else 0.0
            for k in K_VALUES
        },
        "fully_covered": (
            round(float((answerable["recall_full"] >= 0.999).mean()), 3) if len(answerable) else 0.0
        ),
        "mrr": (
            round(
                float(
                    answerable["first_rank"]
                    .apply(lambda r: 1.0 / r if r else 0.0)
                    .mean()
                ),
                3,
            )
            if len(answerable)
            else 0.0
        ),
        "unanswerable_correct": (
            int(unanswerable["returned_empty"].sum()) if len(unanswerable) else 0
        ),
    }
    return frame, summary


def print_report(frame: pd.DataFrame, summary: dict[str, Any], *, using_llm: bool, backend: str) -> None:
    print("=" * 74)
    print("RETRIEVAL RECALL")
    print(f"  question analysis: {'LLM + heuristic' if using_llm else 'heuristic only'}")
    print(f"  vector backend   : {backend}")
    print("-" * 74)
    r = summary["recall"]
    print(
        f"  Answerable: {summary['answerable']}   Unanswerable: {summary['unanswerable']}"
    )
    print(
        f"  Recall (full candidate list): {summary['recall_full']}   "
        f"Fully-retrieved: {summary['fully_covered']}   MRR: {summary['mrr']}"
    )
    print(
        f"  Ranking recall@1={r['@1']}  @3={r['@3']}  @5={r['@5']}  @8={r['@8']}"
    )
    print(
        f"  Unanswerable returned empty (correct): "
        f"{summary['unanswerable_correct']}/{summary['unanswerable']}"
    )
    print("-" * 74)
    print("Per question (MISS = expected file not retrieved at all):")
    for _, row in frame.iterrows():
        if row["kind"] == "unanswerable":
            mark = "ok " if row["returned_empty"] else "BAD"
            print(f"  Q{row['id']} [unanswerable] {mark} (should return empty)")
            continue
        rf = row["recall_full"]
        mark = "ok  " if rf >= 0.999 else ("part" if rf > 0 else "MISS")
        rank = row["first_rank"] if row["first_rank"] else "-"
        print(
            f"  Q{row['id']} [{mark}] retrieved {row['found_total']}/{row['expected']} files "
            f"(recall_full={rf:.2f}, first_rank={rank})"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade retrieval recall vs Data Sources.")
    parser.add_argument("--questions", required=True, help="Sample question file with Data Sources.")
    parser.add_argument("--data-lake", required=True, help="Data-Lake folder.")
    parser.add_argument("--work-dir", default=None, help="Index/cache dir (default: outputs/eval).")
    parser.add_argument("--file-index", default=None, help="Reuse an existing manifest/file_index.json.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild indexes.")
    parser.add_argument("--use-llm-analysis", action="store_true", help="Include the Buoc 1 LLM call.")
    parser.add_argument("--vector-weight", type=float, default=None, help="Override semantic weight.")
    parser.add_argument("--bm25-weight", type=float, default=None, help="Override BM25 weight.")
    parser.add_argument("--tag", default=None, help="Label for this run in the leaderboard.")
    parser.add_argument("--no-log", action="store_true", help="Do not append to the leaderboard.")
    parser.add_argument("--output", default=None, help="CSV path for the per-question table.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = get_config()
    config.use_llm_analysis = bool(args.use_llm_analysis)
    if args.vector_weight is not None:
        config.vector_weight = args.vector_weight
    if args.bm25_weight is not None:
        config.bm25_weight = args.bm25_weight
    work_dir = ensure_dir(
        args.work_dir
        or (Path(args.output).parent if args.output else "approaches/approach_3_agentic_rag/outputs/eval")
    )

    questions = load_questions(args.questions)
    manifest, chunks, vector_index, bm25_index = build_indexes(
        args.data_lake,
        work_dir,
        config=config,
        file_index_path=args.file_index,
        rebuild=args.rebuild_index,
    )
    frame, summary = evaluate(
        questions, manifest, chunks, vector_index, bm25_index, config=config
    )
    backend = str(vector_index.meta.get("kind", "unknown"))
    analysis = "llm+heuristic" if config.use_llm_analysis else "heuristic"
    print_report(frame, summary, using_llm=config.use_llm_analysis, backend=backend)

    output = Path(args.output) if args.output else work_dir / "retrieval_recall.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print("-" * 74)
    print(f"Bảng chi tiết: {output}")

    if not args.no_log:
        answerable = frame[frame["kind"] == "answerable"]
        misses = [
            row["id"] for _, row in answerable.iterrows() if float(row["recall_full"]) < 0.999
        ]
        record = build_retrieval_record(
            tag=args.tag or f"{backend}-{config.vector_weight:g}/{config.bm25_weight:g}",
            backend=backend,
            analysis=analysis,
            vector_weight=config.vector_weight,
            bm25_weight=config.bm25_weight,
            top_k=config.top_k,
            summary=summary,
            misses=misses,
        )
        board = log_run(RESULTS_DIR, "retrieval_recall", record)
        print(f"Leaderboard: {board}")


if __name__ == "__main__":
    main()
