# Leaderboard: retrieval_recall

5 runs, best first. Regenerated automatically; edit `.jsonl` (not this file) to change history.

| rank | tag | backend | weights | top_k | analysis | fully | recall_full | recall@8 | mrr | unans_ok | misses | when |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v8-crossfile-boost+modality-fix | fastembed | 0.5/0.5 | 8 | heuristic | 0.857 | 0.898 | 0.78 | 1.0 | 0/1 | Q10,Q15 | 2026-07-08 10:22 |
| 2 | fastembed-0.5-0.5+maxpool+whisper | fastembed | 0.5/0.5 | 8 | heuristic | 0.786 | 0.855 | 0.732 | 0.933 | 0/1 | Q10,Q12,Q15 | 2026-07-07 21:47 |
| 3 | fastembed-0.5/0.5 | fastembed | 0.5/0.5 | 8 | heuristic | 0.714 | 0.82 | 0.571 | 0.487 | 0/1 | Q10,Q11,Q12,Q15 | 2026-07-06 18:08 |
| 4 | fastembed-0.6/0.4 | fastembed | 0.6/0.4 | 8 | heuristic | 0.643 | 0.745 | 0.571 | 0.521 | 0/1 | Q10,Q11,Q12,Q13,Q15 | 2026-07-06 18:09 |
| 5 | tfidf-baseline | tfidf-char | 0.6/0.4 | 8 | heuristic | 0.643 | 0.724 | 0.661 | 1.0 | 0/1 | Q8,Q10,Q11,Q12,Q15 | 2026-07-06 18:09 |
