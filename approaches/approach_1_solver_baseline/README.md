# Cách 1: Solver Baseline

Thư mục này chứa hướng giải end-to-end ban đầu. Cách 1 thiên về solver/rule cụ thể cho từng nhóm câu hỏi, tận dụng nhiều logic deterministic cho table, SQL, document, image và audio.

## Cấu Trúc

```text
approaches/approach_1_solver_baseline/
  src/
  tests/
  notebooks/
  outputs/          # local, không push
  .venv/            # local, không push
  requirements.txt
  README.md
```

Trong đó:

- `src/`: code pipeline cách 1.
- `tests/`: test cho các module cách 1.
- `notebooks/`: notebook explore/chạy pipeline.
- `outputs/`: index, cache, submission, error analysis và các run cũ.
- `.venv/`: môi trường Python local.

## Chạy Cách 1

Chạy từ root repo:

```powershell
python -m approaches.approach_1_solver_baseline.src.submission `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --output "approaches\approach_1_solver_baseline\outputs\approach1_submission.csv"
```

Hoặc chạy từ thư mục cách 1:

```powershell
cd "D:\iSE\iSE challenge\ise-challenge-2026\approaches\approach_1_solver_baseline"
python -m src.submission `
  --questions "..\..\data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "..\..\data\sample_data_lake\Data-Lake" `
  --output "outputs\approach1_submission.csv"
```

## Ghi Chú

Cách 1 là baseline tốt để so sánh với cách 2 vì đã có pipeline xử lý hoàn chỉnh và nhiều rule được tinh chỉnh theo sample.
