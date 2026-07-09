# iSE Challenge 2026

Repo này chứa ba hướng giải cho bài toán **iSE Summer Challenge 2026 - Multi-Modal Data Lake Question Answering**.

## Cấu Trúc Repo

```text
ise-challenge-2026/
  approaches/
    approach_1_solver_baseline/
    approach_2_hybrid_rag/
    approach_3_agentic_rag/
  data/
    README.md
    sample_data_lake/        # dữ liệu local, không push lên GitHub
  docs/
    pipeline_ise_challenge_2026.md
  .env.example
  .gitignore
  README.md
```

## Ba Hướng Giải

### Cách 1: Solver Baseline

Nằm ở:

```text
approaches/approach_1_solver_baseline/
```

Đây là baseline end-to-end ban đầu. Hướng này thiên về solver/rule cụ thể cho từng nhóm câu hỏi, có nhiều logic deterministic cho table, SQL, image, audio và document.

Chạy cách 1:

```powershell
cd "D:\iSE\iSE challenge\ise-challenge-2026\approaches\approach_1_solver_baseline"
python -m src.submission `
  --questions "..\..\data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "..\..\data\sample_data_lake\Data-Lake" `
  --output "outputs\submission.csv"
```

### Cách 2: Hybrid RAG

Nằm ở:

```text
approaches/approach_2_hybrid_rag/
```

Đây là hướng pipeline rõ tầng hơn: parse Data-Lake, tạo manifest/chunks, phân tích câu hỏi, hybrid retrieval, rồi reasoning bằng deterministic reader hoặc LLM.

Chạy cách 2 và tự rebuild index:

```powershell
python -X utf8 -m approaches.approach_2_hybrid_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --output "approaches\approach_2_hybrid_rag\outputs\submission.csv" `
  --work-dir "approaches\approach_2_hybrid_rag\outputs\run" `
  --rebuild-index
```

Chạy nhanh với index local đã có từ cách 1:

```powershell
python -X utf8 -m approaches.approach_2_hybrid_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --file-index "approaches\approach_1_solver_baseline\outputs\runs\parse_20260630_095706\file_index.json" `
  --output "approaches\approach_2_hybrid_rag\outputs\submission_full_sample.csv"
```

### Cách 3: Agentic Semantic RAG

Nằm ở:

```text
approaches/approach_3_agentic_rag/
```

Hiện thực đầy đủ pipeline đề xuất trong docs: vector index ngữ nghĩa (sentence-transformers/FAISS, fallback TF-IDF), question analysis bằng LLM có cache, hybrid retrieval `0.6×vector + 0.4×BM25`, table reasoning bằng pandas do LLM sinh code, và LLM reasoning trả JSON answer+evidences với chain-of-thought.

Chạy cách 3:

```powershell
python -X utf8 -m approaches.approach_3_agentic_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --output "approaches\approach_3_agentic_rag\outputs\submission.csv"
```

Xem thêm [Cách 3 README](approaches/approach_3_agentic_rag/README.md).

## Dữ Liệu

Dữ liệu sample nằm local ở:

```text
data/sample_data_lake/
```

Folder dữ liệu này được ignore khi push GitHub để repo nhẹ và tránh commit các file lớn. Xem thêm [data/README.md](data/README.md).

## Cài Đặt

Tạo môi trường Python và cài dependency từ root repo:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Biến Môi Trường

Tạo file `.env` từ `.env.example` nếu cần dùng OpenRouter:

```powershell
copy .env.example .env
```

Sau đó điền:

```text
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4.1-mini
```

## Kiểm Tra Nhanh

Chạy test:

```powershell
python -m pytest
```

Compile cách 2:

```powershell
python -m compileall approaches\approach_2_hybrid_rag
```

Smoke test cách 2:

```powershell
python -X utf8 -m approaches.approach_2_hybrid_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --file-index "approaches\approach_1_solver_baseline\outputs\runs\parse_20260630_095706\file_index.json" `
  --output "approaches\approach_2_hybrid_rag\outputs\smoke.csv" `
  --limit 1
```

## Tài Liệu

- [Pipeline đề xuất](docs/pipeline_ise_challenge_2026.md)
- [Cách 1 README](approaches/approach_1_solver_baseline/README.md)
- [Cách 2 README](approaches/approach_2_hybrid_rag/README.md)
