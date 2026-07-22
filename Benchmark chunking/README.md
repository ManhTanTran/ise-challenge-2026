# Cách 3: Agentic Semantic RAG

Cách 3 hiện thực đầy đủ pipeline trong [docs/pipeline_ise_challenge_2026.md](../../docs/pipeline_ise_challenge_2026.md) theo hướng **Agent-based + RAG hybrid**: retrieval hai tầng (semantic vector + BM25) để thu hẹp candidate files, sau đó để LLM quyết định cách đọc từng loại file theo modality và tổng hợp câu trả lời.

Khác biệt chính so với cách 2:

- **Vector index ngữ nghĩa thật** (Bước 0): embedding đa ngôn ngữ bằng `sentence-transformers` + FAISS khi có cài; tự fallback về char TF-IDF (sklearn) để pipeline vẫn chạy offline với requirements gốc.
- **Question analysis bằng LLM** (Bước 1): một call rẻ trả JSON (language, modality_hint, keywords, format_instructions...), merge với heuristic regex; kết quả cache trên disk.
- **Công thức merge của tài liệu** (Bước 2): `score = 0.6 × vector + 0.4 × BM25`, filename match được score tuyệt đối, modality hint chỉ boost chứ không lọc cứng, có ngưỡng `min_relevance` để trả sớm "Not enough data".
- **Code-first table reasoning** (Bước 3): câu hỏi cần tính toán → LLM sinh MỘT biểu thức pandas từ schema, biểu thức được screen an toàn rồi thực thi local — số liệu chính xác thay vì để LLM tự tính.
- **Reasoning JSON + CoT có điều kiện** (Bước 4): câu hỏi tính toán/cross-file có slot `reasoning` trong JSON; evidence bị validate 2 lớp (allowed sources trong prompt + manifest paths khi finalize).

## Pipeline

```text
[Buoc 0 - offline]  manifest.json + text_cache/ + chunks.json
                    + vector_index/ (embeddings + meta) + BM25 (in-memory)
[Buoc 1]  analyzer.py    : regex (file/wildcard/format) + LLM JSON + cache
[Buoc 2]  hybrid.py      : 0.6*vector + 0.4*bm25, filename tuyệt đối, threshold
[Buoc 3]  dispatcher.py  : table (pandas compute) / image (vision, cache)
                           / audio (transcript cache) / document / text
[Buoc 4]  reasoner.py    : JSON {answer, evidences}, CoT, guards, formatter
          -> submission.csv (id, answer, evidences)
```

## Cấu Trúc Folder

```text
approaches/approach_3_agentic_rag/
  config.py            # mọi knob: weights, top_k, threshold, model names
  core/                # models, manifest/chunks, submission validation
  indexing/            # embedder, vector_index, bm25, build
  analysis/            # Buoc 1
  retrieval/           # Buoc 2
  readers/             # Buoc 3: dispatcher, table, image
  reasoning/           # Buoc 4
  cli/run_pipeline.py  # entrypoint
  shared_src/          # reader/helper vendored (self-contained như cách 2)
  outputs/             # index, cache, submission, debug (local)
  tests/
```

## Chạy Nhanh

Từ root repo:

```powershell
python -X utf8 -m approaches.approach_3_agentic_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --output "approaches\approach_3_agentic_rag\outputs\submission.csv"
```

Tái sử dụng index đã parse từ cách 1/2 để không phải extract lại:

```powershell
python -X utf8 -m approaches.approach_3_agentic_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --file-index "approaches\approach_1_solver_baseline\outputs\runs\parse_20260630_095706\file_index.json" `
  --output "approaches\approach_3_agentic_rag\outputs\submission.csv"
```

Flags hữu ích:

- `--limit N` — chỉ giải N câu đầu (debug).
- `--rebuild-index` — parse lại Data-Lake và embed lại từ đầu.
- `--no-llm` — chế độ offline hoàn toàn (heuristic + extractive fallback).
- `--no-llm-analysis` / `--no-vision` / `--no-table-compute` — tắt từng phần để tiết kiệm quota.
- `--embedding-model <name>` — đổi model sentence-transformers.
- `--min-relevance <f>` — chỉnh ngưỡng "Not enough data" (mặc định 0.05).

## OpenRouter

Cần `OPENROUTER_API_KEY` trong `.env` (xem `.env.example` ở root). Model qua env:

```text
OPENROUTER_MODEL=openai/gpt-4.1-mini     # reasoning + table codegen
ISE_ANALYSIS_MODEL=...                   # optional: model rẻ hơn cho Buoc 1
ISE_VISION_MODEL=...                     # optional: model vision riêng
ISE_EMBEDDING_PROVIDER=auto              # auto/local by default; set openrouter for API embeddings
ISE_EMBEDDING_MODEL=...                  # local sentence-transformers or OpenRouter embedding model
ISE_TRANSCRIPTION_PROVIDER=local         # set openrouter to use OpenRouter audio transcription
ISE_TRANSCRIPTION_MODEL=openai/whisper-1 # OpenRouter STT model when provider=openrouter
ISE_LOCAL_WHISPER_MODEL=base             # local openai-whisper model; use large-v3/turbo on Colab GPU
```

Không có key: pipeline vẫn chạy (retrieval + extractive fallback), chất lượng thấp hơn.

Cache để tiết kiệm quota khi chạy lại: `analysis_cache.json` (Bước 1), `vision_cache/` (Bước 3), `text_cache/` + transcript audio (Bước 0), `vector_index/` (embeddings).

## Output Và Debug Artifact

Mỗi run ghi vào work-dir (mặc định là thư mục chứa `--output`):

- `submission.csv` — output 3 cột để submit.
- `submission_partial.csv` — ghi tăng dần từng câu, crash giữa chừng không mất kết quả.
- `predictions_debug.csv` — answer, strategy, readers, retrieved files từng câu.
- `question_profiles.json` — profile Bước 1 của từng câu.
- `retrieval_debug.jsonl` — log retrieval/reasoning chi tiết theo câu.
- `error_analysis.csv` — khi file câu hỏi có cột `Groundtruth`.
- `manifest.json`, `chunks.json`, `text_cache/`, `vector_index/` — artifact Bước 0.

## Test

```powershell
python -m pytest approaches\approach_3_agentic_rag\tests
```

Tests chạy offline hoàn toàn (không cần API key): analyzer heuristics, hybrid retrieval trên lake giả, safety screen của table compute, extractive fallback + evidence validation.
