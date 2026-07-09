# Cách 2: Hybrid RAG

Cách 2 là pipeline end-to-end riêng, xây theo hướng Hybrid RAG đã mô tả trong tài liệu pipeline đề xuất. Mục tiêu là tách rõ các bước parse dữ liệu, indexing, retrieval và reasoning để dễ debug, dễ mở rộng hơn cách 1.

## Pipeline Thực Hiện

1. Build hoặc load `manifest.json` cho folder Data-Lake.
2. Parse file và tạo `text_cache/`.
3. Tạo `chunks.json` từ text cache và metadata.
4. Phân tích từng câu hỏi thành routing hints.
5. Retrieve evidence bằng direct filename match, BM25 và TF-IDF.
6. Đọc file theo modality: table, document, image, audio, SQL.
7. Ưu tiên answer deterministic bằng pandas/SQL/reader có sẵn.
8. Dùng OpenRouter LLM cho context QA khi có `OPENROUTER_API_KEY`.
9. Validate evidence path và ghi `submission.csv`.

Data-Lake gốc không bị copy hoặc chỉnh sửa. Cách 2 chỉ ghi manifest, chunks, cache, debug file và submission trong `approaches/approach_2_hybrid_rag/outputs/`.

## Cấu Trúc Folder

```text
approaches/approach_2_hybrid_rag/
  core/
    indexing.py
    models.py
    question_analysis.py
    validation.py
  retrieval/
    hybrid.py
  reasoning/
    context_builder.py
    engine.py
    table_reasoner.py
  cli/
    run_pipeline.py
  shared_src/
    file_readers.py
    file_indexer.py
    solvers.py
    ...
  outputs/          # local, không push
  tests/
  run_pipeline.py
```

Trong đó:

- `core/`: model dữ liệu, indexing, phân tích câu hỏi, validation.
- `retrieval/`: hybrid retrieval.
- `reasoning/`: build context, reasoning engine, table reasoner.
- `cli/`: entrypoint chạy pipeline.
- `shared_src/`: các reader/helper được đóng gói lại để cách 2 không phụ thuộc code ngoài folder này.
- `outputs/`: nơi ghi manifest, chunks, cache, submission và debug artifact.
- `run_pipeline.py`: wrapper để chạy `python -m approaches.approach_2_hybrid_rag.run_pipeline`.

## Chạy Nhanh

Chạy từ root repo:

```powershell
python -X utf8 -m approaches.approach_2_hybrid_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --output "approaches\approach_2_hybrid_rag\outputs\submission.csv"
```

## Rebuild Index Từ Đầu

Dùng lệnh này nếu muốn cách 2 tự parse lại Data-Lake, không dùng index có sẵn:

```powershell
python -X utf8 -m approaches.approach_2_hybrid_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --output "approaches\approach_2_hybrid_rag\outputs\reindex_demo\submission.csv" `
  --work-dir "approaches\approach_2_hybrid_rag\outputs\reindex_demo" `
  --rebuild-index `
  --limit 1
```

Sau khi chạy, các file chính sẽ nằm ở:

```text
approaches/approach_2_hybrid_rag/outputs/reindex_demo/
  manifest.json
  chunks.json
  text_cache/
  submission.csv
  predictions_debug.csv
  retrieval_debug.jsonl
  question_profiles.json
  error_analysis.csv
```

## Debug Nhanh Bằng Index Có Sẵn

Dùng index đã parse từ cách 1 để chạy nhanh hơn:

```powershell
python -X utf8 -m approaches.approach_2_hybrid_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --file-index "approaches\approach_1_solver_baseline\outputs\runs\parse_20260630_095706\file_index.json" `
  --output "approaches\approach_2_hybrid_rag\outputs\smoke_submission.csv" `
  --limit 5
```

## OpenRouter Tùy Chọn

Không có API key thì pipeline vẫn chạy được các phần deterministic và extractive fallback. Có API key thì document/image QA tốt hơn:

```powershell
set OPENROUTER_API_KEY=your_key_here
set OPENROUTER_MODEL=openai/gpt-4.1-mini
```

Sau đó chạy lại pipeline.

Với ảnh, nếu OCR bằng Tesseract lỗi, rỗng hoặc confidence thấp, pipeline có thể dùng VLM qua OpenRouter để tạo `caption`, `description`, `visible_objects`, `key_values` và bảng nhìn thấy trong ảnh. Các caption này được đưa vào `text_cache/`, `manifest.json` và `chunks.json` để retrieval tìm được ảnh ngay cả khi OCR không đọc ra chữ.

Có thể tắt hoặc ép dùng VLM image parse bằng:

```powershell
set ISE_IMAGE_PARSE_VLM=off
set ISE_IMAGE_PARSE_VLM=always
```

## Output Và Debug Artifact

Mỗi run ghi:

- `submission.csv`: output ba cột để submit.
- `predictions_debug.csv`: answer, strategy và retrieved files.
- `question_profiles.json`: profile phân tích từng câu hỏi.
- `retrieval_debug.jsonl`: log retrieval/reasoning theo từng câu.
- `error_analysis.csv`: có khi file question có `Groundtruth`.
- `manifest.json`: index cấp file.
- `chunks.json`: index cấp đoạn text.
- `text_cache/`: nội dung parse ra từ Data-Lake dạng `.txt`.
- `text_cache/*.image_parse.json`: structured OCR/VLM parse cho ảnh, gồm text, caption, description, bảng/key-value nếu có.
- `vision_cache/`: cache kết quả VLM khi có `OPENROUTER_API_KEY`.

## Run Đã Verify

Lệnh này đã chạy thành công trên sample Data-Lake:

```powershell
python -X utf8 -m approaches.approach_2_hybrid_rag.run_pipeline `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --file-index "approaches\approach_1_solver_baseline\outputs\runs\parse_20260630_095706\file_index.json" `
  --output "approaches\approach_2_hybrid_rag\outputs\submission_full_sample.csv"
```

Kết quả sample trong `error_analysis.csv` đạt `8 / 8` câu exact-match đúng.
