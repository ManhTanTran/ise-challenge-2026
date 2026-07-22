# Benchmark Fixed-Character và Semantic Chunking

## 1. Mục tiêu

Benchmark này so sánh hai cách tạo chunk cho các tài liệu có thể trích xuất trực tiếp thành text gần với Markdown:

- PDF
- DOCX
- TXT và Markdown
- HTML
- EPUB và PPTX (đã được hỗ trợ, nhưng sample hiện tại không có file thuộc hai loại này)

CSV/XLSX, ảnh, audio và PPT cũ không nằm trong benchmark. Các loại này cần benchmark riêng cho table structure, OCR, transcription hoặc conversion.

Benchmark chỉ chạy trên:

```text
data/sample_data_lake/Data-Lake
```

Nó không thay đổi chunking của production pipeline.

## 2. Dữ liệu benchmark

| Định dạng | Số file | Số ký tự đã trích xuất |
|---|---:|---:|
| DOCX | 3 | 59.898 |
| HTML | 3 | 171.841 |
| Markdown | 1 | 1.396 |
| PDF | 2 | 525.809 |
| TXT | 5 | 10.162 |
| **Tổng** | **14** | **768.106** |

Có 11 câu hỏi có `Data Sources` trỏ đến ít nhất một tài liệu thuộc nhóm trên. Câu hỏi được lấy từ:

```text
0.Sample_Data.xlsx
generated_hard_questions.xlsx
generated_sample_data.xlsx
```

## 3. Chuẩn hóa tài liệu

Mỗi file được đọc bằng parser tương ứng trong `shared_src/file_readers.py`:

| Định dạng | Phương pháp trích xuất |
|---|---|
| PDF | PyMuPDF; fallback sang pdfplumber |
| DOCX | Paragraph và từng hàng table qua python-docx |
| HTML | BeautifulSoup, loại script/style/noscript |
| TXT/MD | Đọc text với encoding fallback |
| EPUB | Giải EPUB, lấy text từ HTML bằng BeautifulSoup |
| PPTX | Lấy text theo slide bằng python-pptx |

Hai phương pháp chunking nhận cùng một extracted text. Filename, path và modality được lưu dưới dạng metadata của chunk, không nối vào text trước khi embedding.

## 4. Phương pháp chunking

### 4.1 Fixed-character baseline

Đây là thuật toán đang được dùng trong Approach 3:

1. Gộp whitespace bằng `normalize_spaces()`.
2. Cắt tuần tự mỗi 2.200 ký tự.
3. Hai chunk liền nhau overlap 250 ký tự.
4. Không xét sentence, paragraph hoặc semantic boundary.

Tham số:

```text
fixed_max_chars = 2200
fixed_overlap   = 250
```

### 4.2 Semantic chunking

Semantic chunking tìm điểm chuyển chủ đề dựa trên similarity giữa các câu kề nhau, không gọi LLM:

1. Tách extracted text thành câu bằng `.`, `!`, `?`, `;` và newline.
2. Embed từng câu bằng FastEmbed với model:

   ```text
   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   ```

3. Chuẩn hóa vector và tính cosine similarity:

   $$
   s_i = \cos(e_i, e_{i+1})
   $$

4. Với mỗi document, tính ngưỡng adaptive tại percentile 25:

   $$
   t_d = P_{25}(s_1, s_2, \ldots, s_{n-1})
   $$

5. Tạo boundary sau câu $i$ nếu:

   $$
   s_i < t_d
   $$

   Phép so sánh strict `<` tránh cắt toàn bộ khi nhiều similarity bằng đúng ngưỡng.

6. Chunk nhỏ hơn 180 ký tự được gộp với neighbor có similarity cao hơn, nếu chunk sau merge không vượt quá giới hạn.
7. Chunk vượt 1.200 ký tự được pack/cắt thêm theo sentence; câu đơn lẻ quá dài được cắt tại whitespace.
8. Chunk cuối cùng được embed lại để xây retrieval index. Sentence embedding ở bước 2 chỉ dùng để tìm boundary.

Tham số:

```text
semantic_percentile = 25
semantic_min_chars  = 180
semantic_max_chars  = 1200
semantic_overlap    = 0
```

Đây là semantic chunking Phase 1, không phải Late Chunking. Late Chunking cần encode toàn document trong một forward pass rồi pool contextualized token embeddings theo boundary.

## 5. Phương pháp retrieval

Để cô lập ảnh hưởng của chunking:

1. Fixed và semantic dùng cùng embedding model.
2. Query được embed bằng cùng backend.
3. Cosine similarity được tính giữa query và tất cả chunk.
4. Xếp hạng chunk theo similarity giảm dần.
5. Điểm của một file là điểm lớn nhất trong các chunk thuộc file đó (max-pooling).
6. Xếp hạng file theo điểm max-pooling.

Benchmark không dùng BM25, reranker, LLM reasoning hoặc các rule retrieval của production pipeline.

## 6. Metric

Gọi:

- $E_q$: tập file nguồn đúng của query $q$.
- $C_K(q)$: top-K chunk trả về.
- $S(C_K)$: tập file nguồn của top-K chunk.
- $F_K(q)$: top-K file sau max-pooling.

### Chunk Hit@K

Query được tính là hit nếu có ít nhất một chunk trong top-K thuộc một file đúng:

$$
ChunkHit@K(q) = \mathbb{1}[S(C_K(q)) \cap E_q \ne \emptyset]
$$

Metric này phù hợp với câu hỏi single-source, nhưng có thể che mất việc thiếu file ở câu cross-file.

### Chunk Source Recall@K

Đo tỷ lệ file nguồn đúng xuất hiện trong top-K chunk:

$$
ChunkSourceRecall@K(q) = \frac{|S(C_K(q)) \cap E_q|}{|E_q|}
$$

### File Recall@K

Đo tỷ lệ file nguồn đúng trong top-K file sau max-pooling:

$$
FileRecall@K(q) = \frac{|F_K(q) \cap E_q|}{|E_q|}
$$

### File Fully-Covered@K

Query chỉ đạt nếu toàn bộ file nguồn đúng đều nằm trong top-K:

$$
FullyCovered@K(q) = \mathbb{1}[E_q \subseteq F_K(q)]
$$

Giá trị báo cáo là trung bình trên toàn bộ query. Đây là metric quan trọng cho câu hỏi cross-file.

### MRR

Với $rank_q$ là thứ hạng đầu tiên có kết quả thuộc file đúng:

$$
MRR = \frac{1}{|Q|}\sum_{q \in Q}\frac{1}{rank_q}
$$

Benchmark báo cáo cả `chunk_mrr` và `file_mrr`.

### Answer Text MRR

Với ground truth dạng text đủ dài và xuất hiện nguyên văn trong extracted text, benchmark ghi rank đầu tiên của chunk chứa ground truth. Ground truth quá ngắn hoặc chỉ là số bị loại để tránh match giả.

Sample hiện chỉ có một câu đủ điều kiện, nên metric này chỉ dùng để debug, không dùng làm kết luận chính.

### Chi phí vận hành

- `chunks`: tổng số chunk.
- `mean_chars`, `p95_chars`, `max_chars`: phân phối kích thước chunk.
- `chunk_encode_seconds`: thời gian embed lại chunk cuối cùng.
- `chunk_build_seconds`: thời gian dựng cả hai chunk set, gồm parse và sentence embedding để tìm semantic boundary.

## 7. Kết quả tổng hợp

### 7.1 Kích thước và chi phí

| Metric | Fixed-character | Semantic P25 |
|---|---:|---:|
| Documents | 14 | 14 |
| Chunks | 392 | 1.447 |
| Mean chars/chunk | 2.169,85 | 521,53 |
| P95 chars/chunk | 2.200 | 1.195 |
| Max chars/chunk | 2.200 | 1.200 |
| Chunk encode time | 16,566 giây | 59,164 giây |

Semantic tạo số chunk nhiều hơn khoảng **3,69 lần** và thời gian embed chunk cao hơn khoảng **3,57 lần**.

### 7.2 Retrieval

| Metric | Fixed-character | Semantic P25 | Tốt hơn |
|---|---:|---:|---|
| Chunk MRR | 0,7883 | **0,8750** | Semantic |
| File MRR | 0,7970 | **0,8864** | Semantic |
| Chunk Hit@1 | 0,7273 | **0,8182** | Semantic |
| Chunk Hit@3 | 0,9091 | 0,9091 | Hòa |
| Chunk Hit@5 | 0,9091 | 0,9091 | Hòa |
| Chunk Hit@8 | 0,9091 | **1,0000** | Semantic |
| Chunk Source Recall@1 | 0,5758 | **0,7273** | Semantic |
| Chunk Source Recall@3 | **0,7879** | 0,7576 | Fixed |
| Chunk Source Recall@5 | **0,8333** | 0,8030 | Fixed |
| Chunk Source Recall@8 | 0,8333 | **0,9242** | Semantic |
| File Recall@1 | 0,5758 | **0,7273** | Semantic |
| File Recall@3 | 0,8333 | 0,8333 | Hòa |
| File Recall@5 | 0,8333 | **0,9697** | Semantic |
| File Recall@8 | 0,8788 | **0,9697** | Semantic |
| Fully-Covered@1 | 0,4545 | **0,6364** | Semantic |
| Fully-Covered@3 | 0,7273 | 0,7273 | Hòa |
| Fully-Covered@5 | 0,7273 | **0,9091** | Semantic |
| Fully-Covered@8 | 0,8182 | **0,9091** | Semantic |

Semantic tốt hơn ở file-level retrieval, MRR và coverage khi K lớn. Fixed tốt hơn nhẹ ở `Chunk Source Recall@3/5`, cho thấy semantic chunk nhỏ có thể khiến nhiều chunk của cùng một file chiếm top-K trước khi đủ các nguồn của câu cross-file.

## 8. Bảng kết quả chunk theo định dạng

| Định dạng | Files | Source chars | Fixed chunks | Semantic chunks | Tỷ lệ Semantic/Fixed |
|---|---:|---:|---:|---:|---:|
| DOCX | 3 | 59.898 | 33 | 150 | 4,55× |
| HTML | 3 | 171.841 | 89 | 172 | 1,93× |
| Markdown | 1 | 1.396 | 1 | 2 | 2,00× |
| PDF | 2 | 525.809 | 264 | 1.103 | 4,18× |
| TXT | 5 | 10.162 | 5 | 20 | 4,00× |
| **Tổng** | **14** | **768.106** | **392** | **1.447** | **3,69×** |

## 9. Bảng kết quả chunk theo tài liệu

| Tài liệu | Loại | Source chars | Fixed | Semantic | Sentences | P25 threshold | Semantic boundaries | Hard splits | Tiny merges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `01_smart_library_renovation.txt` | TXT | 2.072 | 1 | 5 | 20 | 0,2453 | 5 | 0 | 1 |
| `02_river_cleanup_community_project.txt` | TXT | 1.976 | 1 | 4 | 21 | 0,1840 | 5 | 0 | 2 |
| `03_mountain_farming_climate_shift.txt` | TXT | 2.079 | 1 | 3 | 21 | 0,0891 | 5 | 0 | 3 |
| `04_ai_customer_support_startup.txt` | TXT | 2.110 | 1 | 4 | 19 | 0,1868 | 5 | 1 | 3 |
| `05_city_bus_schedule_redesign.txt` | TXT | 1.925 | 1 | 4 | 18 | 0,1840 | 4 | 0 | 1 |
| `iSE-AXIOM-Internal Intro.pdf` | PDF | 6.749 | 4 | 20 | 286 | 0,2345 | 71 | 0 | 52 |
| `ise.md` | MD | 1.396 | 1 | 2 | 12 | 0,2288 | 3 | 0 | 2 |
| `KTCT/Chương1.docx` | DOCX | 8.232 | 5 | 14 | 109 | 0,3014 | 27 | 0 | 14 |
| `KTCT/GIAO-TRINH-KHONG-CHUYEN.pdf` | PDF | 519.060 | 260 | 1.083 | 10.137 | 0,2178 | 2.534 | 16 | 1.468 |
| `KTCT/TONG-HOP-QUIZ.docx` | DOCX | 8.171 | 5 | 21 | 321 | 0,1602 | 80 | 1 | 61 |
| `KTCT/đề-thi-giữa-kì.docx` | DOCX | 43.495 | 23 | 115 | 1.797 | 0,1685 | 449 | 0 | 335 |
| `关羽 - 维基百科，自由的百科全书.html` | HTML | 67.132 | 35 | 66 | 86 | 0,2299 | 21 | 55 | 11 |
| `刘备 - 维基百科，自由的百科全书.html` | HTML | 58.375 | 30 | 58 | 86 | 0,2115 | 21 | 47 | 11 |
| `诸葛亮 - 维基百科，自由的百科全书.html` | HTML | 46.334 | 24 | 48 | 127 | 0,1998 | 32 | 36 | 21 |

`Hard splits` cao ở HTML cho thấy một số sentence unit sau khi loại tag vẫn rất dài. Đây là tín hiệu nên cải thiện HTML parser bằng cách giữ block boundary (`p`, `li`, heading, table row) thay vì chỉ lấy visible text thành một chuỗi dài.

## 10. Cách chạy

Từ repository root:

```powershell
python -X utf8 -m approaches.approach_3_agentic_rag.tools.benchmark_document_chunking
```

Sweep percentile và tách output/cache theo từng run:

```powershell
python -X utf8 -m approaches.approach_3_agentic_rag.tools.benchmark_document_chunking `
  --semantic-percentile 15 `
  --output-dir "approaches/approach_3_agentic_rag/outputs/document_chunking_p15"

python -X utf8 -m approaches.approach_3_agentic_rag.tools.benchmark_document_chunking `
  --semantic-percentile 35 `
  --output-dir "approaches/approach_3_agentic_rag/outputs/document_chunking_p35"
```

Artifacts của mỗi run:

```text
chunks_fixed.jsonl
chunks_semantic.jsonl
document_diagnostics.csv
query_results.csv
summary.json
text_cache/
```

## 11. Giới hạn của kết quả

1. Chỉ có 14 tài liệu và 11 câu hỏi document-backed; kết quả chưa đủ để khẳng định semantic chunking luôn tốt hơn.
2. Fixed dùng max 2.200 ký tự, semantic dùng max 1.200 ký tự. Vì vậy đây là so sánh hai cấu hình end-to-end, chưa cô lập hoàn toàn tác động của semantic boundary.
3. Corpus bị chi phối bởi một PDF giáo trình dài với hơn 519 nghìn ký tự.
4. Metric dùng file nguồn đúng, chưa có annotation span/câu evidence cho phần lớn câu hỏi.
5. Benchmark chưa dùng BM25, hybrid retrieval, reranker hoặc LLM answer generation.
6. Semantic P25 tạo nhiều chunk hơn đáng kể, làm tăng storage, thời gian embed và nguy cơ nhiều chunk cùng file chiếm top-K.

Để đo riêng tác động của boundary, benchmark tiếp theo nên thêm một baseline fixed-size với cùng `max_chars=1200` như semantic chunking.
