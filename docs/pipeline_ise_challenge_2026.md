# Pipeline: ISE Summer Challenge 2026 — Multi-Modal Data Lake QA

## Tổng quan

Pipeline nhận câu hỏi ngôn ngữ tự nhiên (đa ngôn ngữ), tự động định vị file liên quan trong một data lake đa phương thức (CSV, Excel, PDF, ảnh, PPTX, audio, HTML, SQL, TXT...), đọc và phân tích nội dung, rồi trả về `answer` và `evidences` theo định dạng `submission.csv`.

Hướng tiếp cận được chọn là **Agent-based + RAG hybrid**: thay vì embed tất cả nội dung vào vector store (tốn token, kém hiệu quả với bảng số và ảnh), pipeline dùng retrieval hai tầng để thu hẹp candidate files, sau đó để LLM tự quyết định cách đọc từng loại file theo modality.

---

## Kiến trúc tổng thể

```
[Data Lake - Google Drive]
    ├── File manifest (tên, loại, path)
    ├── Vector index (FAISS/ChromaDB — text chunks)
    └── Text cache (pre-extracted text)
              │
              ▼
[Bước 1] Question Analysis
    - Detect ngôn ngữ
    - Classify dạng câu hỏi (exact_match / llm_judge)
    - Extract keywords, entities
    - Route: text / table / image / audio / cross-file
              │
              ▼
[Bước 2] Retrieval (Hybrid)
    - Vector search (semantic similarity)
    - BM25 (keyword + filename match)
    - Merge, rerank, lấy top-K candidates
              │
         ┌────┴────────────────────────────────────┐
         ▼         ▼          ▼         ▼          ▼
     [Table]  [Document]  [Image]   [Audio]   [Text/Web]
     CSV/XLS  PDF/PPTX   PNG/JPG   .m4a/.mp3  HTML/MD/SQL
     pandas   pdfplumber  Vision   Whisper    BeautifulSoup
              │
              ▼
[Bước 4] LLM Reasoning (OpenRouter)
    - Assemble context từ tất cả readers
    - Chain-of-Thought prompt
    - Extract answer + evidence list
              │
              ▼
    submission.csv (id, answer, evidences)
```

---

## Các bước chi tiết

### Bước 0: Indexing (Offline — chạy 1 lần trước khi thi)

**Mục đích**: Chuẩn bị index để retrieval nhanh trong lúc thi.

**Input**: Toàn bộ data lake trên Google Drive.

**Output**:
- `manifest.json`: danh sách tất cả file với metadata (tên, loại, path, size, domain/folder).
- Vector index (FAISS hoặc ChromaDB): text chunks từ các file có thể extract text (TXT, MD, HTML, PDF có text layer, CSV headers...).
- Text cache: pre-extracted text cho từng file — tránh đọc lại nhiều lần.

**Công cụ**: `google-api-python-client` để list Drive, `pdfplumber`, `openpyxl`, `python-pptx`, `BeautifulSoup`, `sentence-transformers` (embedding).

**Lưu ý**:
- Với file PDF scanned (không có text layer): chỉ ghi nhận trong manifest, không pre-extract — sẽ dùng vision LLM khi cần.
- Với ảnh và audio: chỉ lưu metadata, không embed — xử lý on-demand.
- Chunk size đề xuất: 512 tokens, overlap 50 tokens.

---

### Bước 1: Question Analysis

**Mục đích**: Hiểu rõ câu hỏi trước khi tìm file.

**Input**: Câu hỏi ngôn ngữ tự nhiên (string).

**Output**:
- `language`: "vi" / "en" / "zh" / ...
- `answer_type`: "exact_match" / "llm_judge"
- `modality_hint`: "table" / "document" / "image" / "audio" / "cross-file" / "no_data"
- `keywords`: danh sách từ khóa để tìm kiếm
- `requires_computation`: True/False (câu hỏi cần tính toán số học)

**Công cụ/Thư viện**: LLM call (OpenRouter) với prompt phân loại ngắn.

**Prompt strategy**:
```
System: Bạn là bộ phân tích câu hỏi cho hệ thống QA trên data lake.
Phân tích câu hỏi sau và trả về JSON với các trường:
language, answer_type (exact_match/llm_judge), modality_hint, keywords[], requires_computation.

User: {question}
```

**Lưu ý / Edge cases**:
- Câu hỏi "Not enough data" — cần hệ thống biết trả về `[]` evidences và answer phù hợp.
- Câu hỏi yêu cầu cross-file (ví dụ Q12 so sánh 3 dự án): modality_hint = "cross-file", cần retrieve nhiều file.
- Câu hỏi có yêu cầu format đặc biệt (làm tròn 2 chữ số, chữ hoa, v.v.): extract instruction này để truyền vào reasoning step.

---

### Bước 2: Retrieval (Hybrid)

**Mục đích**: Thu hẹp từ hàng trăm/nghìn file xuống top-K candidates liên quan nhất.

**Input**: `keywords`, `modality_hint`, `language` từ Bước 1.

**Output**: Danh sách top-K file paths (đề xuất K=5-10).

**Công cụ**: FAISS hoặc ChromaDB (vector search) + rank_bm25 (BM25).

**Chiến lược**:
1. **Vector search**: embed câu hỏi → tìm top-20 chunks gần nhất → group theo file → lấy top-10 file.
2. **BM25 / filename match**: tìm keyword trong tên file, folder, và text cache → lấy top-10 file.
3. **Modality filter**: nếu `modality_hint = "image"`, ưu tiên file PNG/JPG; nếu `audio`, ưu tiên .m4a/.mp3.
4. **Merge & rerank**: union hai danh sách, score = 0.6 × vector_score + 0.4 × bm25_score, lấy top-K.

**Lưu ý / Edge cases**:
- Câu hỏi về file cụ thể ("trong file Credit.csv"): extract tên file → ưu tiên exact filename match với score tuyệt đối.
- Câu hỏi dạng wildcard ("images in number_image/*"): detect pattern → list tất cả file trong folder đó.
- Không tìm được file nào relevant: trả về `answer = "Not enough data to answer."`, `evidences = []`.

---

### Bước 3: Modality Readers

**Mục đích**: Đọc nội dung từ các file candidates theo đúng cách cho từng loại.

**Input**: Danh sách file paths + câu hỏi gốc.

**Output**: `context_chunks` — nội dung đã trích xuất, sẵn sàng để đưa vào LLM.

#### 3a. Table Reader (CSV / Excel)

- **Thư viện**: `pandas`, `sqlite3` (để chạy SQL-style queries nếu cần).
- **Chiến lược**: Load dataframe → nếu câu hỏi cần tính toán (mean, count, filter), generate và execute pandas code → trả về kết quả dạng text.
- **Edge case**: File lớn (>100MB) → chỉ load relevant columns + sample rows trước, rồi filter.

#### 3b. Document Reader (PDF / PPTX / PPT)

- **Thư viện**: `pdfplumber` (PDF có text), `pypdf` fallback, `python-pptx` (PPTX), `LibreOffice` (convert PPT → PPTX).
- **Chiến lược**: Extract text theo trang → chunk → lấy top chunks liên quan đến câu hỏi (cosine sim hoặc keyword filter).
- **Edge case**: PDF scanned → dùng vision LLM (gửi ảnh từng trang). Phát hiện bằng `pdfminer` — nếu không có text layer.

#### 3c. Image / OCR Reader (PNG / JPG / JPEG)

- **Thư viện**: LLM vision (GPT-4o vision hoặc Gemini Flash qua OpenRouter).
- **Chiến lược**: Gửi ảnh + câu hỏi trực tiếp → LLM trả lời dựa trên ảnh.
- **Khi nào dùng OCR thuần** (Tesseract): câu hỏi chỉ cần đọc text đơn giản từ ảnh, không cần hiểu ngữ nghĩa → tiết kiệm token.
- **Edge case**: Folder ảnh nhiều file (number_image/*) → xử lý batch, mỗi ảnh một lần gọi vision, tổng hợp kết quả.

#### 3d. Audio Reader (.m4a / .mp3 / .wav)

- **Thư viện**: `openai-whisper` (local) hoặc Whisper API.
- **Chiến lược**: Transcribe audio → text → xử lý như document.
- **Lưu ý**: Cache transcript để không phải chạy Whisper nhiều lần cho cùng 1 file. Whisper chạy local tốt hơn để tiết kiệm API quota.

#### 3e. Text / Web Reader (HTML / TXT / MD / SQL)

- **Thư viện**: `BeautifulSoup` (HTML), `sqlite3` (SQL), plain read (TXT/MD).
- **Chiến lược HTML**: Strip tags → extract main content → chunk.
- **Chiến lược SQL**: Parse schema → nếu câu hỏi cần query, generate SQL → execute → trả kết quả.

---

### Bước 4: LLM Reasoning

**Mục đích**: Tổng hợp context từ tất cả readers, sinh câu trả lời và danh sách evidence.

**Input**: `question`, `context_chunks`, `answer_type`, format instructions (nếu có).

**Output**: `answer` (string), `evidences` (list of file paths).

**Model đề xuất**: GPT-4o (exact_match) hoặc Gemini 1.5 Pro (llm_judge, cross-file reasoning).

**Prompt template**:
```
System: Bạn là hệ thống QA chuyên nghiệp. Chỉ sử dụng thông tin từ context được cung cấp.
Trả về JSON: {"answer": "...", "evidences": ["file1.csv", ...]}

Yêu cầu format đặc biệt: {format_instructions}

Context:
{context_chunks}

User: {question}
```

**Chain-of-Thought**: Với câu hỏi phức tạp (exact_match cần tính toán, cross-file), thêm "Hãy suy nghĩ từng bước trước khi đưa ra đáp án" vào prompt.

**Lưu ý / Edge cases**:
- Nếu context trống hoặc không liên quan → trả `"Not enough data to answer."` + `evidences = []`.
- Format số: làm tròn, đơn vị, định dạng phần trăm — truyền rõ vào prompt nếu câu hỏi yêu cầu.
- Câu hỏi Yes/No: prompt yêu cầu trả về "Yes" hoặc "No" chính xác.
- Câu trả lời dạng chữ hoa (như Q6 — học bổng): thêm instruction "trả về chữ hoa".

---

## Phụ thuộc & Môi trường

**Ngôn ngữ**: Python 3.10+

**Thư viện chính**:
```
google-api-python-client  # Mount/list Google Drive
pandas, openpyxl          # Table reader
pdfplumber, pypdf         # PDF reader
python-pptx               # PPTX reader
openai-whisper            # Audio transcription
beautifulsoup4, lxml      # HTML reader
sentence-transformers     # Embedding
faiss-cpu                 # Vector index
rank_bm25                 # BM25 retrieval
openai                    # OpenRouter API (dùng base_url override)
```

**Cài đặt OpenRouter**:
```python
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_OPENROUTER_KEY"
)
```

**Môi trường**: Google Colab (GPU T4 free tier, hoặc A100 nếu cần Whisper large).

---

## Phân công đề xuất (4 người)

| Người | Mảng |
|-------|------|
| 1 | Bước 0: Indexing pipeline + file manifest + vector store |
| 2 | Bước 1–2: Question analysis + Hybrid retrieval |
| 3 | Bước 3: Modality readers (Table, Document, Audio) |
| 4 | Bước 3 (Image/OCR) + Bước 4: LLM Reasoning + output formatting |

---

## Rủi ro & Điểm cần chú ý

- **API quota**: OpenRouter có hạn mức. Tránh gọi LLM cho những bước có thể xử lý bằng code (pandas tính mean, đếm rows...). Cache kết quả aggressively.
- **Whisper thời gian chạy**: Transcribe audio tốn thời gian — chạy offline trước khi thi, cache transcript vào disk.
- **PDF scanned**: Phát hiện sớm (pdfminer text layer check) để không lãng phí thời gian extract text rỗng.
- **Cross-file reasoning**: Câu hỏi so sánh nhiều file (Q12) — retrieve đủ file, tổng hợp context cẩn thận, prompt LLM rõ ràng.
- **Exact match format**: Sai format số (thiếu đơn vị, sai làm tròn) = 0 điểm cho câu đó — parse yêu cầu format từ câu hỏi kỹ.
- **"Not enough data"**: Hệ thống phải biết dừng và trả `[]` evidences thay vì hallucinate — thêm guard trong prompt.
- **Deadline 10/7**: Dành ít nhất 1 ngày (9/7) để chạy full pipeline trên toàn bộ sample questions, debug, tune.

---

## Bước tiếp theo được đề xuất

1. **Ngay hôm nay**: Khám phá data lake — list toàn bộ file, phân loại, hiểu cấu trúc folder.
2. **Ngày 1–2**: Build file manifest + prototype Table Reader + Document Reader.
3. **Ngày 3–4**: Build Indexing (vector + BM25) + Question Analysis + Retrieval.
4. **Ngày 5–6**: Build Image Reader + Audio Reader + LLM Reasoning step.
5. **Ngày 7**: Integration test trên 15 sample questions → debug → submit lên leaderboard.
6. **Ngày 8 (9/7)**: Tune prompt, fix edge cases, optimize quota usage, final run.
7. **Ngày 9 (10/7)**: Thi chính thức.
