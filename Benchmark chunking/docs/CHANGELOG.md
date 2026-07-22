# Changelog — Approach 3 (Agentic Semantic RAG)

Chỉ ghi ở đây khi **đổi phương pháp/tham số cần so sánh tốt-xấu** — ví dụ: đổi `top_k`, trọng số `vector/bm25`, ngưỡng `min_relevance`, đổi model, đổi công thức tính điểm retrieval, đổi chiến lược prompt/reasoning. **Không ghi** bug fix thường, cài đặt môi trường, hay refactor code không đổi hành vi — những cái đó không cần đo trước/sau.

Mỗi mục: đổi gì → vì sao → đo ở đâu (link tới `results/`).

---

## 2026-07-09 — 5 bug phát hiện qua bộ câu hỏi tự sinh "generated_hard_questions" (cố ý nhắm điểm yếu)

Bộ 10 câu hỏi khó tự tạo (header Excel lệch dòng, lọc đa điều kiện, đọc chart trong ảnh, so sánh chéo project/file mới) lộ ra 5 bug thật mà bộ 15 câu cũ chưa từng chạm tới. Cả 5 đều là fix áp dụng chung cho pipeline, không hard-code riêng câu nào.

1. **Header Excel lệch dòng** (`shared_src/file_readers.py::_fix_misplaced_header`, `_detect_header_row`): file khoa học có vài dòng metadata/notes trước header thật → pandas đọc `header=0` sai, biến cột thật thành `Unnamed: N`. Phát hiện: cột `Age_ky` không tồn tại theo tên, `compute_table_answer` báo lỗi KeyError, reasoner tự đọc schema thô rồi bịa số (64819.92 thay vì 3098.12). Fix: dò dòng có nhiều giá trị chuỗi ngắn **không trùng lặp** nhất trong 15 dòng đầu (phân biệt với dòng đơn vị đo kiểu "(ppm)" lặp lại), re-read với header đúng.
2. **`modality_hint` chặn vision cho ảnh chart** (`readers/dispatcher.py::_image_block`): câu hỏi đọc số liệu tài chính từ ảnh (biểu đồ CAPEX) bị phân loại `modality_hint="table"` (do từ "kế hoạch", "CAPEX") nên vision không bao giờ chạy dù ảnh là bằng chứng duy nhất — chỉ còn OCR thô, model tự đoán bừa. Fix: bỏ gate theo `modality_hint`, chỉ dựa vào `vision_budget` (giống cách `count_matching_images`/`compute_scholarship_answer` đã làm).
3. **Context budget không công bằng cho câu multi-file** (`readers/dispatcher.py::build_context_blocks`): câu `needs_multiple_sources` lấy 14 candidate, nhưng 7 tài liệu không liên quan (mỗi cái ~8.8K ký tự) chiếm hết ngân sách 65K ký tự trước khi tới file thật sự liên quan (xếp hạng #11, không bao giờ được model nhìn thấy). Fix: chia đều ký tự/file (`per_file_char_budget = max_context_chars // số_candidate`) khi `needs_multiple_sources=True`, thay vì để file xếp hạng cao chiếm hết trước.
4. **Roster compute bịa project giả** (`readers/table.py::compute_roster_answer`, `_ROSTER_EXTRACT_PROMPT`): gửi mọi document candidate (kể cả file hoàn toàn không liên quan) vào LLM để trích "project roster" — model bịa ra 63 project giả từ slide chính trị-kinh tế không hề có project nào, làm sai kết quả chọn min/max. Fix: thêm điều kiện "nếu tài liệu không thực sự có roster project, trả về rỗng" vào prompt (cùng pattern đã dùng sửa Q6 — loại ảnh không phải bảng suất thật).
5. **Roster compute chỉ hỗ trợ "nhiều nhất"** (`is_roster_max_question`, `_roster_superlative_direction`): câu hỏi "ít thành viên nhất" (fewest) không được nhận diện (trigger chỉ khớp "nhiều...nhất"), và ngay cả khi khớp, code cứng `max()` không hỗ trợ chiều ngược lại. Fix: tách hàm phát hiện chiều (max/min), code chọn `min()`/`max()` tương ứng.
6. **Regression phụ khi thêm rule Q9** (`reasoning/reasoner.py`): thêm rule "luôn trả số dạng chữ số, không viết chữ" (sửa Q9: "sáu" → "6") vô tình lấn át hành vi chọn đáp án trắc nghiệm (Q13: trả "7.55" thay vì "C"). Fix: thêm rule ưu tiên rõ ràng — câu hỏi có lựa chọn A/B/C/D thì luôn trả chữ cái, rule này đứng trước rule "digit form".

**Đo**: `results/submission_runs.md` tag `v13-header-fix+vision-gate+context-budget+roster-direction` — bộ câu hỏi cũ (`0.Sample_Data`) exact_match giữ nguyên 8/8, **overall tăng 13/15 → 14/15 (93.3%)** (Q1, Q2, Q13 đều đúng trở lại/lần đầu đúng). Bộ câu hỏi hard tự sinh: overall 6/10 → **8/10 (80%)**. Chi tiết từng bug + lịch sử qua các lần chạy: `docs/known_issues/generated_hard_questions/error_analysis.md`.

## 2026-07-09 — Fix whitespace filter trong `_CODEGEN_PROMPT` (phát hiện qua bộ câu hỏi thứ 2)

- **Đổi**: `readers/table.py::_CODEGEN_PROMPT` — thêm rule: filter `==` trên cột string/categorical phải dùng `.str.strip() == 'value'` thay vì `== 'value'` trần.
- **Vì sao**: chạy pipeline trên bộ câu hỏi mới (`generated_sample_data.xlsx`, cùng Data-Lake, tái sử dụng index) để đo tổng quát hóa — phát hiện Q14 ("có bao nhiêu bản ghi income >50K trong census.csv") sai (dự đoán 0, GT 7841). Nguyên nhân: cột `income` thực tế chứa `' >50K'` (khoảng trắng đầu), nhưng expression LLM viết `== '>50K'` không strip → không khớp dòng nào. Lỗi tổng quát, không riêng file này — bất kỳ cột categorical nào có whitespace thừa (rất phổ biến ở dữ liệu CSV thực tế) đều gặp lại.
- **Đo**: `results/submission_runs.md` — bộ câu hỏi mới (`generated_sample_data`, 15 câu): exact_match 12/13 → **13/13**, tổng thể (exact+llm_judge qua `tools/llm_judge_score.py`) **15/15 (100%)**. Bộ câu hỏi cũ (`0.Sample_Data.xlsx`): không regression, vẫn 8/8 exact_match, 13/15 tổng thể (Q2/Q3 như cũ, xem `docs/known_issues/`).

## 2026-07-08 — Q2/Q5/Q8: shortlist sheet cho table compute, icon-badge digit color, roster compute

- **Q2 — `readers/table.py::compute_table_answer`**: thêm bước `_shortlist_relevant_tables()` chạy trước khi build schema đầy đủ. Với câu cần join nhiều file/sheet, tổng schema (cột + sample rows) có thể vượt xa giới hạn `truncate_text(..., 16000)` — đo trực tiếp Q2: 8 file biomedical = 30 sheet = **170,290 ký tự**, khiến sheet cần thiết (`mmc1` chứa `CNV_class`) bị cắt mất, model tự trả `expression=""` vì "không đủ dữ liệu". Fix: build "table of contents" rẻ (chỉ tên cột; sheet ≤20 dòng như README thì kèm luôn vài dòng mẫu vì cột "value"/"Description" một mình không mang nghĩa) → 1 LLM call rẻ chọn sheet liên quan → chỉ build schema đầy đủ cho phần được chọn. Lỗi/rỗng → fallback dùng toàn bộ như cũ.
  - **Tác dụng phụ phát hiện khi đo lại**: fix này vô tình làm **lộ ra** một sheet (`mmc6` "C-SE-scetyl-site...") mà trước đây bị `truncate_text` cắt mất khỏi Q1 — Q1 trước đó đúng (16) *nhờ may mắn bị cắt*, giờ model thấy cả 2 sheet và tự chọn sheet "trông kỹ hơn" (áp thêm ngưỡng FDR<0.05 không có trong câu hỏi) → sai (21). Sửa bằng cách thêm rule vào `_CODEGEN_PROMPT`: không tự bịa thêm ngưỡng lọc (p-value/FDR/fold-change) không có trong câu hỏi; nếu sheet có mô tả (README) khớp đúng nghĩa câu hỏi thì dùng nguyên trạng.
- **Q5 — `readers/image.py::_STRUCTURED_VISION_PROMPT`**: câu "How many images contain a blue digit?" — ảnh `images.png` là số 7 trắng lồng trong khung tròn đặc xanh dương; model xét đúng nghĩa đen "glyph trắng, không xanh" → sai (GT tính ảnh này là "blue"). Thêm rule: digit dạng icon/badge (hình khối đặc màu, số cắt lỗ màu tương phản) thì màu của icon/badge = màu của digit. Rule đầu tiên (không có ví dụ cụ thể) không đủ mạnh — vẫn sai khi câu hỏi rewrite dùng từ "contain" thay vì "show". Thêm 1 ví dụ cụ thể vào prompt mới đủ mạnh để model áp dụng nhất quán bất kể cách diễn đạt.
- **Q8 — `readers/table.py::compute_roster_answer`** (mới) + `dispatcher.py`: câu "project nào có nhiều thành viên hiện tại nhất, không tính SV mới?" — LLM tự đọc PDF và tự trừ SV mới dễ sai (dự đoán cũ = tổng cả SV mới của sai project). Fix code-first: 1 LLM call trích JSON toàn bộ project (tên, `is_core`, danh sách tên thành viên riêng biệt với `new_member_count`), code tính `len(members)` (không tin số LLM tự đếm) và chọn max trong nhóm `is_core=True`. **Xác nhận với người dùng**: dù câu hỏi yêu cầu "trả về số thứ tự project", cả GT (7) lẫn dự đoán sai cũ (6) đều khớp theo *số lượng thành viên* chứ không phải thứ tự — nên đáp án trả về là số lượng (7), không phải project ordinal (5).
  - Refactor đi kèm (không đổi hành vi): gộp `dispatcher._generic_text` và `analyzer._has_word`/`image._has_word` trùng lặp thành `shared_src/file_readers.py::extract_candidate_text` và `shared_src/utils.py::has_word` dùng chung.
- **Đo**: `results/submission_runs.md` tag `v10-Q2Q5Q8-shortlist+icon-badge+roster-compute` — exact_match **6/8 → 8/8 (100%)**. Q2 (llm_judge) từ "Not enough data" → chứa đúng CDK12/SMARCA4 nhưng định dạng còn dài dòng (liệt kê từng site thay vì gọn tên gene) — cần polish thêm, chưa coi là xong hẳn.

## 2026-07-08 — Code-first scholarship reader (Q6) + fix parse JSON strict

- **Đổi**: `readers/image.py` (`compute_scholarship_answer`, `is_scholarship_slot_question`, `_call_vision_rows`) + `readers/dispatcher.py` (chèn block "Computed by extracting..." trước vòng lặp per-candidate, suppress vision noise cho ảnh đã cover) + `reasoning/reasoner.py` (tổng quát hoá rule tin tưởng "Computed ..." + cache-key version bump).
  - Với câu "học bổng nào có số suất nhiều nhất?": thay vì để LLM tự đọc-chọn từ 1 câu trả lời tự do (dễ chọn nhầm dòng trong bảng đông), mỗi ảnh candidate được yêu cầu trích **toàn bộ** dòng thành JSON có cấu trúc (`scholarship_name`, `off_budget`, `slot_components`), code cộng/so sánh và chọn max — cùng nguyên tắc code-first đã dùng cho bảng/đếm ảnh.
  - Prompt yêu cầu model **tự loại ảnh không phải bảng suất thật** trước khi trích (bỏ qua số tiền/ngày/%) — vòng đầu thiếu điều kiện này khiến model gộp nhầm số tiền học phí từ ảnh học bổng khác (không liên quan ĐHQGHN) vào phép so sánh max, ra sai `HỌC BỔNG 12 THÁNG`.
  - **Giới hạn đã biết**: chỉ kiểm chứng được trên đúng 1 câu hỏi mẫu (Q6) — cách lọc "chỉ trích khi có khái niệm đếm-suất" là suy luận tổng quát từ quan sát lỗi, không hard-code tên/số, nhưng chưa có câu hỏi thứ hai cùng dạng để xác nhận không overfit vào đặc thù dữ liệu mẫu này.
- **Phát hiện thêm khi đo lại (không phải do thay đổi trên)**: Q12 lỗi `llm_error: unparseable response` cả 2 lần retry — nguyên nhân là model trả JSON hợp lệ về nội dung nhưng field `"reasoning"` (CoT nhiều dòng) chứa newline thô chưa escape, vi phạm `json.loads` strict mode. Sửa bằng `json.loads(..., strict=False)` ở 5 điểm parse JSON từ LLM (`reasoner.py`, `analyzer.py`, `image.py`, `table.py`, `shared_src/file_readers.py`) — chỉ nới lỏng đúng quy tắc control-character, không nới các lỗi cú pháp JSON khác.
- **Đo**: `results/submission_runs.md` tag `v9-Q6-scholarship-compute+json-strict-fix` — exact_match **5/8 → 6/8**. Q6 đúng (SHINNYO), Q12 (llm_judge) đúng trở lại. Không câu nào khác bị regression.

## 2026-07-08 — Retrieval cross-file: quét toàn bộ + boost keyword hiếm + fix modality word-boundary

- **Đổi**: `retrieval/hybrid.py` + `config.py` (`cross_file_top_k=14`) + `analysis/analyzer.py`.
  - Câu `needs_multiple_sources` giờ **quét toàn bộ chunk** (không cắt cửa sổ 64) và trả `cross_file_top_k` file.
  - Thêm **boost keyword hiếm** (`_distinctive_content_matches`): file chứa 1 từ đặc trưng xuất hiện ở ≤3 file (proper noun như "NovaCare") được +3.0 điểm — nguyên lý IDF mà điểm tổng làm loãng.
  - Fix `_modality_hint` dùng **word-boundary** (`_has_word`): "hinh" không còn khớp nhầm trong "chinh" → câu text không bị route sang modality=image.
- **Vì sao**: Q12 (so sánh 3 dự án) chỉ lấy 2/3 file — file "NovaCare" (04) xếp hạng chunk #129/549, ngoài cửa sổ 64; và modality bị nhầm thành image đẩy văng file text.
- **Đo**: `results/retrieval_recall.md` tag `v8-crossfile-boost+modality-fix` — Fully-retrieved **0.786 → 0.857** (Q12 giờ 3/3), MRR 1.0. Q2 cũng có đủ 8 table candidate cho multi-table join. End-to-end `run_v8` (`results/submission_runs.md` tag `v8-Q1+Q11+crossfile`): exact_match **4/8 → 5/8** (Q1=16 đúng nhờ fix headerless sheet); Q12 llm_judge retrieve đủ 3/3 file, đáp án khớp GT; Q11 markdown đã sạch.

## 2026-07-08 — Code-first counting cho câu "đếm ảnh theo folder" (vision_count_compute)

- **Đổi**: `readers/image.py` (`count_matching_images`) + `readers/dispatcher.py` + prompt `reasoning/reasoner.py`. Với câu dạng "How many images in <folder> contain X?": viết lại câu hỏi thành Yes/No cho từng ảnh (1 LLM call rẻ), hỏi mỗi ảnh câu đó, đếm số match **bằng code**; chèn 1 block "Computed..." authoritative; đồng thời **bỏ dòng vision cũ** (hỏi nguyên câu hỏi tổng hợp cho từng ảnh → sinh nhiễu). Bật/tắt bằng `use_vision_count_compute` / `--no-vision-count`.
- **Vì sao**: cũ gửi nguyên câu hỏi tổng hợp cho từng ảnh riêng → ảnh trả "Not enough data"/số bịa; Buoc 4 tự đếm từ 12 đoạn nhiễu → sai (Q4=3, Q5=1). Ngoài ra: chỉ thêm block đúng vẫn chưa đủ — Buoc 4 quan sát được vẫn tự đếm lại từ dòng nhiễu cũ, nên phải bỏ hẳn tín hiệu mâu thuẫn.
- **Đo**: `results/submission_runs.md`. Kiểm chứng end-to-end qua pipeline thật (tái dùng cache): **Q4 3→8 = ĐÚNG** (gt 8). Q5 2 (gt 3) — còn lệch 1 do model nhìn nhầm màu 1 ảnh, không phải lỗi kiến trúc. Exact_match kỳ vọng 3/8 → 4/8.

## 2026-07-07 — Sum-pooling → Max-pooling khi gộp điểm chunk theo file

- **Đổi**: `retrieval/hybrid.py` (`_merge_candidate`) — điểm của 1 file = điểm chunk tốt nhất, thay vì cộng dồn điểm mọi chunk khớp.
- **Vì sao**: cộng dồn thiên vị tài liệu nhiều chunk (file dài 61 chunk thắng file 1 chunk dù chunk đó khớp chính xác hơn).
- **Đo**: `results/retrieval_recall.md` — MRR 0.49 → 0.93, Fully-retrieved 71% → 79% (tag `fastembed-0.5-0.5+maxpool+whisper` so với `fastembed-0.5/0.5`).

## 2026-07-07 — Trọng số hybrid retrieval: 0.6/0.4 → 0.5/0.5 (vector/bm25)

- **Đổi**: `config.py` — `vector_weight`, `bm25_weight`.
- **Vì sao**: 0.6/0.4 nghiêng semantic quá mức làm loãng match từ khóa chính xác (`class_grades.sql`); 0.5/0.5 cân bằng cả hai.
- **Đo**: `results/retrieval_recall.md` — Fully-retrieved 64% → 71%, recall_full 0.745 → 0.82.

## 2026-07-07 — Broaden phạm vi "lấy toàn bộ file trong folder" (full-pattern-set)

- **Đổi**: `retrieval/hybrid.py` (`_full_pattern_set`) — mở rộng điều kiện từ chỉ `wildcard_patterns` (cú pháp `folder/*`) sang cả `quoted_phrases`/`explicit_file_hints` (tên folder trong ngoặc kép); mở rộng tập reason được bảo vệ.
- **Vì sao**: câu hỏi thực tế dùng `"number_image"` (quoted), không phải `number_image/*`, nên bản đầu bỏ sót — Q4/Q5 vẫn đếm thiếu.
- **Đo**: offline retrieve() test — Q4/Q5 từ 8/12 (may rủi tie-break) → 12/12 file. Cần đo lại điểm exact_match qua `results/submission_runs.md` sau lần chạy full tiếp theo.

## Backlog — cần thử nghiệm/đổi phương pháp

| # | Vấn đề | Hướng thử nghiệm |
|---|---|---|
| 1 | Q1: sheet Excel 1-cột không header đếm thiếu 1 dòng | Không phải tham số retrieval — sửa cách đọc file, không cần A/B |
| 2 | Q9: extractive fallback trả rác thay vì "Not enough data" | Có thể cần đổi ngưỡng chấp nhận fallback — đáng đo trước/sau |
| 3 | Q12: `modality_hint` phân loại sai (`image` cho câu hỏi thuần text) + 1/3 file cross-file điểm gần 0 | Đang điều tra — nếu sửa bằng cách đổi cách tính modality_hint hoặc boost `needs_multiple_sources`, cần đo lại retrieval_recall |
| 4 | Whisper model size `base` → `medium`/`large` | Đổi model, ảnh hưởng chất lượng transcript — nên đo nếu nâng cấp |
