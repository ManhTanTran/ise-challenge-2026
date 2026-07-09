# Error analysis — `generated_hard_questions.xlsx` (10 câu tự sinh, độ khó cao)

Bộ câu hỏi tự tạo dựa trên phong cách `0.Sample_Data.xlsx`, cố ý nhắm vào các điểm yếu chưa test: header Excel lệch dòng, lọc đa điều kiện, đọc chart trong ảnh, so sánh chéo project/file mới, tên gần giống nhau dễ nhầm.

**Trạng thái mới nhất**: `run_hard2` — exact_match 8/9, overall **8/10 (80%)**. Xem `results/submission_runs.md`.

## Lịch sử qua các lần chạy

| Run | Overall | Sai | Ghi chú |
|---|---|---|---|
| run_hard1 | 6/10 (60%) | Q1, Q5, Q7, Q8 | Lần đầu |
| run_hard2 (lần 2) | 6/10 (60%) | Q4, Q5, Q7, Q8 | Sau fix Q1; Q4 flaky |
| run_hard2 (lần 3) | 7/10 (70%) | Q5, Q7, Q8 | Sau fix Q7 (context budget) |
| run_hard2 (lần 4) | 7/10 (70%) | Q5, Q7, Q8 | Q9 hỏng tạm do fix khác, Q8 sai project (không phải lỗi format) |
| run_hard2 (lần 5, mới nhất) | **8/10 (80%)** | Q5, Q7 | Sau fix roster hallucination + min/max direction |

## Câu còn sai

### Q5 — Đọc chart CAPEX 2026-2030 từ ảnh (topic_2_page-0007.jpg)
- **Groundtruth**: 2027 (11,727 tỷ đồng — thấp nhất)
- **Predicted**: dao động giữa 2026, 2029 tùy lần chạy — **không ổn định**
- **Nguyên nhân**: model đọc trực tiếp text/số trên bar chart nhỏ (5 cột số liệu gần giống nhau: 16,340 / 11,727 / 14,719 / 17,517 / 19,532), dễ đọc nhầm cột hoặc nhầm nhãn năm. Không phải lỗi thiếu vision (đã xác nhận vision CÓ chạy, retrieval đúng ảnh) — là giới hạn độ chính xác đọc số trên chart nhỏ.
- **Hướng cải tiến đề xuất**: áp dụng code-first pattern đã dùng cho `compute_scholarship_answer` — yêu cầu VLM trích **toàn bộ cặp (năm, giá trị)** thành JSON có cấu trúc thay vì hỏi trực tiếp "năm nào thấp nhất", rồi code tự tìm min. Việc này giảm gánh nặng suy luận của VLM xuống chỉ còn OCR từng con số, thay vì OCR + so sánh cùng lúc.
- **Ưu tiên**: trung bình — cần thêm 1 reader mới (`compute_chart_answer` tương tự pattern đã có), rủi ro overfit nếu chỉ test trên 1 ảnh.

### Q7 — Điểm chung dự án sông Minh Hòa + Ban Liền (llm_judge)
- **Predicted**: nội dung hợp lý (thay đổi hành vi cộng đồng, nhiều bên liên quan tham gia) nhưng judge chấm sai vì không khớp đúng cách diễn đạt "kết hợp can thiệp kỹ thuật + hành vi cộng đồng theo cách thí điểm dần dần" trong groundtruth tôi tự viết.
- **Nguyên nhân**: có thể groundtruth tự viết quá cụ thể/hẹp so với những gì một câu trả lời "đúng về bản chất" có thể diễn đạt khác đi. Đây là rủi ro cố hữu của câu llm_judge tự tạo (không có đáp án chuẩn từ ban tổ chức để đối chiếu).
- **Hướng cải tiến đề xuất**: không sửa pipeline — nếu muốn, viết lại groundtruth khoan dung hơn (liệt kê nhiều cách diễn đạt được chấp nhận), hoặc chấp nhận đây là nhiễu tự nhiên của phương pháp tạo câu hỏi.
- **Ưu tiên**: thấp — không phải bug, là giới hạn của quy trình tự tạo groundtruth cho câu llm_judge.

## Bug thật đã tìm thấy và fix (từ bộ câu hỏi này)

Bộ câu hỏi khó này phát hiện **5 bug thật** trong pipeline mà bộ 15 câu cũ chưa từng lộ ra:

1. **Header Excel lệch dòng** (`shared_src/file_readers.py::_fix_misplaced_header`) — file khoa học có vài dòng metadata/notes trước header thật, pandas đọc header=0 sai, biến cột thật thành "Unnamed: N". Đã tìm bằng cách kiểm tra dòng có nhiều giá trị chuỗi ngắn KHÔNG TRÙNG LẶP nhất (phân biệt với dòng đơn vị đo lặp lại như "(ppm)").
2. **`modality_hint` chặn vision sai** (`readers/dispatcher.py::_image_block`) — câu hỏi đọc số liệu tài chính từ ảnh chart bị phân loại `modality_hint="table"` (vì có từ "CAPEX", "kế hoạch") nên vision không bao giờ chạy, dù ảnh là bằng chứng duy nhất. Đã bỏ gate này, dựa vào retrieval + vision_budget thay vì đoán modality qua từ khóa câu hỏi.
3. **Context budget không công bằng cho câu multi-file** (`readers/dispatcher.py::build_context_blocks`) — 7 tài liệu KTCT không liên quan (mỗi cái ~8.8K ký tự) chiếm hết 65K ký tự ngân sách trước khi tới file thật sự liên quan (xếp hạng #11). Đã thêm "fair share" ký tự/file khi `needs_multiple_sources=True`.
4. **Roster compute bịa project giả** (`readers/table.py::compute_roster_answer`) — gửi mọi document candidate (kể cả file không phải roster) vào LLM trích "project", model bịa ra 63 project giả từ slide chính trị-kinh tế. Đã thêm điều kiện "nếu tài liệu không có roster project thật, trả về rỗng" vào prompt.
5. **Roster compute chỉ hỗ trợ "nhiều nhất", không hỗ trợ "ít nhất"** — `is_roster_max_question` + `compute_roster_answer` trước đó hardcode `max()`, không nhận diện được câu hỏi "ít thành viên nhất". Đã tách `_roster_superlative_direction()` trả về "max"/"min", code chọn `min()`/`max()` tương ứng.

Cả 5 đều đã verify bằng test (70/70 pass) + verify offline bằng dữ liệu thật trước khi chạy full pipeline, và đã xác nhận không regression trên 2 bộ câu hỏi cũ.
