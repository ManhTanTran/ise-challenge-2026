# Known Issues — theo dõi lỗi theo từng bộ câu hỏi (dataset)

Cấu trúc: mỗi **dataset** (bộ câu hỏi) có 1 **folder con cùng tên** trong đây.

- `0.Sample_Data/` — mỗi câu chưa đúng là 1 file `Qxx.md`. Nội dung: **triệu chứng → nguyên nhân gốc → hướng fix**. Khi câu đã đúng (xác nhận qua `results/submission_runs.md` ở một run thật) → **xóa file đó đi**.
- Các dataset khác (`generated_hard_questions/`, `generated_sample_data/`, ...) — mỗi folder chứa `error_analysis.md`: tổng hợp điểm số qua các lần chạy + từng câu sai + nguyên nhân + hướng cải tiến đề xuất. Cập nhật file này sau **mỗi lần chạy full pipeline** trên dataset đó (không tạo file mới mỗi lần — sửa/nối vào file cũ).

**Trạng thái mới nhất theo dataset**:

| Dataset | Overall | Ghi chú |
|---|---|---|
| [`0.Sample_Data/`](0.Sample_Data/) | 14/15 (93.3%) | Chỉ còn Q03 mở |
| [`generated_hard_questions/`](generated_hard_questions/error_analysis.md) | 8/10 (80%) | 5 bug thật đã fix; Q5 (đọc chart) + Q7 (llm_judge subjective) còn mở |
| [`generated_sample_data/`](generated_sample_data/error_analysis.md) | 14/15 (93.3%) | Q3 (llm_judge) lệch nhẹ — biến thiên chủ quan, không phải bug |

Chấm cả `exact_match` lẫn `llm_judge` bằng `tools/llm_judge_score.py` (xem bên dưới).

## `0.Sample_Data/` — câu còn mở

| Câu | Loại | Trạng thái |
|---|---|---|
| [Q03](0.Sample_Data/Q03.md) | llm_judge | Trả lời mô tả tác động của chiến lược nhưng thiếu nội dung cốt lõi (Kinh Châu/Ích Châu, liên minh Tôn Quyền, Bắc phạt). Chưa điều tra retrieval/context. |

## Công cụ mới: chấm điểm llm_judge tự động

`tools/llm_judge_score.py` — trước đây `write_error_analysis` chỉ chấm `exact_match`, còn `llm_judge` luôn để trống `is_correct` (phải tự đọc tay). Giờ có 1 LLM call chấm từng câu llm_judge, ra `judge_verdict` + `judge_reason` lặp lại được. Chạy sau mỗi lần `run_pipeline`:

```
python -X utf8 -m approaches.approach_3_agentic_rag.tools.llm_judge_score \
  --error-analysis "approaches/approach_3_agentic_rag/outputs/run_vN/error_analysis.csv"
```

**Lưu ý**: bản thân judge cũng có thể sai (quan sát được ở Q2 run_v10: judge nói thiếu SMARCA4 dù thực ra có mặt, có thể vì câu trả lời quá dài). Coi kết quả judge là tín hiệu mạnh, không phải tuyệt đối.

## Đã fix (lịch sử)

- ~~Q06~~ — code-first scholarship reader (VLM trích bảng có cấu trúc, code chọn max). `run_v9`.
- ~~Q01~~ (regression tạm thời trong lúc fix Q2, đã sửa lại) — shortlist sheet đôi khi làm lộ sheet nhầm lẫn từng bị `truncate_text` che khuất; sửa bằng rule "không tự bịa thêm ngưỡng lọc" trong codegen prompt. `run_v10`.
- ~~Q05~~ — digit dạng icon/badge (số trắng trong khung tròn đặc màu): coi màu badge là màu digit, kèm ví dụ cụ thể trong prompt. `run_v10`.
- ~~Q08~~ — roster compute: trích JSON từng project (is_core, members), code đếm `len(members)` loại SV mới, chọn max trong core projects. Trả về **số lượng thành viên** (đã xác nhận với người dùng), không phải số thứ tự project. `run_v10`.
- ~~Q01~~, ~~Q02~~, ~~Q13~~ — phát hiện qua bộ `generated_hard_questions` (header Excel lệch dòng, context budget không công bằng, regression rule multiple-choice vs digit-form). `run_v13`.

Xem chi tiết từng fix trong `docs/CHANGELOG.md`.
