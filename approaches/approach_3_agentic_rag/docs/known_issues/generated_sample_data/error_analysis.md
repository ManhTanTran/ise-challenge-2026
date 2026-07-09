# Error analysis — `generated_sample_data.xlsx` (15 câu tự sinh)

Bộ câu hỏi tự tạo đầu tiên, tái sử dụng cùng Data-Lake với `0.Sample_Data.xlsx` nhưng hỏi về các file khác (đa số là `da-dev-tables/`, `archeology/` chưa từng được hỏi tới).

**Trạng thái mới nhất**: `run_gen3` — exact_match **13/13 (100%)**, overall **14/15 (93.3%)**. Q3 (llm_judge) lệch nhẹ, xem bên dưới.

## Lịch sử

| Run | Overall | Ghi chú |
|---|---|---|
| run_gen1 | 12/13 exact_match (chưa chấm llm_judge riêng) | Q14 sai: cột `income` trong `census.csv` có khoảng trắng đầu (`' >50K'`), filter `==` không strip → ra 0 thay vì 7841 |
| run_gen2 | 15/15 (100%) | Sau khi thêm rule `.str.strip()` cho filter string vào `_CODEGEN_PROMPT` (`readers/table.py`) |
| run_gen3 | **14/15 (93.3%)** | Chạy lại sau loạt fix header/vision-gate/context-budget/roster (xem `generated_hard_questions/error_analysis.md`) để xác nhận không regression — exact_match giữ nguyên 13/13; Q3 (llm_judge) lệch nhẹ |

## Bug thật đã tìm thấy và fix

**Q14 — whitespace trong cột categorical**: `readers/table.py::_CODEGEN_PROMPT` không nhắc LLM rằng cột string có thể có khoảng trắng thừa không nhìn thấy khi xem sample. Đã thêm rule bắt buộc dùng `.str.strip() == 'value'` thay vì `== 'value'` trần cho mọi filter string. Xem `docs/CHANGELOG.md` mục "Fix whitespace filter".

## Câu còn mở (không phải bug)

**Q3 — Điểm chung NovaCare + redesign xe buýt An Phu (llm_judge)**: câu trả lời đúng phần "dùng dữ liệu/công nghệ để tối ưu" nhưng thiếu nhấn mạnh "tính bền vững phụ thuộc vào con người" mà groundtruth (tự viết) yêu cầu. Cùng loại biến thiên chủ quan đã gặp ở `generated_hard_questions` Q7 — groundtruth tự tạo cho câu llm_judge có thể quá cụ thể so với các cách diễn đạt "đúng bản chất" khác. Không sửa pipeline; nếu muốn cải thiện cần viết lại groundtruth khoan dung hơn.

Cần chạy lại nếu có thay đổi lớn ở `compute_table_answer`/`_CODEGEN_PROMPT`/`reasoner.py` để xác nhận vẫn giữ điểm.
