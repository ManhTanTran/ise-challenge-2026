# Results — bảng xếp hạng thí nghiệm

Thư mục này lưu **lịch sử kết quả** của các lần chạy đánh giá để so sánh các lần tune.

Quy ước: chỉ khi **đổi phương pháp/tham số** (top_k, trọng số vector/bm25, ngưỡng min_relevance, đổi model...) mới cần chạy lại eval/pipeline và log kết quả vào đây + ghi vào `../docs/CHANGELOG.md`. Bug fix/cài đặt môi trường không cần log.

## Files

- `retrieval_recall.jsonl` / `.md` — đo chất lượng **Bước 2 (retrieval)** một mình, offline, miễn phí. Xem phần "Ghi thêm kết quả" bên dưới.
- `submission_runs.jsonl` / `.md` — đo điểm **exact_match thật** của cả pipeline (tốn API). Log bằng `tools/log_submission_result.py` sau mỗi lần chạy `run_pipeline` có groundtruth.
- Quy tắc chung cho cả 2 board: `.jsonl` là **nguồn sự thật** (chỉ append, không ghi đè); `.md` là bảng tự sinh, sắp best-trên-đầu. **Không sửa tay file `.md`** — nó bị ghi đè mỗi lần log. Muốn sửa lịch sử thì sửa `.jsonl`.

## Ghi thêm kết quả

Chạy eval với `--tag` để đặt tên lần chạy; nó tự append vào leaderboard:

```powershell
python -X utf8 -m approaches.approach_3_agentic_rag.tools.eval_retrieval `
  --questions "data\sample_data_lake\0.Sample_Data.xlsx" `
  --data-lake "data\sample_data_lake\Data-Lake" `
  --file-index "approaches\approach_1_solver_baseline\outputs\runs\parse_20260630_095706\file_index.json" `
  --work-dir "approaches\approach_3_agentic_rag\outputs\eval_fastembed" `
  --vector-weight 0.5 --bm25-weight 0.5 `
  --tag "ten-thi-nghiem-cua-ban"
```

Thêm `--no-log` nếu chỉ muốn xem, không ghi vào leaderboard.

### Cột trong bảng `retrieval_recall`

| Cột | Nghĩa |
|---|---|
| `fully` | tỷ lệ câu lấy **đủ tất cả** file đúng (metric chính) |
| `recall_full` | trung bình % file đúng tìm được (trên cả danh sách) |
| `recall@8` | chất lượng xếp hạng trong top-8 |
| `mrr` | thứ hạng file đúng đầu tiên (1.0 = luôn #1) |
| `unans_ok` | số câu "không đủ dữ liệu" trả rỗng đúng / tổng |
| `misses` | các câu chưa lấy đủ file đúng |

## Log điểm submission (sau khi chạy full pipeline có groundtruth)

```powershell
python -X utf8 -m approaches.approach_3_agentic_rag.tools.log_submission_result `
  --error-analysis "approaches\approach_3_agentic_rag\outputs\run_v3\error_analysis.csv" `
  --tag "ten-mo-ta-thay-doi" `
  --notes "tom tat code da doi so voi lan truoc"
```

### Cột trong bảng `submission_runs`

| Cột | Nghĩa |
|---|---|
| `exact_match` | số câu đúng / tổng câu `exact_match` (metric chính, tự chấm được) |
| `exact_pct` | tỷ lệ đúng, dùng để sắp hạng |
| `wrong_ids` | các câu exact_match còn sai |
| `notes` | ghi chú tay — nên trỏ lại mục tương ứng trong `../docs/CHANGELOG.md` |
