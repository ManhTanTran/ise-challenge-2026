from pathlib import Path

from openpyxl import Workbook

from approaches.approach_3_agentic_rag.shared_src.file_readers import read_excel


def test_read_excel_handles_body_wider_than_first_row(tmp_path: Path):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "WideBody"
    worksheet.append(["A", "B", "C", "D"])
    worksheet.append([f"value_{index}" for index in range(38)])
    path = tmp_path / "wide_body.xlsx"
    workbook.save(path)

    result = read_excel(path)

    assert result.error is None
    assert result.metadata["sheets"]["WideBody"]["shape"] == [2, 38]
    assert len(result.metadata["sheets"]["WideBody"]["columns"]) == 38
    assert "Unnamed: 38" in result.content
    assert "value_37" in result.content
