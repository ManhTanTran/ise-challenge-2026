from pathlib import Path

from openpyxl import Workbook

from approaches.approach_3_agentic_rag.shared_src.file_readers import read_csv, read_excel


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


def test_read_excel_skips_leading_title_and_notes_rows(tmp_path: Path):
    # Mirrors climateMeasurements.xlsx: a title row, a long notes row, then
    # the real header several rows down. Row 0 must not become the header.
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Data"
    worksheet.append(["Data Supplement to: some long paper title"])
    worksheet.append([])
    worksheet.append(["Note: methodology details go here"])
    worksheet.append([])
    worksheet.append(["Site", "Age_ky", "Fe"])
    worksheet.append(["ODP 967", 0.04, 5475.7])
    worksheet.append(["ODP 967", 0.06, 1298.3])
    path = tmp_path / "climate_like.xlsx"
    workbook.save(path)

    result = read_excel(path)

    assert result.error is None
    assert result.metadata["sheets"]["Data"]["columns"] == ["Site", "Age_ky", "Fe"]
    assert "Unnamed" not in result.content
    assert "ODP 967" in result.content


def test_read_csv_skips_leading_title_line_before_ragged_header(tmp_path: Path):
    # Reproduces a real Colab traceback: "Expected 1 fields in line 4, saw 4"
    # - pandas locks onto the 1-field title line as the header, then chokes
    # once it reaches the real 4-column data below.
    path = tmp_path / "ragged.csv"
    path.write_text(
        "Some Title Line\nNote: extra metadata here\na,b,c,d\n1,2,3,4\n5,6,7,8\n9,10,11,12\n",
        encoding="utf-8",
    )

    result = read_csv(path)

    assert result.error is None
    assert result.metadata["columns"] == ["a", "b", "c", "d"]
    assert result.metadata["shape"] == [3, 4]


def test_read_csv_skips_genuinely_bad_lines_as_last_resort(tmp_path: Path):
    # A truly ragged row in the middle of otherwise-consistent data (not a
    # leading title row) - skiprows detection won't fix this, so the
    # on_bad_lines="skip" fallback must still return the rest of the file
    # instead of raising.
    path = tmp_path / "mixed_ragged.csv"
    path.write_text("a,b,c\n1,2,3\n4,5,6,7,8\n9,10,11\n", encoding="utf-8")

    result = read_csv(path)

    assert result.error is None
    assert result.metadata["columns"] == ["a", "b", "c"]
