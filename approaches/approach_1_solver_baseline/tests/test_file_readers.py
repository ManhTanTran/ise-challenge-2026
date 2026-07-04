from pathlib import Path

import pandas as pd

from src import file_readers
from src.file_indexer import index_file
from src.file_readers import _parse_tesseract_data, _repair_mojibake, looks_like_mojibake, read_file, read_image


def test_reader_missing_file_does_not_crash(tmp_path: Path):
    result = read_file(tmp_path / "missing.unknown")
    assert result.error


def test_reader_unsupported_file_does_not_crash(tmp_path: Path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"\x00\x01")
    result = read_file(file_path)
    assert result.error


def test_csv_reader_and_index_keys(tmp_path: Path):
    data_lake = tmp_path / "lake"
    cache_dir = tmp_path / "cache"
    data_lake.mkdir()
    csv_path = data_lake / "sample.csv"
    pd.DataFrame({"Limit": [1, 2], "Balance": [3, 4]}).to_csv(csv_path, index=False)

    result = read_file(csv_path, cache_dir=cache_dir, data_lake_dir=data_lake)
    assert result.error is None
    assert result.metadata["columns"] == ["Limit", "Balance"]

    item = index_file(csv_path, data_lake, cache_dir=cache_dir)
    expected = {
        "file_id",
        "filename",
        "relative_path",
        "absolute_path",
        "extension",
        "modality",
        "mime_type",
        "size_bytes",
        "text_preview",
        "columns",
        "sheet_names",
        "extracted_text_path",
        "image_parse_path",
        "ocr_confidence",
        "ocr_engine",
        "status",
        "error_message",
    }
    assert expected.issubset(item)


def test_parse_tesseract_data_returns_structured_text():
    data = {
        "text": ["", "Hoc", "bong", "SHINNYO"],
        "conf": ["-1", "90", "80", "70"],
        "left": [0, 10, 50, 10],
        "top": [0, 10, 10, 30],
        "width": [0, 35, 40, 80],
        "height": [0, 10, 10, 10],
        "block_num": [0, 1, 1, 1],
        "par_num": [0, 1, 1, 1],
        "line_num": [0, 1, 1, 2],
        "word_num": [0, 1, 2, 1],
    }

    parsed = _parse_tesseract_data(data)

    assert parsed["plain_text"] == "Hoc bong\nSHINNYO"
    assert parsed["avg_confidence"] == 80
    assert len(parsed["words"]) == 3
    assert len(parsed["lines"]) == 2
    assert parsed["blocks"][0]["text"] == "Hoc bong\nSHINNYO"


def test_mojibake_detection_catches_broken_vietnamese():
    broken = "H\u00e1\u00bb\u008cC B\u00e1\u00bb\u0094NG"
    assert looks_like_mojibake(broken)
    assert _repair_mojibake(broken) == "HỌC BỔNG"


def test_read_image_low_confidence_uses_vlm_and_cache(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "image.png"
    parse_path = tmp_path / "image_parse.json"
    calls = []

    monkeypatch.setattr(
        file_readers,
        "_local_image_parse",
        lambda path: {
            "plain_text": "hÃ©c bÃ©ng",
            "blocks": [],
            "lines": [],
            "words": [],
            "tables": [],
            "key_values": {},
            "avg_confidence": 10,
            "engine": "pytesseract",
        },
    )

    def fake_vlm(path):
        calls.append(path)
        return {
            "plain_text": "Học bổng SHINNYO",
            "tables": [{"rows": [["SHINNYO", "215"]]}],
            "key_values": {},
            "confidence": 0.95,
        }

    monkeypatch.setattr(file_readers, "_vlm_image_parse", fake_vlm)

    result = read_image(image_path, parse_cache_path=parse_path)
    cached = read_image(image_path, parse_cache_path=parse_path)

    assert "SHINNYO" in result.content
    assert "SHINNYO" in cached.content
    assert len(calls) == 1
    assert result.metadata["image_parse_path"] == str(parse_path)
