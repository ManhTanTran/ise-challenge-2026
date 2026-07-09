from pathlib import Path

from approaches.approach_2_hybrid_rag.shared_src import file_readers
from approaches.approach_2_hybrid_rag.shared_src.file_readers import (
    _should_use_vlm_image_parse,
    read_image,
)


def test_empty_ocr_triggers_vlm_caption():
    assert _should_use_vlm_image_parse(
        {
            "plain_text": "",
            "words": [],
            "lines": [],
            "avg_confidence": 0,
        }
    )


def test_read_image_ocr_failure_uses_vlm_caption_and_cache(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "digit.png"
    parse_path = tmp_path / "digit.txt.image_parse.json"
    calls = []

    def fail_ocr(path: Path):
        raise RuntimeError("tesseract unavailable")

    def fake_vlm(path: Path):
        calls.append(path)
        return {
            "plain_text": "7",
            "caption": "A large blue digit 7 on a circular background.",
            "description": "The image contains exactly one visible blue digit.",
            "visible_objects": ["blue digit", "number 7"],
            "tables": [],
            "key_values": {"digit": "7"},
            "confidence": 0.95,
            "engine": "vlm",
        }

    monkeypatch.setattr(file_readers, "_local_image_parse", fail_ocr)
    monkeypatch.setattr(file_readers, "_vlm_image_parse", fake_vlm)

    result = read_image(image_path, parse_cache_path=parse_path)
    cached = read_image(image_path, parse_cache_path=parse_path)

    assert result.error is None
    assert "Visible text: 7" in result.content
    assert "Caption: A large blue digit 7" in result.content
    assert result.metadata["ocr_engine"] == "vlm"
    assert result.metadata["image_caption"].startswith("A large blue digit")
    assert result.metadata["image_parse_path"] == str(parse_path)
    assert cached.content == result.content
    assert len(calls) == 1
