from pathlib import Path

from approaches.approach_3_agentic_rag.config import Approach3Config
from approaches.approach_3_agentic_rag.core.models import QuestionProfile
from approaches.approach_3_agentic_rag.readers import image as image_module


def _profile(question: str) -> QuestionProfile:
    return QuestionProfile(question_id=1, question=question)


def test_is_scholarship_slot_question_true():
    assert image_module.is_scholarship_slot_question(
        "Học bổng ngoài ngân sách nào có số lượng suất trao nhiều nhất?"
    )
    assert image_module.is_scholarship_slot_question(
        "Which scholarship has the most slots awarded?"
    )


def test_is_scholarship_slot_question_false_without_slot_term():
    # Mentions scholarships but asks nothing about counts/slots.
    assert not image_module.is_scholarship_slot_question(
        "What is the application deadline for the scholarship program?"
    )


def test_is_scholarship_slot_question_false_without_scholarship_term():
    assert not image_module.is_scholarship_slot_question("How many slots are in the parking lot?")


def test_clean_scholarship_name_strips_country_and_uppercases():
    assert image_module._clean_scholarship_name("Shinnyo, Nhật Bản") == "SHINNYO"
    assert image_module._clean_scholarship_name("yamada") == "YAMADA"


def test_extract_numbers_from_mixed_components():
    assert image_module._extract_numbers([13, "8 suất", "05"]) == [13.0, 8.0, 5.0]


def test_compute_scholarship_answer_picks_max_off_budget_row(tmp_path: Path, monkeypatch):
    profile = _profile(
        'Học bổng ngoài ngân sách nào có số lượng suất trao nhiều nhất? Trả lời viết hoa.'
    )
    config = Approach3Config()
    config.vision_model = "test-model"

    candidate = {
        "relative_path": "scholarship1.png",
        "absolute_path": str(tmp_path / "scholarship1.png"),
        "modality": "image",
    }
    Path(candidate["absolute_path"]).write_bytes(b"fake")

    monkeypatch.setattr(image_module, "has_llm", lambda: True)

    def fake_rows(path, *, model, caption=""):
        return [
            {
                "scholarship_name": "ĐINH THIỆN LÝ",
                "country": "",
                "off_budget": False,
                "slot_components": [672.0],
                "_slot_count": 672.0,
                "confidence": "high",
            },
            {
                "scholarship_name": "YAMADA",
                "country": "Nhật Bản",
                "off_budget": True,
                "slot_components": [13.0],
                "_slot_count": 13.0,
                "confidence": "high",
            },
            {
                "scholarship_name": "SHINNYO",
                "country": "Nhật Bản",
                "off_budget": True,
                "slot_components": [13.0, 8.0, 5.0],
                "_slot_count": 26.0,
                "confidence": "high",
            },
        ]

    monkeypatch.setattr(image_module, "_call_vision_rows", fake_rows)

    result = image_module.compute_scholarship_answer(
        profile, [candidate], config=config, cache_dir=tmp_path / "vision_cache"
    )
    assert result is not None
    # ĐINH THIỆN LÝ has the highest raw count but off_budget=False (state
    # budget) - the question asks for "ngoài ngân sách" only, so it must be
    # excluded even though its count dwarfs every off-budget row.
    assert result["answer"] == "SHINNYO"


def test_compute_scholarship_answer_none_when_no_off_budget_rows_and_none_positive(tmp_path: Path, monkeypatch):
    profile = _profile("Học bổng ngoài ngân sách nào có số lượng suất trao nhiều nhất?")
    config = Approach3Config()
    config.vision_model = "test-model"

    candidate = {
        "relative_path": "unrelated.jpg",
        "absolute_path": str(tmp_path / "unrelated.jpg"),
        "modality": "image",
    }
    Path(candidate["absolute_path"]).write_bytes(b"fake")

    monkeypatch.setattr(image_module, "has_llm", lambda: True)
    monkeypatch.setattr(image_module, "_call_vision_rows", lambda path, *, model, caption="": [])

    result = image_module.compute_scholarship_answer(
        profile, [candidate], config=config, cache_dir=tmp_path / "vision_cache"
    )
    assert result is None


def test_compute_scholarship_answer_returns_none_for_unrelated_question(tmp_path: Path, monkeypatch):
    profile = _profile("What is the average balance in Credit.csv?")
    config = Approach3Config()
    monkeypatch.setattr(image_module, "has_llm", lambda: True)
    result = image_module.compute_scholarship_answer(
        profile, [{"relative_path": "a.png", "absolute_path": str(tmp_path / "a.png"), "modality": "image"}],
        config=config,
        cache_dir=tmp_path / "vision_cache",
    )
    assert result is None


def test_compute_scholarship_answer_caches_row_extraction(tmp_path: Path, monkeypatch):
    profile = _profile("Học bổng ngoài ngân sách nào có số lượng suất trao nhiều nhất?")
    config = Approach3Config()
    config.vision_model = "test-model"

    candidate = {
        "relative_path": "scholarship1.png",
        "absolute_path": str(tmp_path / "scholarship1.png"),
        "modality": "image",
    }
    Path(candidate["absolute_path"]).write_bytes(b"fake")
    monkeypatch.setattr(image_module, "has_llm", lambda: True)

    calls = []

    def fake_rows(path, *, model, caption=""):
        calls.append(path)
        return [
            {
                "scholarship_name": "SHINNYO",
                "country": "Nhật Bản",
                "off_budget": True,
                "slot_components": [26.0],
                "_slot_count": 26.0,
                "confidence": "high",
            }
        ]

    monkeypatch.setattr(image_module, "_call_vision_rows", fake_rows)
    cache_dir = tmp_path / "vision_cache"
    image_module.compute_scholarship_answer(profile, [candidate], config=config, cache_dir=cache_dir)
    image_module.compute_scholarship_answer(profile, [candidate], config=config, cache_dir=cache_dir)
    assert len(calls) == 1
