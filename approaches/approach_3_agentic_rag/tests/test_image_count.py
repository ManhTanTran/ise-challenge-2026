from pathlib import Path

from approaches.approach_3_agentic_rag.config import Approach3Config
from approaches.approach_3_agentic_rag.core.models import QuestionProfile
from approaches.approach_3_agentic_rag.readers import image as image_module
from approaches.approach_3_agentic_rag.retrieval.hybrid import is_folder_count_question
from approaches.approach_3_agentic_rag.shared_src.file_readers import _should_use_vlm_image_parse


def _profile(question: str, **kwargs) -> QuestionProfile:
    return QuestionProfile(question_id=1, question=question, **kwargs)


def test_sparse_ocr_triggers_vlm_caption():
    # A picture of a digit that OCR misread as garbage "ra" should escalate to VLM.
    assert _should_use_vlm_image_parse(
        {"plain_text": "ra", "words": ["ra"], "lines": ["ra"], "avg_confidence": 80}
    )


def test_rich_ocr_text_skips_vlm_caption():
    # A real text screenshot with plenty of confident text does not need the VLM.
    assert not _should_use_vlm_image_parse(
        {
            "plain_text": "This is a long confident paragraph of readable text on the slide.",
            "words": ["w"] * 12,
            "lines": ["l"] * 4,
            "avg_confidence": 90,
        }
    )


def test_candidate_caption_joins_caption_and_description():
    caption = image_module._candidate_caption(
        {"image_caption": "A blue number 5.", "image_description": "3D glossy render."}
    )
    assert "blue number 5" in caption.lower()
    assert "glossy" in caption.lower()


def test_is_folder_count_question_true_for_quoted_folder_plus_how_many():
    profile = _profile(
        'How many images in "number_image" contain a blue digit?',
        quoted_phrases=["number_image"],
    )
    assert is_folder_count_question(profile)


def test_is_folder_count_question_false_without_counting_keyword():
    profile = _profile(
        'What does the image in "number_image" show?',
        quoted_phrases=["number_image"],
    )
    assert not is_folder_count_question(profile)


def test_is_folder_count_question_false_without_folder_hint():
    profile = _profile("How many people attended the workshop?")
    assert not is_folder_count_question(profile)


def test_count_matching_images_counts_true_matches_via_code(tmp_path: Path, monkeypatch):
    profile = _profile('How many images in "n" contain a blue digit?')
    config = Approach3Config()
    config.analysis_model = "test-model"
    config.vision_model = "test-model"

    candidates = [
        {"relative_path": f"n/img{i}.jpg", "absolute_path": str(tmp_path / f"img{i}.jpg"), "modality": "image"}
        for i in range(4)
    ]
    for c in candidates:
        Path(c["absolute_path"]).write_bytes(b"fake")

    monkeypatch.setattr(image_module, "has_llm", lambda: True)
    monkeypatch.setattr(
        image_module,
        "_rewrite_per_item_question",
        lambda question, *, model: "Does this image show a blue digit?",
    )

    # img0, img2 match; img1 doesn't; img3 is unparseable (matches=None) -> excluded from both totals.
    canned = {
        candidates[0]["absolute_path"]: {"matches": True, "reason": "blue 5"},
        candidates[1]["absolute_path"]: {"matches": False, "reason": "red 3"},
        candidates[2]["absolute_path"]: {"matches": True, "reason": "blue 7"},
        candidates[3]["absolute_path"]: {"matches": None, "reason": ""},
    }

    def fake_structured(question, path, *, model, caption=""):
        return canned[str(path)]

    monkeypatch.setattr(image_module, "_call_vision_structured", fake_structured)

    result = image_module.count_matching_images(
        profile, candidates, config=config, cache_dir=tmp_path / "vision_cache"
    )
    assert result is not None
    assert result["total"] == 3  # img3's unparseable result is excluded
    assert result["matched_count"] == 2
    assert set(result["matched_files"]) == {"n/img0.jpg", "n/img2.jpg"}


def test_count_matching_images_caches_per_image_results(tmp_path: Path, monkeypatch):
    profile = _profile('How many images in "n" contain a blue digit?')
    config = Approach3Config()
    config.vision_model = "test-model"

    candidate = {
        "relative_path": "n/img0.jpg",
        "absolute_path": str(tmp_path / "img0.jpg"),
        "modality": "image",
    }
    Path(candidate["absolute_path"]).write_bytes(b"fake")

    monkeypatch.setattr(image_module, "has_llm", lambda: True)
    monkeypatch.setattr(
        image_module, "_rewrite_per_item_question", lambda question, *, model: "Does this show blue?"
    )
    calls = []

    def fake_structured(question, path, *, model, caption=""):
        calls.append(path)
        return {"matches": True, "reason": "blue"}

    monkeypatch.setattr(image_module, "_call_vision_structured", fake_structured)

    cache_dir = tmp_path / "vision_cache"
    image_module.count_matching_images(profile, [candidate], config=config, cache_dir=cache_dir)
    image_module.count_matching_images(profile, [candidate], config=config, cache_dir=cache_dir)

    assert len(calls) == 1  # second call hit the on-disk cache, no repeat vision call


def test_count_matching_images_returns_none_when_rewrite_fails(tmp_path: Path, monkeypatch):
    profile = _profile('How many images in "n" contain a blue digit?')
    config = Approach3Config()

    monkeypatch.setattr(image_module, "has_llm", lambda: True)
    monkeypatch.setattr(image_module, "_rewrite_per_item_question", lambda question, *, model: None)
    result = image_module.count_matching_images(
        profile,
        [{"relative_path": "n/img0.jpg", "absolute_path": str(tmp_path / "img0.jpg"), "modality": "image"}],
        config=config,
        cache_dir=tmp_path / "vision_cache",
    )
    assert result is None


def test_structured_vision_prompt_treats_icon_badge_color_as_digit_color():
    # A white "7" cut out of a solid blue circle icon was judged "not blue"
    # because only the glyph strokes were considered - the badge's own color
    # should count too, so the per-image judgment prompt must say so.
    prompt = image_module._STRUCTURED_VISION_PROMPT
    assert "badge" in prompt.lower() or "icon" in prompt.lower()
    assert "dominant color" in prompt.lower()
