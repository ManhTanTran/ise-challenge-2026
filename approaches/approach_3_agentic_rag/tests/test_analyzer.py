from approaches.approach_3_agentic_rag.analysis.analyzer import analyze_question
from approaches.approach_3_agentic_rag.config import Approach3Config


def _offline_config() -> Approach3Config:
    config = Approach3Config()
    config.use_llm_analysis = False
    return config


def _profile(question: str, answer_type: str = "exact_match"):
    return analyze_question(
        {"id": 1, "question": question, "answer_type": answer_type},
        config=_offline_config(),
    )


def test_explicit_filename_detection():
    profile = _profile("What is the mean age in file Credit.csv?")
    assert "credit.csv" in [hint.lower() for hint in profile.explicit_file_hints]
    assert profile.modality_hint == "table"
    assert profile.requires_computation


def test_wildcard_pattern_detection():
    profile = _profile("How many images are in number_image/*?")
    assert "number_image/*" in profile.wildcard_patterns
    assert profile.modality_hint == "image"


def test_format_instructions_decimals():
    profile = _profile("What is the average score, rounded to 2 decimal places?")
    assert profile.format_instructions.get("decimals") == 2


def test_vietnamese_language_detected():
    profile = _profile("Điểm trung bình môn Toán của lớp 10A1 là bao nhiêu?")
    assert profile.language == "vi"
    assert profile.requires_computation


def test_yes_no_binary_flag():
    profile = _profile("Did the revenue increase in 2024?")
    assert profile.format_instructions.get("binary") is True


def test_modality_hint_not_image_for_chinh_substring():
    # "chính" (chinh) must not trigger the image cue "hinh" -> text question.
    profile = _profile(
        "Điểm chung chính trong cách các dự án tạo ra tác động bền vững là gì?",
        answer_type="llm_judge",
    )
    assert profile.modality_hint != "image"


def test_modality_hint_image_for_real_image_word():
    profile = _profile("How many images contain a blue digit?")
    assert profile.modality_hint == "image"
