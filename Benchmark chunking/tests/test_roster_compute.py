from approaches.approach_3_agentic_rag.config import Approach3Config
from approaches.approach_3_agentic_rag.core.models import QuestionProfile
from approaches.approach_3_agentic_rag.readers import table as table_module

Q8_QUESTION = (
    "Project core nào của AXIOM - iSE có nhiều thành viên hiện tại nhất, "
    "không tính số SV mới? Yêu cầu chỉ trả về đúng số thứ tự của project."
)


def _profile(question: str) -> QuestionProfile:
    return QuestionProfile(question_id=1, question=question)


def _candidate(relative_path: str) -> dict:
    return {"relative_path": relative_path, "absolute_path": f"/fake/{relative_path}", "modality": "document"}


def _config() -> Approach3Config:
    config = Approach3Config()
    config.answer_model = "test-model"
    return config


def test_is_roster_max_question_true_for_real_q8_phrasing():
    assert table_module.is_roster_max_question(Q8_QUESTION)


def test_is_roster_max_question_false_without_new_member_exclusion():
    assert not table_module.is_roster_max_question(
        "Project nào có nhiều thành viên nhất?"
    )


def test_is_roster_max_question_false_for_unrelated_question():
    assert not table_module.is_roster_max_question("What is the average balance in Credit.csv?")


def test_is_roster_max_question_true_for_fewest_phrasing():
    # "it thanh vien nhat" (fewest members) - the inverse direction of the
    # original "most" phrasing must also be recognized as a roster question.
    assert table_module.is_roster_max_question(
        "Project core nào có ít thành viên hiện tại nhất, không tính số SV mới?"
    )
    assert table_module.is_roster_max_question(
        "Which core project has the fewest current members, excluding new students?"
    )


def test_roster_superlative_direction_max_vs_min():
    from approaches.approach_3_agentic_rag.shared_src.utils import normalize_for_match

    assert table_module._roster_superlative_direction(
        normalize_for_match("project nào có nhiều thành viên nhất")
    ) == "max"
    assert table_module._roster_superlative_direction(
        normalize_for_match("project nào có ít thành viên nhất")
    ) == "min"
    assert table_module._roster_superlative_direction(
        normalize_for_match("which project has the fewest members")
    ) == "min"


def test_compute_roster_answer_picks_max_current_count_among_core_projects(monkeypatch):
    # Mirrors the real AXIOM PDF: project 4 has more total people (4 current +
    # 2 new = 6) but fewer CURRENT members than project 5 (7 current + 1 new).
    # The winner must be judged on current members only, and a non-core
    # project with an even bigger roster must not win.
    projects = [
        {"project_no": "1", "title": "Data Engineering Research", "is_core": True, "members": ["A", "B", "C"]},
        {"project_no": "4", "title": "Data Intelligence R&D", "is_core": True, "members": ["D", "E", "F", "G"]},
        {
            "project_no": "5",
            "title": "Platform AXIOM",
            "is_core": True,
            "members": ["H", "I", "J", "K", "L", "M", "N"],
        },
        {
            "project_no": "6",
            "title": "Chinh ly tu dong",
            "is_core": False,
            "members": ["O", "P", "Q", "R", "S", "T", "U", "V", "W", "X"],
        },
    ]

    monkeypatch.setattr(table_module, "has_llm", lambda: True)
    monkeypatch.setattr(table_module, "extract_candidate_text", lambda candidate: "fake pdf text")
    monkeypatch.setattr(table_module, "_call_roster_extract", lambda text, *, model: [dict(p, _current_count=len(p["members"])) for p in projects])

    result = table_module.compute_roster_answer(
        _profile(Q8_QUESTION), [_candidate("axiom.pdf")], config=_config()
    )
    assert result is not None
    assert result["answer"] == "7"
    assert result["winner"]["project_no"] == "5"


def test_compute_roster_answer_picks_min_for_fewest_phrasing(monkeypatch):
    fewest_question = "Project core nào có ít thành viên hiện tại nhất, không tính số SV mới?"
    projects = [
        {"project_no": "1", "title": "Data Engineering Research", "is_core": True, "members": ["A", "B", "C"]},
        {"project_no": "2", "title": "Data Engineering R&D", "is_core": True, "members": ["D", "E"]},
        {
            "project_no": "5",
            "title": "Platform AXIOM",
            "is_core": True,
            "members": ["H", "I", "J", "K", "L", "M", "N"],
        },
    ]
    monkeypatch.setattr(table_module, "has_llm", lambda: True)
    monkeypatch.setattr(table_module, "extract_candidate_text", lambda candidate: "fake pdf text")
    monkeypatch.setattr(
        table_module,
        "_call_roster_extract",
        lambda text, *, model: [dict(p, _current_count=len(p["members"])) for p in projects],
    )

    result = table_module.compute_roster_answer(
        _profile(fewest_question), [_candidate("axiom.pdf")], config=_config()
    )
    assert result is not None
    assert result["direction"] == "min"
    assert result["answer"] == "2"
    assert result["winner"]["project_no"] == "2"


def test_compute_roster_answer_returns_none_for_unrelated_question(monkeypatch):
    monkeypatch.setattr(table_module, "has_llm", lambda: True)
    result = table_module.compute_roster_answer(
        _profile("What is the average balance in Credit.csv?"),
        [_candidate("axiom.pdf")],
        config=_config(),
    )
    assert result is None


def test_compute_roster_answer_returns_none_when_extraction_fails(monkeypatch):
    monkeypatch.setattr(table_module, "has_llm", lambda: True)
    monkeypatch.setattr(table_module, "extract_candidate_text", lambda candidate: "fake pdf text")

    def raise_error(text, *, model):
        raise RuntimeError("LLM error")

    monkeypatch.setattr(table_module, "_call_roster_extract", raise_error)

    result = table_module.compute_roster_answer(
        _profile(Q8_QUESTION), [_candidate("axiom.pdf")], config=_config()
    )
    assert result is None


def test_call_roster_extract_excludes_new_members_from_current_count(monkeypatch):
    # The model must not be trusted to subtract "new members" itself - code
    # counts len(members) and the prompt/schema keep "new_member_count"
    # entirely separate so it can never be folded into the current count.
    monkeypatch.setattr(
        table_module,
        "call_llm",
        lambda prompt, *, model, temperature, system: (
            '{"projects": [{"project_no": "4", "title": "X", "is_core": true, '
            '"members": ["A", "B", "C", "D"], "new_member_count": 2}]}'
        ),
    )
    projects = table_module._call_roster_extract("fake text", model="test-model")
    assert len(projects) == 1
    assert projects[0]["_current_count"] == 4  # excludes the 2 new members entirely
