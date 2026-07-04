from pathlib import Path

import pandas as pd

from src import solvers
from src.analyze_errors import evidence_hits_expected
from src.solvers import (
    _count_images_by_question,
    _solve_biomedical_hyperactivated,
    _solve_context_deterministic,
    _solve_from_tables,
    _solve_scholarship_image,
)


def test_acetylproteomics_count_uses_matching_gene_sheet():
    tables = {
        "README": pd.DataFrame({"Sheet": list("ABCDEF"), "Description": list("abcdef")}),
        "D-SE-acetyl": pd.DataFrame({"BRD8": [f"GENE{i}" for i in range(15)]}),
    }

    answer = _solve_from_tables("How many are the significant genes by acetylproteomics?", tables)

    assert answer == "16"


def test_biomedical_hyperactivated_join(tmp_path: Path):
    meta_path = tmp_path / "mmc1.xlsx"
    hyper_path = tmp_path / "hyperactivated.csv"
    drugs_path = tmp_path / "mmc6.xlsx"

    pd.DataFrame(
        {
            "idx": ["S001", "S018", "S031"],
            "Histologic_type": ["Serous", "Endometrioid", "Endometrioid"],
            "CNV_class": ["CNV_HIGH", "CNV_HIGH", "CNV_HIGH"],
        }
    ).to_excel(meta_path, index=False)
    pd.DataFrame(
        {
            "sample_id": ["S001", "S018", "S031"],
            "protein": ["CDK7", "CDK12", "SMARCA4"],
        }
    ).to_csv(hyper_path, index=False)
    with pd.ExcelWriter(drugs_path) as writer:
        pd.DataFrame(
            {
                "gene_name": ["CDK12", "SMARCA4"],
                "drug_name": ["OLAPARIB", "CISPLATIN"],
            }
        ).to_excel(writer, sheet_name="G-FDA approved drugs", index=False)

    answer, evidences = _solve_biomedical_hyperactivated(
        "Which protein sites are found to be hyperactivated in CNV-high endometroid samples and are targeted by FDA-approved drugs?",
        [
            {"absolute_path": str(meta_path), "relative_path": "mmc1.xlsx"},
            {"absolute_path": str(hyper_path), "relative_path": "hyperactivated.csv"},
            {"absolute_path": str(drugs_path), "relative_path": "mmc6.xlsx"},
        ],
    )

    assert answer == "CDK12 and SMARCA4"
    assert evidences == ["mmc1.xlsx", "hyperactivated.csv", "mmc6.xlsx"]


def test_project_member_count_returns_first_number_for_combined_project():
    content = """
Project 5: Platform AXIOM
• Members: Hieu + Thuy, Vinh, Vu, Viet, Giap, Luu + 1 SV moi
• Objective: Build platform

Project 7+8: So hoa + SiFLEX
• Thanh vien: Thuy, Duy Quan, Minh Quan, Le Anh Duy, Tuan Anh + 1 SV moi
• Muc tieu: Build digitization pipeline
"""

    answer, evidences = _solve_context_deterministic(
        "Project core nao cua AXIOM - iSE co nhieu thanh vien hien tai nhat, khong tinh so SV moi?",
        [({"relative_path": "iSE-AXIOM-Internal Intro.pdf"}, content)],
    )

    assert answer == "7"
    assert evidences == ["iSE-AXIOM-Internal Intro.pdf"]


def test_shared_impact_synthesis_requires_three_sources():
    records = [
        ({"relative_path": "01_smart_library_renovation.txt"}, "smart library technology volunteers"),
        ({"relative_path": "02_river_cleanup_community_project.txt"}, "Minh Hoa river cleanup reporting community"),
        ({"relative_path": "04_ai_customer_support_startup.txt"}, "NovaCare customer support humans update knowledge base"),
    ]

    answer, evidences = _solve_context_deterministic(
        "Diem chung chinh trong cach cac du an thu vien thong minh, lam sach song Minh Hoa va NovaCare tao ra tac dong ben vung la gi?",
        records,
    )

    assert "con người" in answer
    assert set(evidences) == {record[0]["relative_path"] for record in records}


def test_digit_count_uses_cached_vision_attributes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(solvers, "VISION_CACHE_DIR", tmp_path / "vision_cache")
    monkeypatch.setattr(solvers, "has_llm", lambda: True)

    image_one = tmp_path / "one.png"
    image_two = tmp_path / "two.png"
    image_one.write_bytes(b"not an actual image")
    image_two.write_bytes(b"not an actual image")

    def fake_answer(prompt, path):
        if Path(path).name == "one.png":
            return '{"digit_count": 1, "digit_values": ["7"], "has_blue_digit": true, "confidence": 1}'
        return '{"digit_count": 2, "digit_values": ["1", "2"], "has_blue_digit": false, "confidence": 1}'

    monkeypatch.setattr(solvers, "answer_image_from_file", fake_answer)
    images = [
        {"absolute_path": str(image_one), "relative_path": "number_image/one.png", "modality": "image"},
        {"absolute_path": str(image_two), "relative_path": "number_image/two.png", "modality": "image"},
    ]

    assert _count_images_by_question('How many images in "number_image" contain exactly one digit?', images) == 1
    assert _count_images_by_question('How many images in "number_image" contain a blue digit?', images) == 1


def test_scholarship_solver_uses_vision_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(solvers, "VISION_CACHE_DIR", tmp_path / "vision_cache")
    monkeypatch.setattr(solvers, "has_llm", lambda: True)
    image = tmp_path / "scholarship.png"
    image.write_bytes(b"not an actual image")

    monkeypatch.setattr(
        solvers,
        "answer_image_from_file",
        lambda prompt, path: '[{"scholarship_name": "LOTTE", "slot_count": 4}, {"scholarship_name": "SHINNYO", "slot_count": 8}]',
    )

    answer, evidences = _solve_scholarship_image(
        "Hoc bong nao co so luong suat trao nhieu nhat?",
        [{"absolute_path": str(image), "relative_path": "scholarship.png", "modality": "image"}],
    )

    assert answer == "SHINNYO"
    assert evidences == ["scholarship.png"]


def test_evidence_match_normalizes_unicode_and_spaces():
    evidence = "KTCT/2NewCh5KTTTrXHCNsửa.ppt"
    expected = "KTCT/2NewCh5 KTTTrXHCNsửa.ppt"

    assert evidence_hits_expected([evidence], [expected]) is True
