from src.retriever import retrieve_files


def _item(path: str, modality: str = "document") -> dict:
    extension = "." + path.rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
    return {
        "relative_path": path,
        "filename": path.rsplit("/", 1)[-1],
        "extension": extension,
        "modality": modality,
        "text_preview": "",
    }


def _paths(candidates: list[dict]) -> list[str]:
    return [item["relative_path"] for item in candidates]


def test_retrieve_quoted_folder_returns_all_files_without_topk_truncation():
    index = [_item(f"number_image/{i}.png", "image") for i in range(12)]
    index.extend([_item("unrelated.txt"), _item("other.png", "image")])

    candidates = retrieve_files(
        'How many images in "number_image" contain exactly one digit?',
        index,
        top_k=8,
    )

    assert len([path for path in _paths(candidates) if path.startswith("number_image/")]) == 12
    assert all(path.startswith("number_image/") for path in _paths(candidates))


def test_retrieve_scholarship_alias():
    index = [_item("scholarship1.png", "image"), _item("KTCT/GIAO-TRINH-KHONG-CHUYEN.pdf")]

    candidates = retrieve_files(
        "Hoc bong ngoai ngan sach nao co so luong suat trao nhieu nhat?",
        index,
    )

    assert _paths(candidates)[0] == "scholarship1.png"


def test_retrieve_ise_member_image_alias():
    index = [
        _item("ise.md"),
        _item("definitely-100-percent-not-ise-members-image.png", "image"),
        _item("KTCT/TONG-HOP-QUIZ.docx"),
    ]

    candidates = retrieve_files(
        "Cho toi xem anh cac thanh vien cua nhom iSE di hihihi",
        index,
    )

    assert _paths(candidates)[:2] == [
        "definitely-100-percent-not-ise-members-image.png",
        "ise.md",
    ]


def test_retrieve_axiom_alias():
    index = [_item("iSE-AXIOM-Internal Intro.pdf"), _item("ise.md")]

    candidates = retrieve_files(
        "Project core nao cua AXIOM - iSE co nhieu thanh vien hien tai nhat, khong tinh so SV moi?",
        index,
    )

    assert _paths(candidates)[0] == "iSE-AXIOM-Internal Intro.pdf"


def test_retrieve_sql_grade_alias():
    index = [_item("class_grades.sql", "table"), _item("KTCT/TONG-HOP-QUIZ.docx")]

    candidates = retrieve_files(
        "Diem trung binh mon Toan cua lop 10A1 la bao nhieu? A. 7.45 B. 7.50 C. 7.55 D. 7.60",
        index,
    )

    assert _paths(candidates)[0] == "class_grades.sql"


def test_retrieve_ktct_market_economy_alias():
    index = [
        _item("KTCT/2NewCh5KTTTrXHCNsua.ppt"),
        _item("topic_16_page-0005.jpg", "image"),
    ]

    candidates = retrieve_files(
        "Doi voi mon hoc Kinh te chinh tri, nen kinh te thi truong dinh huong XHCN o Viet Nam la gi?",
        index,
    )

    assert _paths(candidates)[0] == "KTCT/2NewCh5KTTTrXHCNsua.ppt"


def test_retrieve_sustainable_project_alias_returns_three_documents():
    index = [
        _item("01_smart_library_renovation.txt"),
        _item("02_river_cleanup_community_project.txt"),
        _item("04_ai_customer_support_startup.txt"),
        _item("topic_16_page-0015.jpg", "image"),
    ]

    candidates = retrieve_files(
        "Diem chung chinh trong cach cac du an thu vien thong minh, lam sach song Minh Hoa va NovaCare tao ra tac dong ben vung la gi?",
        index,
    )

    assert set(_paths(candidates)[:3]) == {
        "01_smart_library_renovation.txt",
        "02_river_cleanup_community_project.txt",
        "04_ai_customer_support_startup.txt",
    }


def test_retrieve_audio_workshop_alias():
    index = [_item("workshop_03.22.m4a", "audio"), _item("01_smart_library_renovation.txt")]

    candidates = retrieve_files(
        "Based on the audio meeting summary for the AI workshop for first-year students on March 22, 2026, what was the total number of workshop participants?",
        index,
    )

    assert _paths(candidates)[0] == "workshop_03.22.m4a"


def test_retrieve_biomedical_alias_returns_all_join_sources():
    index = [
        _item("biomedical/hyperactivated.csv", "table"),
        _item("biomedical/1-s2.0-S0092867420301070-mmc1.xlsx", "table"),
        _item("biomedical/1-s2.0-S0092867420301070-mmc6.xlsx", "table"),
        _item("04_ai_customer_support_startup.txt"),
    ]

    candidates = retrieve_files(
        "Which protein sites are found to be hyperactivated in CNV-high endometroid samples and are targeted by FDA-approved drugs?",
        index,
    )

    assert set(_paths(candidates)[:3]) == {
        "biomedical/hyperactivated.csv",
        "biomedical/1-s2.0-S0092867420301070-mmc1.xlsx",
        "biomedical/1-s2.0-S0092867420301070-mmc6.xlsx",
    }


def test_retrieve_airline_report_alias_returns_page_series():
    index = [_item(f"topic_16_page-{i:04d}.jpg", "image") for i in range(1, 14)]
    index.append(_item("topic_15_page-0001.jpg", "image"))

    candidates = retrieve_files(
        "Cac hang hang khong xuat hien trong bao cao nghien cuu tai chinh nganh hang khong cua Viet Nam la gi?",
        index,
        top_k=8,
    )

    topic_16_paths = [path for path in _paths(candidates) if path.startswith("topic_16_page-")]
    assert len(topic_16_paths) == 13
    assert all(path.startswith("topic_16_page-") for path in _paths(candidates))
