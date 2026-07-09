import pandas as pd

from approaches.approach_3_agentic_rag.shared_src.file_readers import _restore_headerless_list_row
from approaches.approach_3_agentic_rag.shared_src.formatter import normalize_answer, strip_markdown


# --- Q1: headerless single-column list sheet ---

def test_headerless_gene_list_recovers_lost_row():
    # A gene list read with header=0 ate "BRD8" as the column name -> 3 rows.
    df = pd.DataFrame({"BRD8": ["DHX15", "SSB", "FUS"]})
    fixed = _restore_headerless_list_row(df)
    assert len(fixed) == 4
    assert fixed["value"].tolist() == ["BRD8", "DHX15", "SSB", "FUS"]


def test_multicolumn_sheet_untouched():
    df = pd.DataFrame({"idx": [1, 2], "class": ["a", "b"]})
    assert _restore_headerless_list_row(df).equals(df)


def test_real_label_header_untouched():
    # "Gene" is a label word, no digit -> keep as header.
    df = pd.DataFrame({"Gene": ["BRD8", "DHX15"]})
    assert _restore_headerless_list_row(df).equals(df)


def test_header_without_digit_untouched():
    # "SSB" is a plausible header-or-value but has no digit -> stay safe, keep it.
    df = pd.DataFrame({"SSB": ["FUS", "TP53"]})
    assert _restore_headerless_list_row(df).equals(df)


def test_sentence_column_untouched():
    # Values are sentences, not codes -> not a code list, keep header.
    df = pd.DataFrame({"Row1": ["this is a full sentence", "another sentence here"]})
    assert _restore_headerless_list_row(df).equals(df)


# --- Q11: markdown stripping ---

def test_strip_markdown_image_syntax():
    out = strip_markdown("See here: ![Ảnh iSE](members.png)")
    assert out == "See here: Ảnh iSE (members.png)"


def test_strip_markdown_link_syntax():
    assert strip_markdown("[the file](data/x.csv)") == "the file (data/x.csv)"


def test_strip_markdown_plain_text_unchanged():
    assert strip_markdown("SHINNYO") == "SHINNYO"


def test_normalize_answer_flattens_markdown():
    raw = "Ảnh ở đây: ![members](definitely-not-ise-members-image.png)"
    out = normalize_answer(raw, question="Cho xem ảnh iSE", answer_type="llm_judge")
    assert "![" not in out and "](" not in out
    assert "definitely-not-ise-members-image.png" in out
