from src.formatter import exact_match, normalize_answer


def test_numeric_rounding_two_decimals():
    answer = normalize_answer("0.86421", question="rounded to two decimal places")
    assert answer == "0.86"


def test_yes_no_normalization():
    assert normalize_answer(" yes ") == "Yes"
    assert normalize_answer("No.") == "No"


def test_multiple_choice_normalization():
    question = "Pick one\nA. 7.45\nB. 7.50\nC. 7.55\nD. 7.60"
    assert normalize_answer("The answer is C", question=question) == "C"


def test_exact_match_normalizes_space_and_case():
    assert exact_match("  Hello   World ", "hello world")
