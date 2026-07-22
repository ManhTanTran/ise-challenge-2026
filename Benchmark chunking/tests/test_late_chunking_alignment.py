import numpy as np

from benchmark_chunking.tools.late_chunking import _document_token_positions


def test_document_token_positions_ignores_special_prefix_and_suffix() -> None:
    # Simulates BGE-M3/XLM-R special tokens around the document sequence.
    positions = _document_token_positions([0, 101, 102, 103, 2], [101, 102, 103])
    assert positions.tolist() == [1, 2, 3]


def test_document_token_positions_rejects_ambiguous_alignment() -> None:
    try:
        _document_token_positions([0, 7, 2, 7, 2], [7])
    except RuntimeError as exc:
        assert "uniquely align" in str(exc)
    else:
        raise AssertionError("ambiguous document sequence should fail")
