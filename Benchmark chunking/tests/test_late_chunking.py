import numpy as np

from approaches.approach_3_agentic_rag.tools.late_chunking import TextSpan, l2_normalize


def test_l2_normalize_keeps_rows_unit_length() -> None:
    vectors = l2_normalize(np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(vectors[0], [0.6, 0.8])
    assert np.allclose(vectors[1], [0.0, 0.0])


def test_span_is_half_open() -> None:
    span = TextSpan(2, 5)
    assert span.start == 2
    assert span.end == 5
