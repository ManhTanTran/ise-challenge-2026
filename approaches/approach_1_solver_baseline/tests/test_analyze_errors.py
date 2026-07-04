import json
from pathlib import Path

import pandas as pd

from src.analyze_errors import analyze_outputs, parse_evidences


def test_parse_evidences_json_string():
    assert parse_evidences('["a.csv", "b.pdf"]') == ["a.csv", "b.pdf"]


def test_analyze_outputs_writes_report(tmp_path: Path):
    error_path = tmp_path / "error_analysis.csv"
    index_path = tmp_path / "file_index.json"
    output_dir = tmp_path / "analysis"

    pd.DataFrame(
        [
            {
                "id": 1,
                "question": "How many rows?",
                "predicted_answer": "3",
                "groundtruth": "4",
                "answer_type": "exact_match",
                "evidences": json.dumps(["a.csv"]),
                "is_correct": False,
                "error_type": "exact_mismatch",
            }
        ]
    ).to_csv(error_path, index=False)
    index_path.write_text(
        json.dumps([{"relative_path": "a.csv", "modality": "table"}]),
        encoding="utf-8",
    )

    result = analyze_outputs(
        error_analysis_path=error_path,
        output_dir=output_dir,
        file_index_path=index_path,
    )

    assert Path(result["report_path"]).exists()
    assert (output_dir / "exact_mismatches.csv").exists()
    enriched = pd.read_csv(output_dir / "error_analysis_enriched.csv")
    assert enriched.loc[0, "primary_modality"] == "table"
