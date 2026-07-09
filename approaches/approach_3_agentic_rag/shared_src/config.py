"""Configuration defaults for the iSE Challenge pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_LAKE_DIR = Path(os.getenv("DATA_LAKE_DIR", PROJECT_ROOT / "Data-Lake"))
QUESTION_PATH = Path(os.getenv("QUESTION_PATH", PROJECT_ROOT / "0.Sample_Data.xlsx"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
SUBMISSION_PATH = Path(os.getenv("SUBMISSION_PATH", OUTPUT_DIR / "submission.csv"))
FILE_INDEX_PATH = Path(os.getenv("FILE_INDEX_PATH", OUTPUT_DIR / "file_index.json"))
EXTRACTED_TEXT_DIR = OUTPUT_DIR / "extracted_texts"
VISION_CACHE_DIR = Path(os.getenv("ISE_VISION_CACHE_DIR", OUTPUT_DIR / "vision_cache"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")


@dataclass(slots=True)
class PipelineConfig:
    """Runtime configuration with CLI-overridable paths."""

    data_lake_dir: Path = DATA_LAKE_DIR
    question_path: Path = QUESTION_PATH
    output_dir: Path = OUTPUT_DIR
    submission_path: Path = SUBMISSION_PATH
    file_index_path: Path = FILE_INDEX_PATH
    extracted_text_dir: Path = EXTRACTED_TEXT_DIR
    vision_cache_dir: Path = VISION_CACHE_DIR
    openrouter_api_key: str = OPENROUTER_API_KEY
    openrouter_model: str = OPENROUTER_MODEL

    def with_overrides(
        self,
        *,
        data_lake_dir: str | Path | None = None,
        question_path: str | Path | None = None,
        output_path: str | Path | None = None,
        file_index_path: str | Path | None = None,
    ) -> "PipelineConfig":
        """Return a copy with explicit CLI path overrides."""

        output_dir = self.output_dir
        submission_path = self.submission_path
        if output_path is not None:
            submission_path = Path(output_path).expanduser()
            output_dir = submission_path.parent

        resolved_index_path = self.file_index_path
        if file_index_path is not None:
            resolved_index_path = Path(file_index_path).expanduser()
        elif output_path is not None:
            resolved_index_path = output_dir / "file_index.json"

        return PipelineConfig(
            data_lake_dir=Path(data_lake_dir).expanduser()
            if data_lake_dir is not None
            else self.data_lake_dir,
            question_path=Path(question_path).expanduser()
            if question_path is not None
            else self.question_path,
            output_dir=output_dir,
            submission_path=submission_path,
            file_index_path=resolved_index_path,
            extracted_text_dir=output_dir / "extracted_texts",
            vision_cache_dir=Path(os.getenv("ISE_VISION_CACHE_DIR", output_dir / "vision_cache")),
            openrouter_api_key=self.openrouter_api_key,
            openrouter_model=self.openrouter_model,
        )


def get_config() -> PipelineConfig:
    """Return a fresh config object."""

    return PipelineConfig()
