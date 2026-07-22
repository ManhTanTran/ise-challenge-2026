"""Approach 3 configuration: pipeline knobs and model selection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

APPROACH_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APPROACH_ROOT / "outputs"

# Embedding defaults to local sentence-transformers/fastembed/TF-IDF. Set
# ISE_EMBEDDING_PROVIDER=openrouter to use OpenRouter's embeddings endpoint.
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "ISE_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# OpenRouter models, chosen for a ~$20 budget (slugs verified live on
# openrouter.ai, 2026-07): Gemini 2.5 Flash is cheap (~$0.30/M in), has strong
# reasoning + native vision + huge context; the lite variant (~$0.10/M) handles
# the cheap high-volume question-analysis classification. Override via .env.
DEFAULT_ANSWER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
DEFAULT_ANALYSIS_MODEL = os.getenv("ISE_ANALYSIS_MODEL", "google/gemini-2.5-flash-lite")
DEFAULT_VISION_MODEL = os.getenv("ISE_VISION_MODEL", DEFAULT_ANSWER_MODEL)


@dataclass(slots=True)
class Approach3Config:
    """Tunable settings for every pipeline step."""

    # Buoc 0: indexing
    chunk_chars: int = 2200
    chunk_overlap: int = 250
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # Buoc 1: question analysis
    use_llm_analysis: bool = True
    analysis_model: str = DEFAULT_ANALYSIS_MODEL

    # Buoc 2: hybrid retrieval
    # 0.5/0.5 measured best on the sample: semantic embeddings recover paraphrase
    # questions (AXIOM pdf, cross-file projects) while equal BM25 keeps exact
    # keyword/filename hits (class_grades.sql) from being diluted.
    top_k: int = 8
    # needs_multiple_sources questions scan every chunk and return this many
    # files, so a topically-relevant but semantically-distant doc isn't dropped.
    cross_file_top_k: int = 14
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    min_relevance: float = 0.05
    modality_boost: float = 0.15

    # Buoc 3: readers
    max_context_files: int = 8
    # "How many images match number_image/*" needs every matched file read, not
    # just the first 8 - hybrid.retrieve already returns the full pattern set
    # for these questions, so the dispatcher must not re-truncate it back down.
    max_context_files_wildcard: int = 24
    max_chars_per_file: int = 12000
    table_sample_rows: int = 8
    use_table_compute: bool = True
    use_vision: bool = True
    vision_model: str = DEFAULT_VISION_MODEL
    max_vision_files: int = 20
    # "How many images contain X" questions: rewrite to a per-image Yes/No
    # question and count matches in code, instead of asking each image the
    # whole-folder question (which it cannot meaningfully answer) and having
    # Buoc 4 tally free text. Mirrors use_table_compute's code-first counting.
    use_vision_count_compute: bool = True

    # Buoc 4: reasoning
    use_llm: bool = True
    answer_model: str = DEFAULT_ANSWER_MODEL
    max_context_chars: int = 65000

    extra: dict = field(default_factory=dict)


def get_config() -> Approach3Config:
    """Return a fresh config with environment defaults applied."""

    return Approach3Config()
