"""Shared data models for approach 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QuestionProfile:
    """Buoc 1 output: routing profile for one question."""

    question_id: Any
    question: str
    answer_type: str = ""
    language: str = "en"
    modality_hint: str = "auto"
    keywords: list[str] = field(default_factory=list)
    quoted_phrases: list[str] = field(default_factory=list)
    explicit_file_hints: list[str] = field(default_factory=list)
    wildcard_patterns: list[str] = field(default_factory=list)
    requires_computation: bool = False
    needs_multiple_sources: bool = False
    format_instructions: dict[str, Any] = field(default_factory=dict)
    analysis_source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def search_query(self) -> str:
        return " ".join([self.question, *self.keywords])


@dataclass(slots=True)
class ContextBlock:
    """Buoc 3 output: one evidence-ready text block from a file."""

    relative_path: str
    modality: str
    text: str
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self, limit: int = 9000) -> str:
        body = self.text.strip()
        if len(body) > limit:
            body = body[: limit - 3] + "..."
        return f"[File: {self.relative_path}] (modality: {self.modality})\n{body}"


@dataclass(slots=True)
class AnswerResult:
    """Buoc 4 output: final answer plus evidences and debug info."""

    answer: str
    evidences: list[str]
    strategy: str
    debug: dict[str, Any] = field(default_factory=dict)
