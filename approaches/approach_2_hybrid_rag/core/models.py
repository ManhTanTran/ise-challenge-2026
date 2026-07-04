"""Shared data models for approach 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class QuestionProfile:
    """Compact routing profile produced before retrieval."""

    question_id: Any
    question: str
    answer_type: str = ""
    language: str = "en"
    modality_hint: str = "auto"
    keywords: list[str] = field(default_factory=list)
    quoted_phrases: list[str] = field(default_factory=list)
    explicit_file_hints: list[str] = field(default_factory=list)
    requires_computation: bool = False
    needs_multiple_sources: bool = False
    format_instructions: dict[str, Any] = field(default_factory=dict)
    expected_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ContextItem:
    """Text context prepared from one retrieved file."""

    relative_path: str
    modality: str
    text: str
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_block(self, limit: int = 8000) -> str:
        body = self.text.strip()
        if len(body) > limit:
            body = body[: limit - 3] + "..."
        return f"Source: {self.relative_path}\nModality: {self.modality}\n{body}"


@dataclass(slots=True)
class AnswerResult:
    """Final answer plus evidence paths and debug metadata."""

    answer: str
    evidences: list[str]
    strategy: str
    debug: dict[str, Any] = field(default_factory=dict)
