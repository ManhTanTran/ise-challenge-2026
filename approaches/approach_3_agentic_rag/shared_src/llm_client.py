"""OpenRouter client helpers for reranking and context QA."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from .config import OPENROUTER_MODEL
from .utils import NOT_ENOUGH_DATA, normalize_spaces, truncate_text

LOGGER = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def has_llm() -> bool:
    """Return True when OpenRouter credentials are available."""

    return bool(os.getenv("OPENROUTER_API_KEY"))


def call_llm(
    prompt: str,
    model: str | None = None,
    temperature: float = 0,
    *,
    system: str | None = None,
) -> str:
    """Call OpenRouter Chat Completions with deterministic defaults."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    selected_model = model or os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL)
    system_prompt = system or (
        "You answer questions using only the provided context. "
        f"If the context is insufficient, answer exactly: {NOT_ENOUGH_DATA}"
    )

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def answer_from_context(question: str, context: str, answer_type: str | None = None) -> str:
    """Answer a question using only supplied text context."""

    prompt = f"""
Use only the context below to answer the question.

Rules:
- If the context is insufficient, answer exactly: {NOT_ENOUGH_DATA}
- For exact-match questions, return only the final answer.
- For multiple-choice questions, return only A, B, C, or D.
- For Yes/No questions, return only Yes or No.
- For numeric questions, return only the number unless a unit is required.

Answer type: {answer_type or "unknown"}

Question:
{question}

Context:
{truncate_text(context, 60000)}
""".strip()
    return normalize_spaces(call_llm(prompt))


def rerank_files(question: str, candidates: list[dict[str, Any]]) -> list[str]:
    """Ask the LLM to reorder likely evidence files and return paths."""

    if not candidates:
        return []
    compact = [
        {
            "relative_path": item.get("relative_path"),
            "modality": item.get("modality"),
            "preview": truncate_text(item.get("text_preview", ""), 500),
        }
        for item in candidates[:20]
    ]
    prompt = f"""
Choose the most likely evidence files for the question using only names and previews.
Return a JSON array of relative_path strings, most relevant first.

Question:
{question}

Candidates:
{json.dumps(compact, ensure_ascii=False)}
""".strip()
    try:
        raw = call_llm(prompt)
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []
        parsed = json.loads(match.group(0))
        return [str(item) for item in parsed if isinstance(item, str)]
    except Exception as exc:
        LOGGER.warning("LLM reranking failed: %s", exc)
        return []


def generate_pandas_plan(question: str, table_schema: dict[str, Any]) -> str:
    """Generate a concise pandas plan for ambiguous table questions."""

    prompt = f"""
Given this table schema, produce a short deterministic pandas plan.
Do not compute the answer. Do not use external knowledge.
Return only JSON with keys: operation, columns, filters, rounding, output_format.

Question:
{question}

Schema:
{json.dumps(table_schema, ensure_ascii=False)}
""".strip()
    return call_llm(prompt)


def answer_image_from_file(question: str, image_path: str | Path, context: str = "") -> str:
    """Ask a vision-capable OpenRouter model to answer about one image."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    selected_model = os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL)
    path = Path(image_path)
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the image and provided context. "
                    f"If insufficient, answer exactly: {NOT_ENOUGH_DATA}"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Question: {question}\nContext: {context}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            },
        ],
        temperature=0,
    )
    return normalize_spaces(response.choices[0].message.content or "")


def transcribe_audio_file(audio_path: str | Path, model: str | None = None) -> str:
    """Transcribe audio through OpenRouter's audio transcription endpoint."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    selected_model = model or os.getenv("ISE_TRANSCRIPTION_MODEL", "openai/whisper-1")

    from openai import OpenAI

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    path = Path(audio_path)
    with path.open("rb") as handle:
        response = client.audio.transcriptions.create(
            model=selected_model,
            file=handle,
        )
    text = getattr(response, "text", "")
    if not text and isinstance(response, dict):
        text = str(response.get("text", ""))
    return normalize_spaces(text)
