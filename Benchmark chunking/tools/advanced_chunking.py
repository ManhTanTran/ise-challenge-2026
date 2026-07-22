"""Adapters for PIC, official RAPTOR, and precomputed HiChunk boundaries.

All adapters return the same flat chunk-row contract.  Generation is cached
separately from retrieval so every method can be embedded and scored by the
same BGE-M3 model.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .benchmark_late_chunking import _fixed_spans, _sentence_spans
from .late_chunking import LongContextTokenEmbedder, TextSpan


def _guard_spans(text: str, spans: list[TextSpan], max_chars: int) -> list[TextSpan]:
    guarded: list[TextSpan] = []
    for span in spans:
        for child in _fixed_spans(text[span.start : span.end], max_chars, 0):
            guarded.append(TextSpan(span.start + child.start, span.start + child.end))
    return guarded


def pic_spans(
    text: str,
    summary: str,
    embedder: LongContextTokenEmbedder,
    *,
    max_chars: int = 1200,
) -> tuple[list[TextSpan], dict[str, Any]]:
    """Implement the requested PIC rule exactly, followed by a size guard.

    Sentences are labelled by whether cosine(sentence, summary) is at least
    the document mean. Adjacent sentences with the same label are grouped.
    """

    sentences = _sentence_spans(text, max(len(text), 1))
    if not sentences:
        return [], {"sentence_count": 0, "threshold": None, "groups": 0}
    sentence_texts = [text[span.start : span.end] for span in sentences]
    sentence_vectors = embedder.encode_texts(sentence_texts)
    summary_vector = embedder.encode_texts([summary])[0]
    similarities = sentence_vectors @ summary_vector
    threshold = float(np.mean(similarities))
    states = similarities >= threshold
    groups: list[TextSpan] = []
    current = sentences[0]
    current_state = bool(states[0])
    for span, state in zip(sentences[1:], states[1:]):
        if bool(state) == current_state:
            current = TextSpan(current.start, span.end)
        else:
            groups.append(current)
            current = span
            current_state = bool(state)
    groups.append(current)
    return _guard_spans(text, groups, max_chars), {
        "sentence_count": len(sentences),
        "threshold": threshold,
        "groups": len(groups),
        "positive_sentences": int(np.sum(states)),
    }


class JsonSummaryCache:
    """Small resumable cache shared by PIC and RAPTOR LLM summaries."""

    def __init__(self, path: Path, generator: Callable[[str, int], str]):
        self.path = path
        self.generator = generator
        self.values: dict[str, str] = {}
        if path.exists():
            self.values = json.loads(path.read_text(encoding="utf-8"))

    def summarize(self, context: str, max_tokens: int = 300) -> str:
        key = hashlib.sha256(
            f"{max_tokens}\0{context}".encode("utf-8", errors="ignore")
        ).hexdigest()
        if key not in self.values:
            self.values[key] = self.generator(context, max_tokens)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.values, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return self.values[key]


def openai_summary_generator(model: str = "gpt-4o-mini") -> Callable[[str, int], str]:
    """Create the GPT summarizer lazily, so cached/offline runs need no SDK."""

    def generate(context: str, max_tokens: int) -> str:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Add it in Kaggle Secrets before running PIC/RAPTOR.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if api_key.startswith("sk-or-") and not base_url:
            base_url = "https://openrouter.ai/api/v1"
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize faithfully. Preserve names, numbers, dates, headings, and facts useful for retrieval.",
                },
                {"role": "user", "content": context},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        return response.choices[0].message.content or ""

    return generate


def build_raptor_rows(
    text: str,
    *,
    relative_path: str,
    modality: str,
    embedder: LongContextTokenEmbedder,
    summarizer: JsonSummaryCache,
    official_repo: Path,
    max_tokens: int = 100,
    num_layers: int = 3,
    summarization_length: int = 150,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Build one official RAPTOR tree and flatten leaf + summary nodes."""

    repo = str(official_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from raptor.EmbeddingModels import BaseEmbeddingModel
    from raptor.SummarizationModels import BaseSummarizationModel
    from raptor.cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig

    class SharedEmbedding(BaseEmbeddingModel):
        def create_embedding(self, value: str):
            return embedder.encode_texts([value])[0]

    class CachedSummary(BaseSummarizationModel):
        def summarize(self, context: str, max_tokens: int = 150):
            return summarizer.summarize(context, max_tokens)

    config = ClusterTreeConfig(
        max_tokens=max_tokens,
        num_layers=num_layers,
        summarization_length=summarization_length,
        summarization_model=CachedSummary(),
        embedding_models={"shared_bge_m3": SharedEmbedding()},
        cluster_embedding_model="shared_bge_m3",
    )
    tree = ClusterTreeBuilder(config).build_from_text(text, use_multithreading=False)
    level_by_index = {
        node.index: level
        for level, nodes in tree.layer_to_nodes.items()
        for node in nodes
    }
    ordered = sorted(tree.all_nodes.values(), key=lambda node: (level_by_index[node.index], node.index))
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for index, node in enumerate(ordered):
        level = int(level_by_index[node.index])
        rows.append(
            {
                "chunk_id": f"{relative_path}::raptor::{index}",
                "relative_path": relative_path,
                "filename": Path(relative_path).name,
                "extension": Path(relative_path).suffix.lower(),
                "modality": modality,
                "chunk_index": index,
                "text": node.text,
                "char_count": len(node.text),
                "start_char": None,
                "end_char": None,
                "method": "raptor_all_nodes",
                "raptor_level": level,
                "raptor_is_leaf": level == 0,
                "raptor_children": sorted(node.children),
            }
        )
        vectors.append(np.asarray(node.embeddings["shared_bge_m3"], dtype=np.float32))
    return rows, np.asarray(vectors, dtype=np.float32), {
        "nodes": len(rows),
        "leaves": len(tree.leaf_nodes),
        "layers": int(tree.num_layers),
    }


def load_hichunk_splits(path: Path) -> dict[str, list[tuple[str, int]]]:
    """Load HiChunk's documented ``{file: {splits: [[text, level]]}}`` format."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    loaded: dict[str, list[tuple[str, int]]] = {}
    for source, value in raw.items():
        splits = value.get("splits", value) if isinstance(value, dict) else value
        loaded[source.replace("\\", "/").casefold()] = [
            (str(item[0]), int(item[1])) for item in splits if item and str(item[0]).strip()
        ]
    return loaded


def hichunk_rows(
    relative_path: str,
    modality: str,
    split_index: dict[str, list[tuple[str, int]]],
    *,
    lookup_key: str | None = None,
) -> list[dict[str, Any]]:
    key = (lookup_key or relative_path).replace("\\", "/").casefold()
    splits = split_index.get(key)
    if splits is None:
        # Official scripts often use only the basename as the JSON key.
        splits = split_index.get(Path(relative_path).name.casefold(), [])
    return [
        {
            "chunk_id": f"{relative_path}::hichunk::{index}",
            "relative_path": relative_path,
            "filename": Path(relative_path).name,
            "extension": Path(relative_path).suffix.lower(),
            "modality": modality,
            "chunk_index": index,
            "text": text,
            "char_count": len(text),
            "start_char": None,
            "end_char": None,
            "method": "hichunk_flat",
            "hichunk_level": level,
        }
        for index, (text, level) in enumerate(splits)
    ]
