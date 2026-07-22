"""Token-level long-context embedding utilities for the Phase 2 benchmark.

This module deliberately does not use the OpenRouter embedding endpoint.  Late
Chunking needs the contextual vector for every token (``last_hidden_state``),
whereas that endpoint exposes only the final pooled vector for an input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A half-open character span in the normalized source document."""

    start: int
    end: int


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _document_token_positions(model_token_ids: Iterable[int], document_token_ids: Iterable[int]) -> np.ndarray:
    """Locate the supplied document tokens inside a model-ready sequence.

    Fast tokenizers cannot create ``special_tokens_mask`` through
    ``prepare_for_model`` on recent Transformers versions.  The supplied
    document ids are, however, retained as one contiguous subsequence between
    whatever prefix/suffix special tokens the tokenizer adds.
    """

    model_ids = list(model_token_ids)
    document_ids = list(document_token_ids)
    if not document_ids:
        return np.asarray([], dtype=np.int64)
    width = len(document_ids)
    matches = [
        start
        for start in range(len(model_ids) - width + 1)
        if model_ids[start : start + width] == document_ids
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely align document tokens inside the model-ready sequence "
            f"(matches={len(matches)}, document_tokens={width})."
        )
    return np.arange(matches[0], matches[0] + width, dtype=np.int64)


class LongContextTokenEmbedder:
    """Expose contextual token vectors and span pooling from a HF encoder.

    Text longer than ``window_tokens`` is encoded in overlapping macro windows.
    A token occurring in two windows receives the mean of both contextual
    representations; the overlap therefore preserves local context at a macro
    boundary without creating duplicate retrieval chunks.
    """

    kind = "local-token-level"

    def __init__(
        self,
        model_name: str,
        *,
        window_tokens: int = 2048,
        overlap_tokens: int = 256,
        device: str = "auto",
    ) -> None:
        if window_tokens < 32:
            raise ValueError("window_tokens must be at least 32")
        if not 0 <= overlap_tokens < window_tokens:
            raise ValueError("overlap_tokens must be in [0, window_tokens)")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised at runtime
            raise RuntimeError(
                "Late Chunking requires transformers. Install with: "
                "python -m pip install 'transformers>=4.45' sentencepiece"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("Late Chunking requires a fast tokenizer for offset mappings.")
        self.model = AutoModel.from_pretrained(model_name)
        self.device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.model.to(self.device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)
        limits = [
            int(value)
            for value in (
                getattr(self.model.config, "max_position_embeddings", None),
                getattr(self.tokenizer, "model_max_length", None),
            )
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        self.max_model_tokens = min(limits) if limits else None
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        if self.max_model_tokens is not None:
            max_content_tokens = self.max_model_tokens - special_count
            if max_content_tokens < 32:
                raise RuntimeError("Model context is too small after reserving special tokens.")
            self.window_tokens = min(window_tokens, max_content_tokens)

    def encode_query(
        self, texts: Iterable[str], *, max_tokens: int = 512, batch_size: int = 16
    ) -> np.ndarray:
        """Mean-pool query token states, matching span pooling on documents."""

        text_list = [str(text) for text in texts]
        vectors: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(text_list), batch_size):
                batch = self.tokenizer(
                    text_list[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_tokens,
                    return_tensors="pt",
                    return_special_tokens_mask=True,
                )
                special = batch.pop("special_tokens_mask").bool()
                attention = batch["attention_mask"].bool()
                batch = {key: value.to(self.device) for key, value in batch.items()}
                hidden = self.model(**batch).last_hidden_state.detach().cpu().numpy()
                mask = (attention & ~special).numpy()
                for row, row_mask in zip(hidden, mask):
                    vectors.append(row[row_mask].mean(axis=0) if row_mask.any() else row[0])
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Naive control: encode each text independently with identical pooling."""

        return self.encode_query(texts, max_tokens=self.window_tokens)

    def contextualize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(offset_mapping, token_vectors)`` for all non-special tokens."""

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            truncation=False,
        )
        token_ids = encoded["input_ids"]
        offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
        if not token_ids:
            return offsets.reshape(0, 2), np.zeros((0, self.dimension), dtype=np.float32)

        sums = np.zeros((len(token_ids), self.dimension), dtype=np.float32)
        counts = np.zeros(len(token_ids), dtype=np.int32)
        stride = self.window_tokens - self.overlap_tokens
        starts = [0]
        while starts[-1] + self.window_tokens < len(token_ids):
            next_start = min(starts[-1] + stride, len(token_ids) - self.window_tokens)
            if next_start == starts[-1]:
                break
            starts.append(next_start)

        with self._torch.inference_mode():
            for start in starts:
                stop = min(len(token_ids), start + self.window_tokens)
                prepared = self.tokenizer.prepare_for_model(
                    token_ids[start:stop],
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                if prepared["input_ids"].ndim == 1:
                    prepared = {
                        key: value.unsqueeze(0) for key, value in prepared.items()
                    }
                if (
                    self.max_model_tokens is not None
                    and prepared["input_ids"].shape[-1] > self.max_model_tokens
                ):
                    raise RuntimeError(
                        "Prepared macro-window exceeds model context: "
                        f"{prepared['input_ids'].shape[-1]} > {self.max_model_tokens}."
                    )
                positions = _document_token_positions(
                    prepared["input_ids"][0].tolist(), token_ids[start:stop]
                )
                prepared = {key: value.to(self.device) for key, value in prepared.items()}
                hidden = self.model(**prepared).last_hidden_state[0].detach().cpu().numpy()
                token_hidden = hidden[positions]
                expected = stop - start
                if len(token_hidden) != expected:
                    raise RuntimeError(
                        "Tokenizer/model content-token alignment failed: "
                        f"expected {expected} document tokens, got {len(token_hidden)}."
                    )
                sums[start:stop] += token_hidden
                counts[start:stop] += 1
                if stop == len(token_ids):
                    break
        if (counts == 0).any():
            raise RuntimeError("A macro-window left token embeddings uncovered.")
        return offsets, sums / counts[:, None]

    def pool_spans(
        self, offsets: np.ndarray, token_vectors: np.ndarray, spans: Iterable[TextSpan]
    ) -> np.ndarray:
        """Mean-pool tokens overlapping each character span, then L2-normalize."""

        vectors: list[np.ndarray] = []
        for span in spans:
            mask = (offsets[:, 1] > span.start) & (offsets[:, 0] < span.end)
            if not mask.any():
                raise ValueError(f"No token overlaps span [{span.start}, {span.end}).")
            vectors.append(token_vectors[mask].mean(axis=0))
        return l2_normalize(np.asarray(vectors, dtype=np.float32))
