"""Buoc 3: dispatch retrieved candidates to modality readers.

Every candidate becomes one or more ContextBlocks:
- tables get schema + sample rows, plus an exact computed result when the
  question needs arithmetic (LLM writes pandas, we execute it),
- images get their indexed OCR/caption text plus an on-demand vision answer,
- audio uses the transcript cached during indexing (Whisper runs offline),
- documents/text/web reuse the selected chunks or the cached extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import Approach3Config
from ..core.models import ContextBlock, QuestionProfile
from ..retrieval.hybrid import is_folder_count_question
from ..shared_src.file_readers import extract_candidate_text
from ..shared_src.utils import truncate_text
from .image import compute_scholarship_answer, count_matching_images, vision_answer
from .table import compute_roster_answer, compute_table_answer, table_context_text

LOGGER = logging.getLogger(__name__)


def build_context_blocks(
    profile: QuestionProfile,
    candidates: list[dict[str, Any]],
    *,
    config: Approach3Config,
    vision_cache_dir: str | Path,
) -> list[ContextBlock]:
    """Read every candidate with the reader that matches its modality."""

    blocks: list[ContextBlock] = []
    vision_budget = config.max_vision_files

    # Folder/pattern counting questions get their full matching set from
    # retrieval (see hybrid._full_pattern_set), which only returns more than
    # config.top_k candidates when that bypass engaged. Re-truncating here at
    # the small default limit would silently drop files before they're ever
    # read, undercounting answers like "how many images contain a blue digit".
    effective_limit = (
        config.max_context_files_wildcard
        if len(candidates) > config.max_context_files
        else config.max_context_files
    )
    windowed_candidates = candidates[:effective_limit]

    # Cross-file questions can retrieve many candidates (cross_file_top_k=14),
    # and a handful of large but barely-relevant documents (matched by broad
    # BM25/vector overlap) can each claim the full max_chars_per_file budget
    # before a short, genuinely-relevant file lower in the ranking ever gets
    # read - the combined context then exceeds max_context_chars and
    # reasoner.py's final truncate_text silently drops that file entirely
    # (observed: 7 unrelated ~8.8k-char slide decks used 62k of a 65k budget,
    # cutting off a relevant 3.7k-char text file ranked #11). Give every file
    # a fair share of the total budget instead of first-come-first-served.
    per_file_char_budget = config.max_chars_per_file
    if profile.needs_multiple_sources and windowed_candidates:
        fair_share = config.max_context_chars // len(windowed_candidates)
        per_file_char_budget = min(config.max_chars_per_file, max(fair_share, 1000))

    # Table compute runs ONCE across every table candidate in this window (not
    # per-file) so one expression can join data across files - e.g. "proteins
    # filtered by significance in file A that are also targeted by drugs in
    # file B" needs both tables in the same namespace to resolve.
    table_candidates = [c for c in windowed_candidates if c.get("modality") == "table"]
    if profile.requires_computation and table_candidates:
        computed = compute_table_answer(profile, table_candidates, config=config)
        if computed:
            primary = table_candidates[0]
            blocks.append(
                ContextBlock(
                    relative_path=str(primary.get("relative_path", "")),
                    modality="table",
                    text=(
                        "Computed deterministically with pandas across: "
                        f"{', '.join(computed['sources'])}.\n"
                        f"Expression: {computed['expression']}\n"
                        f"Result: {computed['result']}"
                    ),
                    score=float(primary.get("score", 0.0)),
                    reason=str(primary.get("reason", "")),
                    metadata={"reader": "table_compute", "sources": computed["sources"]},
                )
            )

    # Same code-first principle for "how many images ... ?" questions: count
    # matches once in code instead of asking each image the whole-folder
    # question (meaningless to a single image) and trusting Buoc 4 to tally
    # free text across N images.
    image_candidates = [c for c in windowed_candidates if c.get("modality") == "image"]
    # Paths already covered by a code-computed count. Their old per-image
    # "Vision model answer for this image: <noise>" line is suppressed below
    # (see the main loop): with both a correct "Computed by checking each
    # image..." block AND that noisy line present, Buoc 4 was observed
    # re-deriving its own (wrong) tally from the noisy per-image lines instead
    # of trusting the computed count - the fix isn't just adding the right
    # number, it's also removing the contradicting one.
    counted_image_paths: set[str] = set()
    if is_folder_count_question(profile) and image_candidates:
        counted = count_matching_images(
            profile, image_candidates, config=config, cache_dir=vision_cache_dir
        )
        if counted:
            counted_image_paths = {str(c.get("relative_path", "")) for c in image_candidates}
            primary = image_candidates[0]
            matched = ", ".join(counted["matched_files"]) if counted["matched_files"] else "(none)"
            blocks.append(
                ContextBlock(
                    relative_path=str(primary.get("relative_path", "")),
                    modality="image",
                    text=(
                        f'Computed by checking each image individually against: "{counted["criterion"]}"\n'
                        f"Matched {counted['matched_count']} of {counted['total']} images: {matched}\n"
                        "This count is authoritative - do not recount or override it."
                    ),
                    score=float(primary.get("score", 0.0)),
                    reason=str(primary.get("reason", "")),
                    metadata={"reader": "image_count_compute", "matched_files": counted["matched_files"]},
                )
            )

    # Same code-first split for "which scholarship has the most slots?"
    # questions: a single free-text vision answer over a crowded multi-row
    # table is unreliable (the model, or noisy OCR text next to it in
    # context, latches onto the wrong row), so every candidate image is asked
    # to extract its rows as structured JSON and code picks the max.
    if image_candidates:
        scholarship = compute_scholarship_answer(
            profile, image_candidates, config=config, cache_dir=vision_cache_dir
        )
        if scholarship:
            counted_image_paths |= set(scholarship["sources"])
            primary_path = scholarship["sources"][0]
            primary = next(
                (c for c in image_candidates if str(c.get("relative_path", "")) == primary_path),
                image_candidates[0],
            )
            blocks.append(
                ContextBlock(
                    relative_path=primary_path,
                    modality="image",
                    text=(
                        "Computed by extracting every scholarship row from each image and "
                        f"picking the maximum slot count.\nWinner: {scholarship['answer']}\n"
                        "This is authoritative - do not re-derive it from raw OCR/caption text."
                    ),
                    score=float(primary.get("score", 0.0)),
                    reason=str(primary.get("reason", "")),
                    metadata={"reader": "image_scholarship_compute", "sources": scholarship["sources"]},
                )
            )

    # Same code-first split for "which project has the most/fewest current
    # members, excluding new students?" questions: a single free-text read of
    # a multi-project roster is unreliable (wrong project picked, wrong
    # direction answered, or the "new student" count folded in despite the
    # question excluding it), so every candidate document is asked to extract
    # its project rosters as structured JSON and code picks the min/max
    # current-member count.
    document_candidates = [c for c in windowed_candidates if c.get("modality") == "document"]
    if document_candidates:
        roster = compute_roster_answer(profile, document_candidates, config=config)
        if roster:
            primary_path = roster["sources"][0]
            primary = next(
                (c for c in document_candidates if str(c.get("relative_path", "")) == primary_path),
                document_candidates[0],
            )
            extreme = "maximum" if roster["direction"] == "max" else "minimum"
            blocks.append(
                ContextBlock(
                    relative_path=primary_path,
                    modality="document",
                    text=(
                        f"Computed by extracting every project's roster and picking the "
                        f"{extreme} current-member count (new members excluded).\nWinner: {roster['answer']}\n"
                        "This is authoritative - do not re-derive it from raw document text."
                    ),
                    score=float(primary.get("score", 0.0)),
                    reason=str(primary.get("reason", "")),
                    metadata={"reader": "roster_compute", "sources": roster["sources"]},
                )
            )

    for candidate in windowed_candidates:
        modality = str(candidate.get("modality", "unknown"))
        relative_path = str(candidate.get("relative_path", ""))
        base = {
            "relative_path": relative_path,
            "modality": modality,
            "score": float(candidate.get("score", 0.0)),
            "reason": str(candidate.get("reason", "")),
        }

        if modality == "table":
            blocks.extend(_table_blocks(candidate, base, config=config, char_budget=per_file_char_budget))
        elif modality == "image":
            block, used_vision = _image_block(
                profile,
                candidate,
                base,
                config=config,
                vision_cache_dir=vision_cache_dir,
                vision_allowed=vision_budget > 0 and relative_path not in counted_image_paths,
                char_budget=per_file_char_budget,
            )
            if used_vision:
                vision_budget -= 1
            if block:
                blocks.append(block)
        else:
            text = extract_candidate_text(candidate)
            if text:
                blocks.append(
                    ContextBlock(
                        **base,
                        text=truncate_text(text, per_file_char_budget),
                        metadata={"reader": "audio_transcript" if modality == "audio" else "text"},
                    )
                )
    return blocks


def _table_blocks(
    candidate: dict[str, Any],
    base: dict[str, Any],
    *,
    config: Approach3Config,
    char_budget: int,
) -> list[ContextBlock]:
    """Schema + sample rows for one table file (exact computed result, if any,
    is added once for the whole question by build_context_blocks)."""

    blocks: list[ContextBlock] = []
    text = table_context_text(candidate, config=config) or extract_candidate_text(candidate)
    if text:
        blocks.append(
            ContextBlock(
                **base,
                text=truncate_text(text, char_budget),
                metadata={"reader": "table_schema"},
            )
        )
    return blocks


def _image_block(
    profile: QuestionProfile,
    candidate: dict[str, Any],
    base: dict[str, Any],
    *,
    config: Approach3Config,
    vision_cache_dir: str | Path,
    vision_allowed: bool,
    char_budget: int,
) -> tuple[ContextBlock | None, bool]:
    pieces: list[str] = []
    indexed_text = extract_candidate_text(candidate)
    if indexed_text:
        pieces.append(f"Indexed OCR/caption:\n{truncate_text(indexed_text, 4000)}")

    # Not gated on profile.modality_hint: a question can read as a plain
    # analytical/table question by wording (e.g. "which year had the lowest
    # CAPEX?") while its only evidence is a chart rendered as an image with no
    # backing data table - modality_hint misses that entirely. Retrieval
    # already decided this candidate is relevant; trust it and let
    # vision_allowed (budget + not already covered by count/scholarship
    # compute) be the only gate, consistent with count_matching_images and
    # compute_scholarship_answer, neither of which checks modality_hint.
    used_vision = False
    if vision_allowed:
        answer = vision_answer(profile, candidate, config=config, cache_dir=vision_cache_dir)
        if answer:
            pieces.append(f"Vision model answer for this image:\n{answer}")
            used_vision = True

    if not pieces:
        return None, used_vision
    return (
        ContextBlock(
            **base,
            text=truncate_text("\n\n".join(pieces), char_budget),
            metadata={"reader": "image_vision" if used_vision else "image_index"},
        ),
        used_vision,
    )


