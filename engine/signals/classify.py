"""Strict two-stage shortlist classification with hierarchy-safe constraints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.signals.prompts import (
    TOPIC_ABSTENTION_TOKEN,
    build_locked_classification_prompt,
    build_primary_classification_prompt,
    locked_classification_schema,
    primary_classification_schema,
)
from engine.signals.protocol import ConstrainedTextGenerator
from engine.signals.taxonomy import TaxonomyCandidate, TaxonomyIndex
from engine.signals.text_types import (
    CategoryCitation,
    ClassificationPayload,
    ClassificationResult,
    EntityValue,
    GenerationAttempt,
    KeywordValue,
    SentimentValue,
)
from engine.transcribe.chunk import chunk_transcript
from engine.transcribe.types import AnalysisWindow, TranscriptionRun


class InvalidClassificationOutput(ValueError):
    """The model response is not the exact strict contract requested."""


class HierarchyRedundancyError(InvalidClassificationOutput):
    """Two selected candidates overlap along one hierarchy path."""


@dataclass(frozen=True)
class CandidateRecord:
    time_range: TimeRange
    candidates: tuple[TaxonomyCandidate, ...]


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InvalidClassificationOutput(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise InvalidClassificationOutput(f"non-finite JSON number: {value}")


def _parse_json_object(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise InvalidClassificationOutput("classification output must be text")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except InvalidClassificationOutput:
        raise
    except (json.JSONDecodeError, TypeError, UnicodeError) as exc:
        raise InvalidClassificationOutput("classification output must be one JSON object") from exc
    if not isinstance(value, dict):
        raise InvalidClassificationOutput("classification output must be one JSON object")
    return value


def _strict_payload(value: dict[str, Any]) -> ClassificationPayload:
    expected = {
        "primary_category",
        "secondary_categories",
        "keywords",
        "entities",
        "sentiment",
    }
    if set(value) != expected:
        raise InvalidClassificationOutput("classification output fields do not match contract")
    secondary = value["secondary_categories"]
    keywords = value["keywords"]
    entities = value["entities"]
    if not isinstance(secondary, list) or not isinstance(keywords, list) or not isinstance(entities, list):
        raise InvalidClassificationOutput("classification output arrays have invalid types")
    try:
        return ClassificationPayload(
            primary_category=value["primary_category"],
            secondary_categories=tuple(secondary),
            keywords=tuple(KeywordValue.model_validate(item) for item in keywords),
            entities=tuple(EntityValue.model_validate(item) for item in entities),
            sentiment=SentimentValue.model_validate(value["sentiment"]),
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise InvalidClassificationOutput("classification output values are invalid") from exc


def _is_ancestor(
    first: TaxonomyCandidate, second: TaxonomyCandidate
) -> bool:
    return len(first.path) < len(second.path) and second.path[: len(first.path)] == first.path


def validate_classification_output(
    raw: str,
    candidates: Sequence[TaxonomyCandidate],
) -> ClassificationPayload:
    payload = _strict_payload(_parse_json_object(raw))
    candidate_tuple = tuple(candidates)
    candidate_by_id = {candidate.taxonomy_id: candidate for candidate in candidate_tuple}
    if len(candidate_by_id) != len(candidate_tuple) or not candidate_tuple:
        raise ValueError("candidates must have unique IDs and cannot be empty")
    selected = (payload.primary_category, *payload.secondary_categories)
    if any(category_id not in candidate_by_id for category_id in selected):
        raise InvalidClassificationOutput("selected category is outside the supplied shortlist")
    selected_candidates = [candidate_by_id[category_id] for category_id in selected]
    for index, first in enumerate(selected_candidates):
        for second in selected_candidates[index + 1 :]:
            if _is_ancestor(first, second) or _is_ancestor(second, first):
                raise HierarchyRedundancyError(
                    "redundant ancestor/descendant categories: "
                    f"{first.taxonomy_id}, {second.taxonomy_id}"
                )
    return payload


def validate_cited_classification_output(
    raw: str,
    candidates: Sequence[TaxonomyCandidate],
    window: AnalysisWindow,
) -> tuple[ClassificationPayload, tuple[CategoryCitation, ...]]:
    value = _parse_json_object(raw)
    if set(value) != {
        "primary_category", "secondary_categories", "keywords", "entities",
        "sentiment", "category_evidence",
    }:
        raise InvalidClassificationOutput("cited output fields do not match contract")
    payload = validate_classification_output(
        json.dumps(
            {key: item for key, item in value.items() if key != "category_evidence"},
            allow_nan=False,
        ),
        candidates,
    )
    raw_citations = value["category_evidence"]
    if not isinstance(raw_citations, list):
        raise InvalidClassificationOutput("category_evidence must be an array")
    try:
        citations = tuple(
            CategoryCitation.model_validate_json(json.dumps(item, allow_nan=False))
            for item in raw_citations
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise InvalidClassificationOutput("invalid category citations") from exc
    cited_ids = tuple(item.category_id for item in citations)
    if not cited_ids or cited_ids[0] != payload.primary_category:
        raise InvalidClassificationOutput("citations must start with the primary category")
    remaining = iter(payload.secondary_categories)
    if not all(category_id in remaining for category_id in cited_ids[1:]):
        raise InvalidClassificationOutput("citations must match selected category roles")
    if len(cited_ids) != len(set(cited_ids)):
        raise InvalidClassificationOutput("duplicate category citation")
    # A proposed secondary without cited support is dropped rather than failing
    # the window; every proposal is still independently verified downstream.
    payload = payload.model_copy(update={"secondary_categories": cited_ids[1:]})
    for citation in citations:
        for index in citation.segment_indices:
            if type(index) is not int or index >= len(window.segments):
                raise InvalidClassificationOutput("cited index outside analysis window")
            segment_range = window.segments[index].time_range
            if (
                segment_range.end_seconds <= segment_range.start_seconds
                or segment_range.start_seconds < window.time_range.start_seconds
                or segment_range.end_seconds > window.time_range.end_seconds
            ):
                raise InvalidClassificationOutput("invalid cited segment range")
    return payload, citations


def _prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def _validate_primary_output(
    raw: str,
    candidates: Sequence[TaxonomyCandidate],
) -> str:
    value = _parse_json_object(raw)
    if set(value) != {"primary_category"}:
        raise InvalidClassificationOutput("primary output fields do not match contract")
    primary_id = value["primary_category"]
    candidate_ids = {candidate.taxonomy_id for candidate in candidates}
    if not isinstance(primary_id, str) or primary_id not in (candidate_ids | {TOPIC_ABSTENTION_TOKEN}):
        raise InvalidClassificationOutput(
            "primary category is outside the supplied shortlist"
        )
    return primary_id


def classify_window(
    window: AnalysisWindow,
    candidates: Sequence[TaxonomyCandidate],
    *,
    generator: ConstrainedTextGenerator,
    token: CancellationToken,
    checkpoint: Callable[[GenerationAttempt], Any],
) -> ClassificationResult:
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("classification requires at least one candidate")
    if len({candidate.taxonomy_id for candidate in candidate_tuple}) != len(
        candidate_tuple
    ):
        raise ValueError("classification candidates must have unique IDs")
    attempts: list[GenerationAttempt] = []

    def generate_stage(
        *,
        attempt_number: int,
        stage: str,
        prompt: str,
        schema: dict,
        max_tokens: int,
    ):
        token.raise_if_cancelled()
        try:
            generated = generator.generate_json(
                prompt,
                schema,
                max_tokens=max_tokens,
                token=token,
            )
        except JobCancelled as exc:
            attempt = GenerationAttempt(
                attempt=attempt_number,
                stage=stage,
                prompt_sha256=_prompt_digest(prompt),
                error_code=type(exc).__name__,
                reason=str(exc),
            )
            attempts.append(attempt)
            checkpoint(attempt)
            raise
        except Exception as exc:
            attempt = GenerationAttempt(
                attempt=attempt_number,
                stage=stage,
                prompt_sha256=_prompt_digest(prompt),
                error_code=type(exc).__name__,
                reason=str(exc),
            )
            attempts.append(attempt)
            checkpoint(attempt)
            return None

        attempt = GenerationAttempt(
            attempt=attempt_number,
            stage=stage,
            prompt_sha256=_prompt_digest(prompt),
            raw=generated.raw,
            finish_reason=generated.finish_reason,
            seconds=generated.seconds,
            token_count=generated.token_count,
        )
        attempts.append(attempt)
        # The raw attempt is durable before any validation or retry decision.
        checkpoint(attempt)
        token.raise_if_cancelled()
        return generated

    primary_prompt = build_primary_classification_prompt(
        window, candidate_tuple, topic_sufficiency=True
    )
    primary_generated = generate_stage(
        attempt_number=1,
        stage="primary",
        prompt=primary_prompt,
        schema=primary_classification_schema(candidate_tuple, topic_sufficiency=True),
        max_tokens=64,
    )
    if primary_generated is None:
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="GENERATION_ERROR",
        )
    if primary_generated.finish_reason != "stop":
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="INCOMPLETE_GENERATION",
        )
    try:
        primary_id = _validate_primary_output(primary_generated.raw, candidate_tuple)
    except InvalidClassificationOutput:
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="INVALID_CLASSIFICATION_OUTPUT",
        )

    token.raise_if_cancelled()
    if primary_id == TOPIC_ABSTENTION_TOKEN:
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="INSUFFICIENT_TOPIC_EVIDENCE",
            policy_version="english-audio-topic-sufficiency-v1",
        )
    return classify_locked_payload(
        window, candidate_tuple, primary_id, generator=generator, token=token,
        prior_attempts=attempts, checkpoint=checkpoint,
    )


def classify_locked_payload(
    window: AnalysisWindow,
    candidates: Sequence[TaxonomyCandidate],
    primary_id: str,
    *,
    generator: ConstrainedTextGenerator,
    token: CancellationToken,
    prior_attempts: Sequence[GenerationAttempt],
    checkpoint: Callable[[GenerationAttempt], Any],
) -> ClassificationResult:
    """Generate and strictly validate the payload for a previously locked primary."""
    candidate_tuple = tuple(candidates)
    if (
        not candidate_tuple
        or len({item.taxonomy_id for item in candidate_tuple}) != len(candidate_tuple)
        or primary_id not in {item.taxonomy_id for item in candidate_tuple}
    ):
        raise ValueError("locked primary requires a unique candidate shortlist")
    attempts = list(prior_attempts)
    if len(attempts) != 1 or attempts[0].attempt != 1:
        raise ValueError("locked payload requires the retained primary attempt")
    payload_prompt = build_locked_classification_prompt(
        window, candidate_tuple, primary_id, topic_sufficiency=True
    )
    token.raise_if_cancelled()
    try:
        payload_generated = generator.generate_json(
            payload_prompt,
            locked_classification_schema(
                candidate_tuple, primary_id, topic_sufficiency=True,
                segment_count=len(window.segments),
            ),
            max_tokens=1024, token=token,
        )
    except Exception as exc:
        attempt = GenerationAttempt(
            attempt=2, stage="payload", prompt_sha256=_prompt_digest(payload_prompt),
            error_code=type(exc).__name__, reason=str(exc),
        )
        attempts.append(attempt)
        checkpoint(attempt)
        token.raise_if_cancelled()
        if isinstance(exc, JobCancelled):
            raise
        payload_generated = None
    else:
        attempt = GenerationAttempt(
            attempt=2, stage="payload", prompt_sha256=_prompt_digest(payload_prompt),
            raw=payload_generated.raw, finish_reason=payload_generated.finish_reason,
            seconds=payload_generated.seconds, token_count=payload_generated.token_count,
        )
        attempts.append(attempt)
        checkpoint(attempt)
        token.raise_if_cancelled()
    if payload_generated is None:
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="GENERATION_ERROR",
        )
    if payload_generated.finish_reason != "stop":
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="INCOMPLETE_GENERATION",
        )
    try:
        payload, citations = validate_cited_classification_output(
            payload_generated.raw, candidate_tuple, window
        )
        if payload.primary_category != primary_id:
            raise InvalidClassificationOutput(
                "final primary category does not match the locked primary"
            )
    except HierarchyRedundancyError:
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="HIERARCHY_REDUNDANCY",
        )
    except InvalidClassificationOutput:
        return ClassificationResult(
            time_range=window.time_range,
            payload=None,
            attempts=tuple(attempts),
            error_code="INVALID_CLASSIFICATION_OUTPUT",
        )
    return ClassificationResult(
        time_range=window.time_range,
        payload=payload,
        attempts=tuple(attempts),
        category_citations=citations,
        policy_version="english-audio-topic-sufficiency-v1",
    )


def analysis_windows_from_run(run: TranscriptionRun) -> tuple[AnalysisWindow, ...]:
    """Chunk each contiguous successful transcription run independently."""
    output: list[AnalysisWindow] = []
    contiguous_segments = []

    def flush() -> None:
        if contiguous_segments:
            output.extend(chunk_transcript(tuple(contiguous_segments)))
            contiguous_segments.clear()

    for result in run.windows:
        if result.error_code is not None or not result.segments:
            flush()
            continue
        contiguous_segments.extend(result.segments)
    flush()
    return tuple(output)


def classify_transcription_run(
    run: TranscriptionRun,
    *,
    taxonomy_index: TaxonomyIndex,
    generator: ConstrainedTextGenerator,
    token: CancellationToken,
    candidate_checkpoint: Callable[[CandidateRecord], Any],
    attempt_checkpoint: Callable[[GenerationAttempt], Any],
    result_checkpoint: Callable[[ClassificationResult], Any],
    top_k: int = 15,
) -> tuple[ClassificationResult, ...]:
    results: list[ClassificationResult] = []
    for window in analysis_windows_from_run(run):
        token.raise_if_cancelled()
        transcript = " ".join(segment.text for segment in window.segments)
        candidates = taxonomy_index.shortlist(transcript, top_k=top_k, token=token)
        candidate_checkpoint(
            CandidateRecord(time_range=window.time_range, candidates=candidates)
        )
        token.raise_if_cancelled()
        result = classify_window(
            window,
            candidates,
            generator=generator,
            token=token,
            checkpoint=attempt_checkpoint,
        )
        result_checkpoint(result)
        results.append(result)
    return tuple(results)
