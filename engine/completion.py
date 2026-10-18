"""Pure completion policy for conservative English-only result promotion."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from engine.domain import TimeRange
from engine.signals.text_types import ClassificationResult
from engine.transcribe.types import TranscriptionRun


_MAX_TERMINAL_VOCAL_UNCERTAINTY_SECONDS = 30.0


def _has_approved_terminal_vocal_uncertainty(run: TranscriptionRun) -> bool:
    """Allow only one short, uncertain tail after successful English windows."""
    windows = run.windows
    if run.status != "partial" or len(windows) < 2:
        return False
    if windows[0].time_range.start_seconds != 0:
        return False
    if any(
        previous.time_range.end_seconds != current.time_range.start_seconds
        for previous, current in zip(windows, windows[1:])
    ):
        return False

    tail = windows[-1]
    tail_duration = tail.time_range.end_seconds - tail.time_range.start_seconds
    if (
        tail.error_code != "VOCAL_PRESENCE_UNCERTAIN"
        or tail.failure_stage != "vocal_screening"
        or tail.failure_reason != "vocal_presence_uncertain"
        or tail.language is not None
        or tail.segments
        or tail_duration > _MAX_TERMINAL_VOCAL_UNCERTAINTY_SECONDS
    ):
        return False

    return all(
        window.error_code is None
        and window.failure_stage is None
        and window.failure_reason is None
        and window.language == "en"
        and bool(window.segments)
        and all(segment.language == "en" for segment in window.segments)
        for window in windows[:-1]
    )


def completion_errors(
    run: TranscriptionRun,
    classifications: tuple[ClassificationResult, ...],
) -> tuple[str, ...]:
    """Return blockers, allowing only the approved short uncertain tail window."""
    errors: list[str] = []
    allowed_terminal_vocal_uncertainty = _has_approved_terminal_vocal_uncertainty(run)
    if run.status != "ready" and not allowed_terminal_vocal_uncertainty:
        errors.append(f"transcription_{run.status}")
    if not run.windows:
        errors.append("no_transcription_windows")

    segments = tuple(segment for window in run.windows for segment in window.segments)
    if not segments:
        errors.append("no_usable_transcript")
    windows_to_validate = (
        run.windows[:-1]
        if allowed_terminal_vocal_uncertainty
        else run.windows
    )
    for window in windows_to_validate:
        if window.error_code is not None:
            errors.append(f"transcription_window_failed:{window.error_code}")
        elif window.failure_stage is not None or window.failure_reason is not None:
            errors.append(
                "transcription_window_uncertain:"
                + (window.failure_stage or window.failure_reason or "unknown")
            )

    eligible = tuple(
        result
        for result in classifications
        if result.payload is not None
        or result.error_code == "INSUFFICIENT_TOPIC_EVIDENCE"
    )
    for result in classifications:
        if result not in eligible:
            errors.append(
                "classification_failed:" + (result.error_code or "missing_payload")
            )

    for segment in segments:
        covers = tuple(
            result
            for result in eligible
            if result.time_range.start_seconds <= segment.time_range.start_seconds
            and result.time_range.end_seconds >= segment.time_range.end_seconds
        )
        if len(covers) != 1:
            errors.append("transcript_segment_classification_coverage_incomplete")
            break

    if eligible:
        for result in eligible:
            if not any(
                result.time_range.start_seconds <= segment.time_range.start_seconds
                and result.time_range.end_seconds >= segment.time_range.end_seconds
                for segment in segments
            ):
                errors.append("classification_without_transcript_evidence")
                break
    elif segments:
        errors.append("no_successful_classifications")

    return tuple(errors)


# english-audio-aggregation-v3 coverage rules. Failed and non-English windows are
# excluded from the topic decision instead of blocking it, within these caps.
COVERAGE_POLICY = "english-audio-aggregation-v3"
MAX_NON_ENGLISH_SHARE = 0.20
MAX_TECHNICAL_FAILURE_SHARE = 0.20
_NON_ENGLISH = "NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE"


class CoverageAssessment(BaseModel):
    """Durable record of which ranges were excluded and why a job may proceed."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    policy_version: Literal["english-audio-aggregation-v3"] = COVERAGE_POLICY
    audio_seconds: float = Field(gt=0)
    non_english_share: float = Field(ge=0, le=1)
    technical_failure_share: float = Field(ge=0, le=1)
    # Technical failures in transcription or classification, excluded from the decision.
    failed_ranges: tuple[TimeRange, ...] = ()
    # Out-of-scope audio: uncertain vocals (uncapped) and non-English speech (capped).
    excluded_ranges: tuple[TimeRange, ...] = ()
    error_code: str | None = None


def _span(time_range: TimeRange) -> float:
    return time_range.end_seconds - time_range.start_seconds


def assess_coverage(
    run: TranscriptionRun,
    classifications: tuple[ClassificationResult, ...],
    *,
    audio_seconds: float,
) -> CoverageAssessment:
    """Apply the v3 caps; ``error_code`` names the first rule a job fails."""
    windows = run.windows
    vocal_uncertain = [w for w in windows if w.failure_reason == "vocal_presence_uncertain"]
    non_english = [w for w in windows if w.error_code == _NON_ENGLISH]
    technical = [
        w for w in windows
        if w.error_code is not None and w not in vocal_uncertain and w not in non_english
    ]
    english = [w for w in windows if w.error_code is None and w.language == "en" and w.segments]
    failed_classifications = [
        result for result in classifications
        if result.payload is None and result.error_code != "INSUFFICIENT_TOPIC_EVIDENCE"
    ]

    vocal_seconds = sum(_span(w.time_range) for w in english + non_english + technical)
    non_english_share = (
        sum(_span(w.time_range) for w in non_english) / vocal_seconds if vocal_seconds else 0.0
    )
    technical_source_seconds = sum(_span(w.time_range) for w in technical)
    english_scope = technical_source_seconds + sum(_span(r.time_range) for r in classifications)
    technical_share = (
        (technical_source_seconds + sum(_span(r.time_range) for r in failed_classifications))
        / english_scope
        if english_scope else 0.0
    )

    covered_to = 0.0
    contiguous = True
    for window in windows:
        contiguous = contiguous and window.time_range.start_seconds == covered_to
        covered_to = window.time_range.end_seconds
    complete = contiguous and covered_to >= audio_seconds

    if non_english_share > MAX_NON_ENGLISH_SHARE:
        error = "NON_ENGLISH_SHARE_EXCEEDED"
    elif not complete:
        error = "TRANSCRIPTION_INCOMPLETE"
    elif not any(w.segments for w in english):
        error = "NO_USABLE_ENGLISH_TRANSCRIPT"
    elif technical_share > MAX_TECHNICAL_FAILURE_SHARE:
        error = "TECHNICAL_FAILURE_SHARE_EXCEEDED"
    else:
        error = None
    return CoverageAssessment(
        audio_seconds=float(audio_seconds),
        non_english_share=min(1.0, non_english_share),
        technical_failure_share=min(1.0, technical_share),
        failed_ranges=tuple(w.time_range for w in technical)
        + tuple(r.time_range for r in failed_classifications),
        excluded_ranges=tuple(w.time_range for w in vocal_uncertain + non_english),
        error_code=error,
    )
