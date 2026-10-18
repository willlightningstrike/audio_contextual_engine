from __future__ import annotations

import math

import pytest

from engine.domain import TimeRange, TranscriptSegment
from engine.signals.classify import analysis_windows_from_run
from engine.signals.text_types import (
    CategoryCitation,
    ClassificationPayload,
    ClassificationResult,
    GenerationAttempt,
    SentimentValue,
)
from engine.transcribe.types import TranscriptionRun, WindowResult


def _range(start: float, end: float) -> TimeRange:
    return TimeRange(start_seconds=start, end_seconds=end)


def _segment(start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(time_range=_range(start, end), text="English topic", confidence=0.9, language="en")


def _result(window, primary: str | None, citations: tuple[CategoryCitation, ...] = ()) -> ClassificationResult:
    return ClassificationResult(
        time_range=window.time_range,
        payload=(
            ClassificationPayload(
                primary_category=primary,
                secondary_categories=tuple(c.category_id for c in citations if c.category_id != primary),
                keywords=(),
                entities=(),
                sentiment=SentimentValue(label="neutral", score=0.0),
            )
            if primary is not None else None
        ),
        attempts=(GenerationAttempt(attempt=1, stage="primary", prompt_sha256="0" * 64),),
        error_code="INSUFFICIENT_TOPIC_EVIDENCE" if primary is None else None,
        category_citations=citations,
        policy_version="english-audio-topic-sufficiency-v1",
    )


def cited_run(transcript_seconds: float, support_seconds: float):
    if transcript_seconds > 30:
        segments = tuple(_segment(start, start + 30) for start in range(0, int(transcript_seconds), 30))
    else:
        segments = (_segment(0, support_seconds),)
        if transcript_seconds > support_seconds:
            segments += (_segment(support_seconds, transcript_seconds),)
    run = TranscriptionRun(
        windows=(WindowResult(time_range=_range(0, transcript_seconds), segments=segments, language="en"),),
        status="ready",
    )
    windows = analysis_windows_from_run(run)
    results = tuple(
        _result(window, "A", (CategoryCitation(category_id="A", segment_indices=tuple(
            index for index, segment in enumerate(window.segments)
            if segment.time_range.end_seconds <= support_seconds
        )),))
        for window in windows if window.segments[0].time_range.start_seconds < support_seconds
    )
    if len(results) < len(windows):
        results += tuple(_result(window, None) for window in windows[len(results):])
    return run, results


@pytest.mark.parametrize(
    ("transcript_seconds", "support_seconds", "audio_seconds", "expected"),
    [(20, 10, 40, "insufficient"), (20, 10.01, 40.04, "eligible"),
     (20, 11, 44.01, "insufficient"), (600, 600, 3600, "insufficient"),
     (1800, 1800, 1800, "eligible")],
)
def test_materiality_boundaries(transcript_seconds, support_seconds, audio_seconds, expected):
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run, classifications = cited_run(transcript_seconds, support_seconds)
    decision = evaluate_topic_sufficiency(run, classifications, audio_seconds=audio_seconds)
    assert decision.outcome == expected
    assert decision.transcript_seconds == transcript_seconds
    assert dict(decision.primary_role_support_seconds) == {"A": support_seconds}


def test_all_abstain_is_insufficient_and_abstained_speech_stays_in_denominator():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run, results = cited_run(10, 10)
    window = analysis_windows_from_run(run)[0]
    abstained = evaluate_topic_sufficiency(run, (_result(window, None),), audio_seconds=10)
    assert abstained.outcome == "insufficient"
    assert abstained.transcript_seconds == 10

    run = TranscriptionRun(windows=(
        WindowResult(time_range=_range(0, 10), segments=(_segment(0, 10),), language="en"),
        WindowResult(time_range=_range(10, 10)),
        WindowResult(time_range=_range(10, 15), segments=(_segment(10, 15),), language="en"),
    ), status="ready")
    classified_window, abstained_window = analysis_windows_from_run(run)
    partial = evaluate_topic_sufficiency(
        run, (
            _result(classified_window, "A", (CategoryCitation(category_id="A", segment_indices=(0,)),)),
            _result(abstained_window, None),
        ),
        audio_seconds=15,
    )
    assert partial.transcript_seconds == 15
    assert dict(partial.primary_role_support_seconds) == {"A": 10}


def test_overlapping_transcript_ranges_count_once_across_partitioned_runs():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run = TranscriptionRun(windows=(
        WindowResult(time_range=_range(0, 10), segments=(_segment(0, 10),), language="en"),
        WindowResult(time_range=_range(9, 9)),
        WindowResult(time_range=_range(9, 19), segments=(_segment(9, 19),), language="en"),
    ), status="ready")
    first, second = analysis_windows_from_run(run)
    decision = evaluate_topic_sufficiency(run, (
        _result(first, "A", (CategoryCitation(category_id="A", segment_indices=(0,)),)),
        _result(second, None),
    ), audio_seconds=40)
    assert decision.transcript_seconds == 19
    assert decision.outcome == "eligible"


@pytest.mark.parametrize("audio_seconds", [0, -1, math.inf, math.nan])
def test_invalid_audio_duration_is_rejected(audio_seconds):
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run, results = cited_run(10, 10)
    with pytest.raises(ValueError, match="audio_seconds"):
        evaluate_topic_sufficiency(run, results, audio_seconds=audio_seconds)


def test_uncited_category_payload_is_rejected():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run, _ = cited_run(10, 10)
    window = analysis_windows_from_run(run)[0]
    with pytest.raises(ValueError, match="citation"):
        evaluate_topic_sufficiency(run, (_result(window, "A"),), audio_seconds=10)


def test_citation_index_outside_its_window_is_rejected():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run, _ = cited_run(10, 10)
    window = analysis_windows_from_run(run)[0]
    with pytest.raises(ValueError, match="index"):
        evaluate_topic_sufficiency(run, (_result(window, "A", (
            CategoryCitation(category_id="A", segment_indices=(1,)),
        )),), audio_seconds=10)


def test_overlapping_primary_windows_are_rejected():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run = TranscriptionRun(windows=(
        WindowResult(time_range=_range(0, 10), segments=(_segment(0, 10),), language="en"),
        WindowResult(time_range=_range(8, 8)),
        WindowResult(time_range=_range(8, 18), segments=(_segment(8, 18),), language="en"),
    ), status="ready")
    first, second = analysis_windows_from_run(run)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_topic_sufficiency(run, (
            _result(first, "A", (CategoryCitation(category_id="A", segment_indices=(0,)),)),
            _result(second, "B", (CategoryCitation(category_id="B", segment_indices=(0,)),)),
        ), audio_seconds=18)


def test_empty_failed_source_window_rejects_otherwise_eligible_evidence():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run = TranscriptionRun(windows=(
        WindowResult(time_range=_range(0, 10), segments=(_segment(0, 10),), language="en"),
        WindowResult(time_range=_range(10, 20), error_code="MODEL_FAILURE",
                     failure_stage="transcription", failure_reason="model_call_failed"),
    ), status="partial")
    window = analysis_windows_from_run(run)[0]
    with pytest.raises(ValueError, match="failed|ready|partial"):
        evaluate_topic_sufficiency(run, (
            _result(window, "A", (CategoryCitation(category_id="A", segment_indices=(0,)),)),
        ), audio_seconds=20)


def test_approved_short_terminal_uncertain_tail_uses_full_audio_denominator():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run = TranscriptionRun(windows=(
        WindowResult(time_range=_range(0, 30), segments=(_segment(1, 12),), language="en"),
        WindowResult(time_range=_range(30, 35), error_code="VOCAL_PRESENCE_UNCERTAIN",
                     failure_stage="vocal_screening", failure_reason="vocal_presence_uncertain"),
    ), status="partial")
    window = analysis_windows_from_run(run)[0]
    decision = evaluate_topic_sufficiency(run, (
        _result(window, "A", (CategoryCitation(category_id="A", segment_indices=(0,)),)),
    ), audio_seconds=35)
    assert decision.outcome == "eligible"
    assert decision.audio_seconds == 35
    assert decision.transcript_seconds == 11


def test_partial_run_without_window_error_cannot_pass_gate():
    from engine.signals.topic_sufficiency import evaluate_topic_sufficiency

    run, results = cited_run(10, 10)
    partial = run.model_copy(update={"status": "partial"})
    with pytest.raises(ValueError, match="ready|partial"):
        evaluate_topic_sufficiency(partial, results, audio_seconds=10)
