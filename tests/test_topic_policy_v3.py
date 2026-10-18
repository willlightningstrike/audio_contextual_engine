"""english-audio-aggregation-v3: coverage caps and mixed-topic decisions."""

from __future__ import annotations

import pytest

from engine.completion import assess_coverage
from engine.domain import TimeRange, TranscriptSegment
from engine.signals.classify import analysis_windows_from_run
from engine.signals.text_types import (
    CategoryCitation,
    ClassificationPayload,
    ClassificationResult,
    GenerationAttempt,
    SentimentValue,
)
from engine.signals.topic_sufficiency import TopicDecision, evaluate_topic_policy_v3
from engine.standards import load_taxonomy
from engine.transcribe.types import TranscriptionRun, WindowResult

AUTOS, FINANCE, SPORTS, BASEBALL, BASKETBALL, SOCCER = "1", "391", "483", "545", "547", "533"
COOKING, TRAVEL = "216", "653"


def _range(start: float, end: float) -> TimeRange:
    return TimeRange(start_seconds=start, end_seconds=end)


def _source_window(index: int, kind: str) -> WindowResult:
    time_range = _range(index * 30, index * 30 + 30)
    if kind == "en":
        segment = TranscriptSegment(
            time_range=time_range, text="English topic", confidence=0.9, language="en"
        )
        return WindowResult(time_range=time_range, segments=(segment,), language="en")
    if kind == "music":
        return WindowResult(
            time_range=time_range, error_code="VOCAL_PRESENCE_UNCERTAIN",
            failure_stage="vocal_screening", failure_reason="vocal_presence_uncertain",
        )
    if kind == "fr":
        return WindowResult(
            time_range=time_range, language="fr",
            error_code="NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE", failure_stage="language_detection",
            failure_reason="non_english_or_unconfirmed_language",
        )
    return WindowResult(
        time_range=time_range, language="en", error_code="MODEL_FAILURE",
        failure_stage="timestamp_validation", failure_reason="invalid_timestamps",
    )


def _result(window, label) -> ClassificationResult:
    attempts = (GenerationAttempt(attempt=1, stage="primary", prompt_sha256="0" * 64),)
    if label == "fail":
        return ClassificationResult(
            time_range=window.time_range, payload=None, attempts=attempts,
            error_code="INVALID_CLASSIFICATION_OUTPUT",
        )
    if label is None:
        return ClassificationResult(
            time_range=window.time_range, payload=None, attempts=attempts,
            error_code="INSUFFICIENT_TOPIC_EVIDENCE",
            policy_version="english-audio-topic-sufficiency-v1",
        )
    primary, *secondaries = label if isinstance(label, tuple) else (label,)
    every = tuple(range(len(window.segments)))
    return ClassificationResult(
        time_range=window.time_range,
        payload=ClassificationPayload(
            primary_category=primary, secondary_categories=tuple(secondaries),
            keywords=(), entities=(), sentiment=SentimentValue(label="neutral", score=0.0),
        ),
        attempts=attempts,
        category_citations=tuple(
            CategoryCitation(category_id=category_id, segment_indices=every)
            for category_id in (primary, *secondaries)
        ),
        policy_version="english-audio-topic-sufficiency-v1",
    )


def scenario(windows: str, labels: list):
    """``windows``: one character per 30 s window (e English, m music, f French,
    x ASR failure). Each run of up to three English windows is one 90 s analysis
    window; ``labels`` gives each analysis window's topic, None, or "fail"."""
    kinds = {"e": "en", "m": "music", "f": "fr", "x": "failure"}
    source = tuple(_source_window(index, kinds[code]) for index, code in enumerate(windows))
    status = "ready" if all(window.error_code is None for window in source) else "partial"
    run = TranscriptionRun(windows=source, status=status)
    analysis = analysis_windows_from_run(run)
    assert len(analysis) == len(labels)
    return run, tuple(_result(window, label) for window, label in zip(analysis, labels))


def decide(windows: str, labels: list) -> TopicDecision:
    run, results = scenario(windows, labels)
    assessment = assess_coverage(run, results, audio_seconds=len(windows) * 30.0)
    assert assessment.error_code is None
    return evaluate_topic_policy_v3(
        run, results, audio_seconds=len(windows) * 30.0, taxonomy=load_taxonomy()
    )


# Coverage caps


def test_interior_music_break_is_excluded_not_blocking():
    run, results = scenario("eeemeee", [COOKING, COOKING])
    assessment = assess_coverage(run, results, audio_seconds=210)
    assert assessment.error_code is None
    assert assessment.excluded_ranges == (_range(90, 120),)
    assert assessment.failed_ranges == ()


def test_short_non_english_quote_is_excluded_within_cap():
    run, results = scenario("e" * 9 + "f" + "e" * 9, [COOKING] * 6)
    assessment = assess_coverage(run, results, audio_seconds=570)
    assert assessment.error_code is None
    assert assessment.non_english_share == pytest.approx(1 / 19)
    assert assessment.excluded_ranges == (_range(270, 300),)


def test_mostly_non_english_audio_stays_out_of_scope():
    run, results = scenario("eeeffeee", [COOKING, COOKING])
    assessment = assess_coverage(run, results, audio_seconds=240)
    assert assessment.non_english_share == pytest.approx(0.25)
    assert assessment.error_code == "NON_ENGLISH_SHARE_EXCEEDED"


def test_one_failed_window_in_six_is_excluded_as_partial_coverage():
    run, results = scenario("e" * 18, [BASEBALL, BASEBALL, "fail", BASEBALL, BASEBALL, BASEBALL])
    assessment = assess_coverage(run, results, audio_seconds=540)
    assert assessment.error_code is None
    assert assessment.technical_failure_share == pytest.approx(1 / 6)
    assert assessment.failed_ranges == (_range(180, 270),)


def test_too_many_technical_failures_still_fail():
    run, results = scenario("eeexeeeeee", [BASEBALL, "fail", BASEBALL])
    assessment = assess_coverage(run, results, audio_seconds=300)
    # One failed ASR window and one failed 90 s classification of 300 s in scope.
    assert assessment.technical_failure_share == pytest.approx(120 / 300)
    assert assessment.error_code == "TECHNICAL_FAILURE_SHARE_EXCEEDED"
    assert _range(90, 120) in assessment.failed_ranges


def test_unvisited_windows_cannot_complete():
    run, results = scenario("eee", [COOKING])
    assessment = assess_coverage(run, results, audio_seconds=120)
    assert assessment.error_code == "TRANSCRIPTION_INCOMPLETE"


def test_audio_without_english_speech_has_no_usable_transcript():
    run, results = scenario("mmm", [])
    assessment = assess_coverage(run, results, audio_seconds=90)
    assert assessment.error_code == "NO_USABLE_ENGLISH_TRANSCRIPT"


# Topic decisions


def test_single_dominant_topic_is_unchanged():
    decision = decide("e" * 9, [BASEBALL, BASEBALL, None])
    assert decision.topic_structure == "single"
    assert decision.primary_category == BASEBALL
    assert decision.secondary_categories == ()


def test_window_winner_becomes_secondary_to_dominant_topic():
    # Autos wins 3 of 5 windows; Personal Finance wins 2 and was never cited as a
    # secondary. v2 ignored it; v3 reports it as the secondary.
    decision = decide("e" * 15, [AUTOS, FINANCE, AUTOS, FINANCE, AUTOS])
    assert decision.topic_structure == "single"
    assert (decision.primary_category, decision.secondary_categories) == (AUTOS, (FINANCE,))
    assert dict(decision.selected_support_seconds) == {AUTOS: 270, FINANCE: 180}


def test_sibling_topics_roll_up_to_shared_parent():
    decision = decide("e" * 9, [BASEBALL, BASKETBALL, SOCCER])
    assert decision.topic_structure == "parent_rollup"
    assert decision.primary_category == SPORTS
    # Descendants of the chosen parent are redundant as secondaries.
    assert decision.secondary_categories == ()
    assert dict(decision.selected_support_seconds) == {SPORTS: 270}


def test_rollup_keeps_an_unrelated_secondary():
    decision = decide("e" * 15, [BASEBALL, BASKETBALL, SOCCER, TRAVEL, None])
    assert decision.topic_structure == "parent_rollup"
    assert (decision.primary_category, decision.secondary_categories) == (SPORTS, (TRAVEL,))


def test_unrelated_major_topics_are_reported_as_multi_topic():
    decision = decide("e" * 12, [AUTOS, AUTOS, FINANCE, COOKING])
    # Autos is exactly 50%, which is not a strict majority.
    assert decision.topic_structure == "multi_topic"
    assert decision.primary_category == AUTOS
    # Equal support is ordered by category ID.
    assert decision.secondary_categories == (COOKING, FINANCE)


def test_fragmented_topics_still_abstain():
    decision = decide("e" * 21, [AUTOS, FINANCE, COOKING, TRAVEL, BASEBALL, None, None])
    # Each topic is one window in seven (about 14%), below the 20% major-topic share.
    assert decision.outcome == "insufficient"
    assert decision.topic_structure is None


def test_failed_windows_are_left_out_of_the_decision():
    decision = decide("e" * 18, [BASEBALL, BASEBALL, "fail", BASEBALL, None, None])
    assert decision.transcript_seconds == 450
    assert decision.primary_category == BASEBALL
    assert decision.topic_structure == "single"


def test_long_programme_with_failed_windows_can_publish():
    # Mirrors the 93-minute baseball case: one window in ten failed verification.
    labels = [BASEBALL] * 9 + ["fail"]
    run, results = scenario("e" * 30, labels)
    assessment = assess_coverage(run, results, audio_seconds=900)
    assert assessment.error_code is None
    decision = evaluate_topic_policy_v3(run, results, audio_seconds=900, taxonomy=load_taxonomy())
    assert decision.primary_category == BASEBALL


def test_v2_decisions_keep_their_serialized_form():
    decision = TopicDecision(
        outcome="insufficient", primary_category=None, transcript_seconds=1.0,
        audio_seconds=2.0, primary_role_support_seconds=(), secondary_role_support_seconds=(),
    )
    assert "topic_structure" not in decision.model_dump(mode="json")
    with pytest.raises(ValueError):
        TopicDecision(
            outcome="insufficient", primary_category=None, transcript_seconds=1.0,
            audio_seconds=2.0, primary_role_support_seconds=(), secondary_role_support_seconds=(),
            topic_structure="single",
        )
