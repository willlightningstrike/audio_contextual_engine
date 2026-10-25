"""english-audio-aggregation-v4: window share decides dominance, cited time sets the floor."""

from __future__ import annotations

from engine.domain import TimeRange, TranscriptSegment
from engine.signals.classify import analysis_windows_from_run
from engine.signals.text_types import (
    CategoryCitation,
    ClassificationPayload,
    ClassificationResult,
    GenerationAttempt,
    SentimentValue,
)
from engine.signals.topic_sufficiency import evaluate_topic_policy_v3, evaluate_topic_policy_v4
from engine.standards import load_taxonomy
from engine.transcribe.types import TranscriptionRun, WindowResult

WATCHES, WEARABLES, BASEBALL, BASKETBALL, SPORTS = "581", "637", "545", "547", "483"
COOKING, TRAVEL = "216", "653"


def _range(start: float, end: float) -> TimeRange:
    return TimeRange(start_seconds=start, end_seconds=end)


def _source_window(index: int) -> WindowResult:
    """A 30 s window of continuous English speech in three 10 s segments."""
    start = index * 30
    segments = tuple(
        TranscriptSegment(time_range=_range(start + offset, start + offset + 10),
                          text="English topic", confidence=0.9, language="en")
        for offset in (0, 10, 20)
    )
    return WindowResult(time_range=_range(start, start + 30), segments=segments, language="en")


def _result(window, label) -> ClassificationResult:
    """``label`` is None (abstain) or (category, number of cited segments)."""
    attempts = (GenerationAttempt(attempt=1, stage="primary", prompt_sha256="0" * 64),)
    if label is None:
        return ClassificationResult(
            time_range=window.time_range, payload=None, attempts=attempts,
            error_code="INSUFFICIENT_TOPIC_EVIDENCE",
            policy_version="english-audio-topic-sufficiency-v1",
        )
    category, cited = label
    return ClassificationResult(
        time_range=window.time_range,
        payload=ClassificationPayload(
            primary_category=category, secondary_categories=(), keywords=(), entities=(),
            sentiment=SentimentValue(label="neutral", score=0.0),
        ),
        attempts=attempts,
        category_citations=(CategoryCitation(category_id=category, segment_indices=tuple(range(cited))),),
        policy_version="english-audio-topic-sufficiency-v1",
    )


def decide(labels: list, policy=evaluate_topic_policy_v4):
    """Each label is one 90 s analysis window of nine 10 s speech segments."""
    run = TranscriptionRun(windows=tuple(_source_window(i) for i in range(3 * len(labels))), status="ready")
    analysis = analysis_windows_from_run(run)
    assert len(analysis) == len(labels)
    results = tuple(_result(window, label) for window, label in zip(analysis, labels))
    return policy(run, results, audio_seconds=90.0 * len(labels), taxonomy=load_taxonomy())


def test_consistent_topic_with_partial_citations_is_single():
    # The Tissot shape: 5 of 6 windows agree, but each cites only 4 of 9 segments.
    labels = [(WATCHES, 4)] * 3 + [(WEARABLES, 4)] + [(WATCHES, 4)] * 2
    assert decide(labels, evaluate_topic_policy_v3).outcome == "insufficient"

    decision = decide(labels)
    assert decision.policy_version == "english-audio-aggregation-v4"
    assert (decision.outcome, decision.primary_category, decision.topic_structure) == (
        "eligible", WATCHES, "single")
    assert dict(decision.primary_window_support_seconds) == {WATCHES: 450.0, WEARABLES: 90.0}
    assert decision.selected_support_seconds == ((WATCHES, 200.0), (WEARABLES, 40.0))


def test_sparse_citations_stay_below_the_audio_floor():
    # Every window agrees, but one cited segment per window is only 11% of the audio.
    decision = decide([(WATCHES, 1)] * 6)
    assert decision.outcome == "insufficient"
    assert dict(decision.primary_window_support_seconds) == {WATCHES: 540.0}


def test_window_share_must_exceed_half():
    assert decide([(WATCHES, 9), (WATCHES, 9), None, None]).outcome == "insufficient"
    assert decide([(WATCHES, 9), (WATCHES, 9), (WATCHES, 9), None]).outcome == "eligible"


def test_cited_audio_floor_is_inclusive():
    # 3 windows x 3 cited segments = 90 s, exactly 25% of 360 s.
    assert decide([(WATCHES, 3)] * 3 + [None]).outcome == "eligible"
    assert decide([(WATCHES, 3), (WATCHES, 3), (WATCHES, 2), None]).outcome == "insufficient"


def test_sibling_windows_roll_up_by_window_share():
    decision = decide([(BASEBALL, 4), (BASEBALL, 4), (BASKETBALL, 4), (BASKETBALL, 4)])
    assert (decision.topic_structure, decision.primary_category) == ("parent_rollup", SPORTS)


def test_unrelated_majors_use_window_share():
    decision = decide([(COOKING, 4), (COOKING, 4), (TRAVEL, 4), (TRAVEL, 4)])
    assert decision.topic_structure == "multi_topic"
    assert set(decision.secondary_categories) | {decision.primary_category} == {COOKING, TRAVEL}


def test_v3_decisions_do_not_serialize_the_v4_field():
    decision = decide([(WATCHES, 9)] * 2, evaluate_topic_policy_v3)
    assert "primary_window_support_seconds" not in decision.model_dump(mode="json")


def test_path_hierarchy_keeps_a_parent_and_its_child_out_of_one_result():
    # Regression: Horse Racing's parent_id skips Equine Sports, so
    # aggregation published Equine Sports + Horse Racing together.
    from engine.standards import path_ancestors

    assert path_ancestors(load_taxonomy())["497"] == ("496", SPORTS)
    decision = decide([("496", 9), ("496", 9), ("497", 9)])
    assert decision.primary_category == "496"
    assert "497" not in decision.secondary_categories
