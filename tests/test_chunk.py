from __future__ import annotations

import math

import pytest

from engine.domain import TimeRange, TranscriptSegment
from engine.transcribe.chunk import chunk_transcript


def _segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        time_range=TimeRange(start_seconds=start, end_seconds=end),
        text=text,
        confidence=1,
        language="en",
    )


def test_empty_has_no_analysis_window():
    assert chunk_transcript(()) == ()


def test_three_thirty_second_segments_and_five_second_tail_make_ninety_plus_five():
    segments = (
        _segment(0, 30, "one"),
        _segment(30, 60, "two"),
        _segment(60, 90, "three"),
        _segment(90, 95, "tail"),
    )

    windows = chunk_transcript(segments)

    assert [window.segments for window in windows] == [segments[:3], segments[3:]]
    assert [
        (window.time_range.start_seconds, window.time_range.end_seconds)
        for window in windows
    ] == [(0, 90), (90, 95)]


def test_exactly_ninety_second_span_stays_in_one_window():
    segments = (_segment(10, 40, "one"), _segment(70, 100, "two"))

    windows = chunk_transcript(segments)

    assert len(windows) == 1
    assert windows[0].time_range == TimeRange(start_seconds=10, end_seconds=100)


def test_gap_counts_toward_target_span_without_fabricating_context():
    segments = (
        _segment(0, 10, "first"),
        _segment(85, 90, "still within"),
        _segment(100, 105, "new group"),
    )

    windows = chunk_transcript(segments)

    assert [window.segments for window in windows] == [segments[:2], segments[2:]]
    assert windows[0].time_range == TimeRange(start_seconds=0, end_seconds=90)
    assert windows[1].time_range == TimeRange(start_seconds=100, end_seconds=105)


def test_final_short_tail_is_retained_without_padding():
    tail = _segment(180, 181, "tail")

    windows = chunk_transcript((_segment(0, 30, "one"), _segment(90, 120, "two"), tail))

    assert windows[-1].segments == (tail,)
    assert windows[-1].time_range == tail.time_range


@pytest.mark.parametrize(
    "segments",
    [
        (_segment(10, 20, "later"), _segment(0, 10, "earlier")),
        (_segment(0, 20, "first"), _segment(19, 30, "overlap")),
        (_segment(0, 31, "too long"),),
    ],
)
def test_invalid_order_overlap_or_overlong_segment_is_rejected(segments):
    with pytest.raises(ValueError, match="ordered|overlap|30 seconds"):
        chunk_transcript(segments)


@pytest.mark.parametrize("target", [0, -1, math.inf, math.nan])
def test_target_must_be_positive_and_finite(target: float):
    with pytest.raises(ValueError, match="positive finite"):
        chunk_transcript((_segment(0, 1, "one"),), target_seconds=target)
