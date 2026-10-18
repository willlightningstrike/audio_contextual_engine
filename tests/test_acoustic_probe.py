import math

import pytest

from engine.domain import TimeRange
from engine.signals.acoustic_probe import AcousticObservation, schedule_audio_windows, summarize_observations


def observation(start, end, foreground="yes", native_genre=None):
    return AcousticObservation(time_range={"start_seconds": start, "end_seconds": end}, foreground=foreground, native_genre=native_genre, model_id="test")


def test_one_hour_schedule_has_complete_coverage():
    windows = schedule_audio_windows(3600.0)
    assert len(windows) == 120
    assert windows[0].start_seconds == 0
    assert windows[-1].end_seconds == 3600
    assert all(a.end_seconds == b.start_seconds for a, b in zip(windows, windows[1:]))


@pytest.mark.parametrize("duration,lengths", [(61, [30, 30, 1]), (30, [30]), (1, [1])])
def test_schedule_preserves_tail(duration, lengths):
    assert [w.end_seconds - w.start_seconds for w in schedule_audio_windows(duration)] == lengths


def test_float_rounding_does_not_schedule_an_empty_tail():
    duration = 0.1 + 0.2
    windows = schedule_audio_windows(duration, window_seconds=0.1)
    assert len(windows) == 3
    assert windows[-1].end_seconds == duration
    observations = tuple(
        AcousticObservation(time_range=span, foreground="yes", model_id="test")
        for span in windows
    )
    assert summarize_observations(duration, observations).foreground_seconds == duration


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, True, False, "30", None])
def test_invalid_duration_or_window_is_rejected(value):
    with pytest.raises(ValueError):
        schedule_audio_windows(value)
    with pytest.raises(ValueError):
        schedule_audio_windows(30, window_seconds=value)
    with pytest.raises(ValueError):
        summarize_observations(value, ())


@pytest.mark.parametrize("accepted,at_boundary", [(70, True), (69.999, False)])
def test_ledger_preserves_policy_boundary_without_classifying(accepted, at_boundary):
    summary = summarize_observations(100, (observation(0, accepted, native_genre="rock"), observation(accepted, 100, "no")))
    assert summary.duration_seconds == 100
    assert summary.foreground_seconds == accepted
    assert (summary.foreground_seconds / summary.duration_seconds >= 0.70) is at_boundary
    assert summary.native_genre_seconds == {"rock": accepted}


def test_uncertain_and_unlabelled_foreground_ledger():
    summary = summarize_observations(100, (observation(70, 100, "uncertain"), observation(0, 20, native_genre="rock"), observation(20, 40), observation(40, 70, native_genre="rock")))
    assert summary.foreground_seconds == 70
    assert summary.native_genre_seconds == {"rock": 50}


@pytest.mark.parametrize("ranges", [[], [(0, 50), (0, 50)], [(0, 60), (50, 100)], [(0, 40), (50, 100)], [(1, 100)], [(0, 99)], [(0, 101)]])
def test_incomplete_or_overlapping_coverage_is_rejected(ranges):
    with pytest.raises(ValueError):
        summarize_observations(100, tuple(observation(a, b) for a, b in ranges))


@pytest.mark.parametrize("start,end", [(True, 2), (0, True), ("0", 2), (0, "2"), (-1, 2), (0, 0), (2, 1), (math.nan, 2), (0, math.inf)])
def test_raw_endpoints_are_validated_before_coercion(start, end):
    with pytest.raises(ValueError):
        observation(start, end)


@pytest.mark.parametrize("foreground", ["no", "uncertain"])
def test_genre_requires_accepted_foreground(foreground):
    with pytest.raises(ValueError):
        observation(0, 30, foreground, "rock")


def test_strict_labels_and_frozen_observation():
    with pytest.raises(ValueError):
        observation(0, 30, native_genre=["rock", "pop"])
    with pytest.raises(ValueError):
        observation(0, 30, foreground="maybe")
    item = AcousticObservation(time_range=TimeRange(start_seconds=0, end_seconds=30), foreground="yes", model_id="test")
    with pytest.raises(ValueError):
        item.foreground = "no"
