from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest
from pydantic import ValidationError

from engine.domain import JobStage
from engine.events import Progress


def test_progress_never_moves_backwards():
    current = Progress(stage=JobStage.TRANSCRIBING, percent=42.0, message="window 2")
    later = current.advance(stage=JobStage.TRANSCRIBING, percent=41.0, message="window 3")
    assert later.percent == 42.0


def test_progress_advance_increases_revision_and_is_immutable():
    current = Progress(stage=JobStage.QUEUED, percent=0, message="waiting")
    later = current.advance(stage=JobStage.EXTRACTING_AUDIO, percent=5, message="extracting")

    assert later.revision == current.revision + 1
    with pytest.raises((ValidationError, FrozenInstanceError, TypeError)):
        current.percent = 99  # type: ignore[misc]


@pytest.mark.parametrize("percent", [-0.01, 100.01])
def test_progress_rejects_out_of_bounds_percentages(percent: float):
    with pytest.raises(ValidationError):
        Progress(stage=JobStage.QUEUED, percent=percent, message="waiting")


def test_progress_advance_bounds_a_pipeline_supplied_percentage():
    current = Progress(stage=JobStage.QUEUED, percent=5, message="waiting")
    later = current.advance(stage=JobStage.TRANSCRIBING, percent=500, message="window 1")
    assert later.percent == 100


@pytest.mark.parametrize("percent", [math.nan, math.inf, -math.inf])
def test_progress_rejects_non_finite_percentages(percent: float):
    with pytest.raises(ValidationError):
        Progress(stage=JobStage.QUEUED, percent=percent, message="waiting")


@pytest.mark.parametrize("percent", [math.nan, math.inf, -math.inf])
def test_progress_advance_rejects_non_finite_percentages(percent: float):
    current = Progress(stage=JobStage.QUEUED, percent=0, message="waiting")
    with pytest.raises(ValueError, match="finite"):
        current.advance(stage=JobStage.TRANSCRIBING, percent=percent, message="window 1")
