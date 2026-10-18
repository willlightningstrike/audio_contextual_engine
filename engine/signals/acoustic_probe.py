"""Pure complete-track acoustic evidence scheduling and duration accounting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engine.domain import TimeRange


def _finite_seconds(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("seconds must be numeric, finite, and nonnegative")
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("seconds must be finite and positive" if positive else "seconds must be finite and nonnegative")
    return float(value)


class AcousticObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    time_range: TimeRange
    foreground: Literal["yes", "no", "uncertain"]
    native_genre: str | None = None
    raw: str | None = None
    model_id: str

    @field_validator("time_range", mode="before")
    @classmethod
    def _strict_range(cls, value: object) -> TimeRange:
        if isinstance(value, TimeRange):
            start, end = value.start_seconds, value.end_seconds
        elif isinstance(value, Mapping):
            if set(value) != {"start_seconds", "end_seconds"}:
                raise ValueError("time range requires only start_seconds and end_seconds")
            start, end = value["start_seconds"], value["end_seconds"]
        else:
            raise ValueError("time range must be a TimeRange or endpoint mapping")
        start = _finite_seconds(start)
        end = _finite_seconds(end)
        if end <= start:
            raise ValueError("observation range must have positive duration")
        return TimeRange(start_seconds=start, end_seconds=end)

    @model_validator(mode="after")
    def _genre_requires_foreground(self) -> AcousticObservation:
        if self.native_genre is not None:
            if self.foreground != "yes" or not self.native_genre.strip():
                raise ValueError("native genre requires accepted foreground and a nonblank label")
        return self


class AcousticSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    duration_seconds: float = Field(gt=0)
    foreground_seconds: float = Field(ge=0)
    native_genre_seconds: dict[str, float]


def schedule_audio_windows(duration_seconds: float, *, window_seconds: float = 30.0) -> tuple[TimeRange, ...]:
    """Schedule every second of the track, including a shorter final window."""
    duration = _finite_seconds(duration_seconds, positive=True)
    window = _finite_seconds(window_seconds, positive=True)
    count = math.ceil(duration / window)
    return tuple(
        TimeRange(start_seconds=i * window, end_seconds=min(duration, (i + 1) * window))
        for i in range(count)
        if i * window < duration
    )


def summarize_observations(duration_seconds: float, observations: Sequence[AcousticObservation]) -> AcousticSummary:
    """Require exact nonoverlapping coverage and total accepted evidence seconds."""
    duration = _finite_seconds(duration_seconds, positive=True)
    ordered = sorted(observations, key=lambda item: item.time_range.start_seconds)
    cursor = 0.0
    accepted: list[float] = []
    genres: dict[str, list[float]] = {}
    for item in ordered:
        span = item.time_range
        start = _finite_seconds(span.start_seconds)
        end = _finite_seconds(span.end_seconds)
        if start != cursor or end <= start or end > duration:
            raise ValueError("observations must cover the track exactly without gaps or overlaps")
        cursor = end
        if item.foreground == "yes":
            seconds = end - start
            accepted.append(seconds)
            if item.native_genre is not None:
                genres.setdefault(item.native_genre, []).append(seconds)
    if cursor != duration:
        raise ValueError("observations must cover the complete track")
    return AcousticSummary(
        duration_seconds=duration,
        foreground_seconds=math.fsum(accepted),
        native_genre_seconds={label: math.fsum(seconds) for label, seconds in genres.items()},
    )
