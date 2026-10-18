"""Immutable MLX-free records for bounded transcription work."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from engine.domain import TimeRange, TranscriptSegment


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)


class ScreenResult(_FrozenModel):
    speech_max: float = Field(ge=0, le=1)
    singing_max: float = Field(ge=0, le=1)
    admitted: bool


class TimestampCorrection(_FrozenModel):
    segment_index: StrictInt = Field(ge=0)
    original_start_seconds: float
    corrected_start_seconds: float
    local_end_seconds: float


class WindowResult(_FrozenModel):
    time_range: TimeRange
    segments: tuple[TranscriptSegment, ...] = ()
    language: str | None = None
    error_code: str | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    attempt_count: StrictInt = Field(default=1, gt=0)
    timestamp_corrections: tuple[TimestampCorrection, ...] = ()


class TranscriptionRun(_FrozenModel):
    windows: tuple[WindowResult, ...]
    status: Literal["ready", "partial", "failed"]


class AnalysisWindow(_FrozenModel):
    time_range: TimeRange
    segments: tuple[TranscriptSegment, ...]
