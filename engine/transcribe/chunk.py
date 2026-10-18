"""Deterministically group validated transcript segments into analysis windows."""

from __future__ import annotations

import math
from collections.abc import Sequence

from engine.domain import TimeRange, TranscriptSegment
from engine.transcribe.types import AnalysisWindow


def chunk_transcript(
    segments: Sequence[TranscriptSegment], target_seconds: float = 90.0
) -> tuple[AnalysisWindow, ...]:
    """Group whole segments whose first-to-last span fits the target duration."""
    if (
        isinstance(target_seconds, bool)
        or not isinstance(target_seconds, (int, float))
        or not math.isfinite(float(target_seconds))
        or target_seconds <= 0
    ):
        raise ValueError("target_seconds must be positive finite seconds")
    if not segments:
        return ()

    validated: list[TranscriptSegment] = []
    previous_start = -1.0
    previous_end = -1.0
    for segment in segments:
        start = segment.time_range.start_seconds
        end = segment.time_range.end_seconds
        if end - start > 30:
            raise ValueError("transcript segments must not exceed 30 seconds")
        if start < previous_start:
            raise ValueError("transcript segments must be ordered")
        if start < previous_end:
            raise ValueError("transcript segments must not overlap")
        validated.append(segment)
        previous_start = start
        previous_end = end

    output: list[AnalysisWindow] = []
    current: list[TranscriptSegment] = []
    first_start = 0.0
    for segment in validated:
        if not current:
            current = [segment]
            first_start = segment.time_range.start_seconds
            continue
        if segment.time_range.end_seconds - first_start <= target_seconds:
            current.append(segment)
            continue
        output.append(
            AnalysisWindow(
                time_range=TimeRange(
                    start_seconds=current[0].time_range.start_seconds,
                    end_seconds=current[-1].time_range.end_seconds,
                ),
                segments=tuple(current),
            )
        )
        current = [segment]
        first_start = segment.time_range.start_seconds

    output.append(
        AnalysisWindow(
            time_range=TimeRange(
                start_seconds=current[0].time_range.start_seconds,
                end_seconds=current[-1].time_range.end_seconds,
            ),
            segments=tuple(current),
        )
    )
    return tuple(output)
