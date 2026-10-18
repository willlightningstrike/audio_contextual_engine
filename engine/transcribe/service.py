"""Conservative English-only transcription over bounded audio work units."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import PreparedMedia, TimeRange, TranscriptSegment
from engine.ingest.slice import bounded_windows, slice_audio_window
from engine.models.asr import InvalidAsrTimestampError, NoUsableTranscriptError
from engine.signals.protocol import VocalScreener
from engine.transcribe.protocol import Transcriber
from engine.transcribe.types import TimestampCorrection, TranscriptionRun, WindowResult


VOCAL_PRESENCE_UNCERTAIN = "VOCAL_PRESENCE_UNCERTAIN"
NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE = "NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE"
NO_USABLE_TRANSCRIPT = "NO_USABLE_TRANSCRIPT"
MODEL_FAILURE = "MODEL_FAILURE"

_AUDIO_SLICING = "audio_slicing"
_VOCAL_SCREENING = "vocal_screening"
_LANGUAGE_DETECTION = "language_detection"
_TRANSCRIPTION = "transcription"
_TIMESTAMP_VALIDATION = "timestamp_validation"

_AUDIO_SLICE_FAILED = "audio_slice_failed"
_MODEL_CALL_FAILED = "model_call_failed"
_INVALID_LANGUAGE_PROBABILITIES = "invalid_language_probabilities"
_INVALID_TIMESTAMPS = "invalid_timestamps"
_NO_USABLE_TRANSCRIPT_REASON = "no_usable_transcript"


def _validated_language_probabilities(probabilities: object) -> dict[str, float]:
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("invalid ASR language probabilities")
    validated: dict[str, float] = {}
    for language, probability in probabilities.items():
        if (
            not isinstance(language, str)
            or not language
            or isinstance(probability, bool)
            or not isinstance(probability, (int, float))
        ):
            raise ValueError("invalid ASR language probabilities")
        score = float(probability)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid ASR language probabilities")
        validated[language] = score
    return validated


def _top_language(probabilities: object) -> str | None:
    try:
        validated = _validated_language_probabilities(probabilities)
    except ValueError:
        return None
    maximum = max(validated.values())
    leaders = [language for language, score in validated.items() if score == maximum]
    return leaders[0] if len(leaders) == 1 else None


def _rebase_segments(
    segments: object, source_window: TimeRange
) -> tuple[TranscriptSegment, ...]:
    converted, _ = _rebase_segments_with_corrections(segments, source_window)
    return converted


def _rebase_segments_with_corrections(
    segments: object, source_window: TimeRange
) -> tuple[tuple[TranscriptSegment, ...], tuple[TimestampCorrection, ...]]:
    if not isinstance(segments, (list, tuple)):
        raise ValueError("ASR segments must be a sequence")
    local_duration = source_window.end_seconds - source_window.start_seconds
    converted: list[TranscriptSegment] = []
    corrections: list[TimestampCorrection] = []
    previous_end = 0.0
    for index, segment in enumerate(segments):
        local = segment.time_range
        start = float(local.start_seconds)
        end = float(local.end_seconds)
        original_start = start
        if (
            math.isfinite(start)
            and math.isfinite(previous_end)
            and start > 0
            and previous_end > 0
            and start < previous_end
            and start == math.nextafter(previous_end, -math.inf)
            and end > previous_end
        ):
            start = previous_end
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end < start
            or end > local_duration
            or start < previous_end
            or end - start > 30
        ):
            raise ValueError("ASR segment timestamps are invalid for the bounded window")
        if start != original_start:
            corrections.append(
                TimestampCorrection(
                    segment_index=index,
                    original_start_seconds=original_start,
                    corrected_start_seconds=start,
                    local_end_seconds=end,
                )
            )
        converted.append(
            TranscriptSegment(
                time_range=TimeRange(
                    start_seconds=source_window.start_seconds + start,
                    end_seconds=source_window.start_seconds + end,
                ),
                text=segment.text,
                confidence=segment.confidence,
                language=segment.language,
            )
        )
        previous_end = end
    return tuple(converted), tuple(corrections)


def _persist_result(
    result: WindowResult,
    *,
    completed: int,
    total: int,
    token: CancellationToken,
    checkpoint: Callable[[WindowResult], Any],
    on_progress: Callable[[int, int], Any],
) -> None:
    token.raise_if_cancelled()
    checkpoint(result)
    on_progress(completed, total)
    token.raise_if_cancelled()


def transcribe_media(
    media: PreparedMedia,
    *,
    transcriber: Transcriber,
    screener: VocalScreener,
    work_dir: Path,
    token: CancellationToken,
    checkpoint: Callable[[WindowResult], Any],
    on_progress: Callable[[int, int], Any],
    non_english_stop_seconds: float = 0.0,
) -> TranscriptionRun:
    """Screen, language-gate, transcribe, and checkpoint each bounded window.

    Processing stops once non-English windows exceed ``non_english_stop_seconds``.
    The default stops at the first such window, as earlier versions did.
    """
    if (
        isinstance(non_english_stop_seconds, bool)
        or not isinstance(non_english_stop_seconds, (int, float))
        or not math.isfinite(non_english_stop_seconds)
        or non_english_stop_seconds < 0
    ):
        raise ValueError("non_english_stop_seconds must be finite and non-negative")
    source_windows = bounded_windows(media.duration_seconds)
    completed: list[WindowResult] = []
    total = len(source_windows)
    non_english_seconds = 0.0

    for index, source_window in enumerate(source_windows, start=1):
        token.raise_if_cancelled()
        stop_for_language = False
        failure_stage = _AUDIO_SLICING
        try:
            audio = slice_audio_window(media, source_window, Path(work_dir) / "asr-windows", token)
            failure_stage = _VOCAL_SCREENING
            screen = screener.screen(audio, token)
            if not screen.admitted:
                result = WindowResult(
                    time_range=source_window,
                    error_code=VOCAL_PRESENCE_UNCERTAIN,
                    failure_stage=_VOCAL_SCREENING,
                    failure_reason="vocal_presence_uncertain",
                )
            else:
                duration = source_window.end_seconds - source_window.start_seconds
                local_window = TimeRange(start_seconds=0, end_seconds=duration)
                invalid_language_probabilities = False
                try:
                    failure_stage = _LANGUAGE_DETECTION
                    probabilities = transcriber.detect_language_probabilities(
                        audio, local_window, token
                    )
                    language = _top_language(
                        _validated_language_probabilities(probabilities)
                    )
                except ValueError:
                    language = None
                    invalid_language_probabilities = True
                if language != "en":
                    result = WindowResult(
                        time_range=source_window,
                        language=language,
                        error_code=NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE,
                        failure_stage=_LANGUAGE_DETECTION,
                        failure_reason=(
                            _INVALID_LANGUAGE_PROBABILITIES
                            if invalid_language_probabilities
                            else "non_english_or_unconfirmed_language"
                        ),
                    )
                    non_english_seconds += source_window.end_seconds - source_window.start_seconds
                    stop_for_language = non_english_seconds > non_english_stop_seconds
                else:
                    failure_stage = _TRANSCRIPTION
                    raw_segments = transcriber.transcribe_window(
                        audio, local_window, "en", token
                    )
                    try:
                        segments, timestamp_corrections = (
                            _rebase_segments_with_corrections(raw_segments, source_window)
                        )
                    except ValueError:
                        result = WindowResult(
                            time_range=source_window,
                            language="en",
                            error_code=MODEL_FAILURE,
                            failure_stage=_TIMESTAMP_VALIDATION,
                            failure_reason=_INVALID_TIMESTAMPS,
                        )
                        segments = None
                    if segments == ():
                        result = WindowResult(
                            time_range=source_window,
                            language="en",
                            error_code=NO_USABLE_TRANSCRIPT,
                            failure_stage=_TRANSCRIPTION,
                            failure_reason=_NO_USABLE_TRANSCRIPT_REASON,
                        )
                    elif segments is not None:
                        result = WindowResult(
                            time_range=source_window,
                            segments=segments,
                            language="en",
                            timestamp_corrections=timestamp_corrections,
                        )
        except JobCancelled:
            raise
        except InvalidAsrTimestampError:
            result = WindowResult(
                time_range=source_window,
                language="en",
                error_code=MODEL_FAILURE,
                failure_stage=_TIMESTAMP_VALIDATION,
                failure_reason=_INVALID_TIMESTAMPS,
            )
        except NoUsableTranscriptError:
            result = WindowResult(
                time_range=source_window,
                language="en",
                error_code=NO_USABLE_TRANSCRIPT,
                failure_stage=_TRANSCRIPTION,
                failure_reason=_NO_USABLE_TRANSCRIPT_REASON,
            )
        except Exception:
            reason = (
                _AUDIO_SLICE_FAILED
                if failure_stage == _AUDIO_SLICING
                else _MODEL_CALL_FAILED
            )
            result = WindowResult(
                time_range=source_window,
                error_code=MODEL_FAILURE,
                failure_stage=failure_stage,
                failure_reason=reason,
            )

        _persist_result(
            result,
            completed=index,
            total=total,
            token=token,
            checkpoint=checkpoint,
            on_progress=on_progress,
        )
        completed.append(result)
        if stop_for_language:
            break

    has_segments = any(result.segments for result in completed)
    all_succeeded = len(completed) == total and all(result.error_code is None for result in completed)
    status = "ready" if all_succeeded else "partial" if has_segments else "failed"
    return TranscriptionRun(windows=tuple(completed), status=status)
