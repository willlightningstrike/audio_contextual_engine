"""Bounded MLX-Audio Whisper adapter; all values returned are MLX-free."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from numbers import Real
from pathlib import Path
from typing import Any
import wave
import warnings

from engine.cancel import CancellationToken
from engine.domain import TimeRange, TranscriptSegment
from engine.models import collect_mlx_evidence, require_resolved_local_directory


MAX_AUDIO_SECONDS = 30.0
TAIL_ANOMALY_SECONDS = 2.0
_WHISPER_PUNCTUATION = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"


class BlankAsrSegmentWarning(UserWarning):
    """An otherwise valid ASR segment contained no text."""


class NoUsableTranscriptError(ValueError):
    """ASR yielded no usable text; this does not establish vocal absence."""


class InvalidAsrTimestampError(ValueError):
    """ASR returned a timestamp that cannot form a valid local range."""


def _require_bounded_window(window: TimeRange) -> None:
    if window.end_seconds - window.start_seconds > MAX_AUDIO_SECONDS:
        raise ValueError("model audio windows must not exceed 30 seconds")


def _validated_bounded_wav(audio: Path) -> tuple[Path, float]:
    path = Path(audio).resolve(strict=True)
    if path.suffix.lower() != ".wav":
        raise ValueError("model audio input must be a normalized WAV")
    try:
        with wave.open(str(path), "rb") as stream:
            duration = stream.getnframes() / stream.getframerate()
    except (wave.Error, ZeroDivisionError) as exc:
        raise ValueError("model audio input must be a readable WAV") from exc
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError("model audio windows must not exceed 30 seconds")
    return path, duration


def _require_bounded_wav(audio: Path) -> Path:
    path, _ = _validated_bounded_wav(audio)
    return path


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("invalid ASR word timestamps")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("invalid ASR word timestamps")
    return converted


def _word_anomaly_score(word: Mapping[str, Any]) -> float:
    probability = _finite_number(word.get("probability", 0.0))
    start = _finite_number(word.get("start"))
    end = _finite_number(word.get("end"))
    if end < start:
        raise ValueError("invalid ASR word timestamps")
    duration = end - start
    score = 0.0
    if probability < 0.15:
        score += 1.0
    if duration < 0.133:
        score += (0.133 - duration) * 15
    if duration > 2.0:
        score += duration - 2.0
    return score


def _is_segment_anomaly(segment: Mapping[str, Any]) -> bool:
    raw_words = segment.get("words")
    if raw_words in (None, []):
        return False
    if not isinstance(raw_words, list):
        raise ValueError("invalid ASR word timestamps")
    words: list[Mapping[str, Any]] = []
    for word in raw_words:
        if not isinstance(word, Mapping):
            raise ValueError("invalid ASR word timestamps")
        text = word.get("word")
        if not isinstance(text, str):
            raise ValueError("invalid ASR word timestamps")
        if text not in _WHISPER_PUNCTUATION:
            words.append(word)
    scored_words = words[:8]
    score = sum(_word_anomaly_score(word) for word in scored_words)
    return bool(words) and (score >= 3 or score + 0.01 >= len(scored_words))


def _filter_terminal_word_anomalies(
    segments: list[Any], audio_duration: float
) -> list[Any]:
    nonblank_indices = [
        index
        for index, segment in enumerate(segments)
        if isinstance(segment, Mapping)
        and isinstance(segment.get("text"), str)
        and segment["text"].strip()
    ]
    discarded: set[int] = set()
    for index in reversed(nonblank_indices):
        segment = segments[index]
        try:
            start = _finite_number(segment.get("start"))
            end = _finite_number(segment.get("end"))
        except ValueError as exc:
            raise InvalidAsrTimestampError("invalid ASR segment timestamps") from exc
        if (
            index == nonblank_indices[-1]
            and len(nonblank_indices) > 1
            and start > audio_duration
            and end >= start
            and segment.get("words") in (None, [])
            and all(
                isinstance(later, Mapping)
                and isinstance(later.get("text"), str)
                and not later["text"].strip()
                for later in segments[index + 1 :]
            )
        ):
            discarded.add(index)
            continue
        in_tail = 0 <= audio_duration - end < TAIL_ANOMALY_SECONDS
        if not in_tail or not _is_segment_anomaly(segment):
            break
        discarded.add(index)
    return [segment for index, segment in enumerate(segments) if index not in discarded]


def _default_load(path: Path) -> Any:
    from mlx_audio.stt import load

    return load(str(path), lazy=False, strict=True)


def _default_language_inputs(path: Path, n_mels: int) -> Any:
    # This version-pinned helper boundary is deliberately confined to this adapter.
    from mlx_audio.stt.utils import load_audio
    from mlx_audio.stt.models.whisper.audio import log_mel_spectrogram, pad_or_trim

    return log_mel_spectrogram(pad_or_trim(load_audio(str(path))), n_mels=n_mels)


class MlxAsrAdapter:
    def __init__(
        self,
        snapshot_path: Path,
        *,
        load_fn: Callable[[Path], Any] = _default_load,
        language_inputs_fn: Callable[[Path, int], Any] = _default_language_inputs,
    ) -> None:
        self._snapshot_path = require_resolved_local_directory(snapshot_path)
        self._load_fn = load_fn
        self._language_inputs_fn = language_inputs_fn
        self._model: Any | None = None
        self._last_language_tensor: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_fn(self._snapshot_path)
        return self._model

    def transcribe_window(
        self, audio: Path, window: TimeRange, language: str, token: CancellationToken
    ) -> list[TranscriptSegment]:
        _require_bounded_window(window)
        audio_path, audio_duration = _validated_bounded_wav(audio)
        token.raise_if_cancelled()
        result = self.model.generate(
            str(audio_path),
            language=language,
            task="transcribe",
            return_timestamps=True,
            word_timestamps=True,
            verbose=False,
        )
        token.raise_if_cancelled()
        segments = result.get("segments") if isinstance(result, Mapping) else getattr(result, "segments", None)
        if not isinstance(segments, list):
            raise ValueError("invalid ASR output: segments must be a list")
        segments = _filter_terminal_word_anomalies(segments, audio_duration)
        converted: list[TranscriptSegment] = []
        for index, item in enumerate(segments):
            try:
                if not isinstance(item, Mapping):
                    raise TypeError
                text = item["text"]
                if not isinstance(text, str):
                    raise TypeError
                raw_start = item["start"]
                raw_end = item["end"]
                if (
                    isinstance(raw_start, bool)
                    or isinstance(raw_end, bool)
                    or not isinstance(raw_start, Real)
                    or not isinstance(raw_end, Real)
                ):
                    raise InvalidAsrTimestampError("invalid ASR segment timestamps")
                try:
                    start = float(raw_start)
                    end = float(raw_end)
                except (OverflowError, ValueError) as exc:
                    raise InvalidAsrTimestampError(
                        "invalid ASR segment timestamps"
                    ) from exc
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                    raise InvalidAsrTimestampError("invalid ASR segment timestamps")
                try:
                    time_range = TimeRange(start_seconds=start, end_seconds=end)
                except ValueError as exc:
                    raise InvalidAsrTimestampError("invalid ASR segment timestamps") from exc
                if not text.strip():
                    warnings.warn(f"skipping blank ASR segment at index {index}", BlankAsrSegmentWarning, stacklevel=2)
                    continue
                if not any(character.isalnum() for character in text):
                    continue
                confidence = item.get("confidence", item.get("avg_logprob", 1.0))
                if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
                    confidence = 1.0
                converted.append(
                    TranscriptSegment(
                        time_range=time_range,
                        text=text.strip(),
                        confidence=float(confidence),
                        language=language,
                    )
                )
            except InvalidAsrTimestampError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid ASR segment") from exc
        if not converted:
            raise NoUsableTranscriptError("ASR produced no usable transcript; vocal presence is unknown")
        return converted

    def release(self) -> None:
        loaded = self._model is not None
        self._model = None
        self._last_language_tensor = None
        if loaded:
            from mlx import core as mx

            mx.clear_cache()

    def detect_language_probabilities(
        self, audio: Path, window: TimeRange, token: CancellationToken
    ) -> dict[str, float]:
        _require_bounded_window(window)
        audio_path = _require_bounded_wav(audio)
        token.raise_if_cancelled()
        # The real-model smoke test pins and exercises this narrow upstream call.
        model = self.model
        self._last_language_tensor = self._language_inputs_fn(audio_path, int(model.dims.n_mels))
        raw = model.detect_language(self._last_language_tensor)
        token.raise_if_cancelled()
        if isinstance(raw, tuple):
            raw = raw[1]
        if not isinstance(raw, Mapping):
            raise ValueError("invalid ASR language output")
        probabilities: dict[str, float] = {}
        for key, value in raw.items():
            if (
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, Real)
            ):
                raise ValueError("invalid ASR language probabilities")
            probability = float(value)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("invalid ASR language probabilities")
            probabilities[key] = probability
        if not probabilities:
            raise ValueError("invalid ASR language probabilities")
        return probabilities

    def runtime_evidence(self) -> dict[str, Any]:
        tensors = {} if self._last_language_tensor is None else {"language_mel": self._last_language_tensor}
        return collect_mlx_evidence(self.model, tensors)
