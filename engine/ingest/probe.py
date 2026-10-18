"""Safe, audio-focused ffprobe parsing and stream selection."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import AudioStream, ContainerKind, SourceMetadata


class ProbeError(ValueError):
    """Base error for unusable ffprobe results."""


class InvalidProbe(ProbeError):
    """ffprobe did not provide the fields needed for audio ingestion."""


class NoAudioStream(ProbeError):
    """The source has no selectable audio stream."""


class RequestedStreamInvalid(ProbeError):
    """The requested stream is absent or is not audio."""


@dataclass(frozen=True)
class MediaProbe:
    container_kind: ContainerKind
    audio_streams: tuple[AudioStream, ...]
    default_audio_stream: int | None
    duration_seconds: float
    source_metadata: SourceMetadata

    @property
    def selected_audio_stream(self) -> int:
        return choose_audio_stream(self, None).index


def _as_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidProbe(f"ffprobe {field} must be an object")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise InvalidProbe(f"ffprobe {field} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidProbe(f"ffprobe {field} must be a finite non-negative number") from error
    if not math.isfinite(number) or number < 0:
        raise InvalidProbe(f"ffprobe {field} must be a finite non-negative number")
    return number


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidProbe(f"ffprobe {field} must be a positive integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.isascii() and value.isdecimal():
        number = int(value)
    else:
        raise InvalidProbe(f"ffprobe {field} must be a positive integer")
    if number < 1:
        raise InvalidProbe(f"ffprobe {field} must be a positive integer")
    return number


def parse_probe(payload: Mapping[str, object]) -> MediaProbe:
    """Convert ffprobe JSON into the narrow metadata needed by the pipeline."""
    format_data = _as_mapping(payload.get("format"), "format")
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, Sequence) or isinstance(raw_streams, (str, bytes)):
        raise InvalidProbe("ffprobe streams must be an array")

    duration_seconds = _positive_float(format_data.get("duration"), "format.duration")
    tags = format_data.get("tags", {})
    tags = _as_mapping(tags, "format.tags")
    title = tags.get("title")
    if title is not None and not isinstance(title, str):
        raise InvalidProbe("ffprobe format.tags.title must be a string")

    audio_streams: list[AudioStream] = []
    default_audio_stream: int | None = None
    has_video = False
    for raw_stream in raw_streams:
        stream = _as_mapping(raw_stream, "stream")
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            disposition = _as_mapping(stream.get("disposition", {}), "stream.disposition")
            if disposition.get("attached_pic") != 1:
                has_video = True
        if codec_type != "audio":
            continue
        index = stream.get("index")
        codec_name = stream.get("codec_name")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or not isinstance(codec_name, str) or not codec_name:
            raise InvalidProbe("ffprobe audio stream must have a non-negative index and codec")
        if any(audio.index == index for audio in audio_streams):
            raise InvalidProbe("ffprobe audio stream indexes must be unique")
        stream_tags = _as_mapping(stream.get("tags", {}), "stream.tags")
        language = stream_tags.get("language")
        if language is not None and not isinstance(language, str):
            raise InvalidProbe("ffprobe stream.tags.language must be a string")
        audio_streams.append(
            AudioStream(
                index=index,
                codec_name=codec_name,
                language=language,
                channels=_optional_positive_int(stream.get("channels"), "stream.channels"),
                sample_rate=_optional_positive_int(stream.get("sample_rate"), "stream.sample_rate"),
            )
        )
        disposition = _as_mapping(stream.get("disposition", {}), "stream.disposition")
        if disposition.get("default") == 1 and default_audio_stream is None:
            default_audio_stream = index

    return MediaProbe(
        container_kind=ContainerKind.VIDEO if has_video else ContainerKind.AUDIO if audio_streams else ContainerKind.OTHER,
        audio_streams=tuple(audio_streams),
        default_audio_stream=default_audio_stream,
        duration_seconds=duration_seconds,
        source_metadata=SourceMetadata(title=title, duration_seconds=duration_seconds),
    )


def choose_audio_stream(probe: MediaProbe, requested_stream: int | None) -> AudioStream:
    if not probe.audio_streams:
        raise NoAudioStream("source has no audio streams")
    if requested_stream is not None:
        if isinstance(requested_stream, bool) or not isinstance(requested_stream, int) or requested_stream < 0:
            raise RequestedStreamInvalid("requested stream must be a non-negative integer")
        for stream in probe.audio_streams:
            if stream.index == requested_stream:
                return stream
        raise RequestedStreamInvalid("requested stream does not exist or is not audio")
    if probe.default_audio_stream is not None:
        return next(stream for stream in probe.audio_streams if stream.index == probe.default_audio_stream)
    return probe.audio_streams[0]


ProbeRunner = Callable[..., object]


class PopenFactory(Protocol):
    def __call__(self, args: list[str], **kwargs: object) -> subprocess.Popen[str]: ...


def _terminate_and_reap(process: subprocess.Popen[str]) -> None:
    """Best-effort stop with an explicit wait even if pipe decoding fails."""
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.communicate(timeout=0.1)
    except (OSError, UnicodeError, subprocess.SubprocessError):
        pass
    try:
        process.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait()
        except (AttributeError, OSError, subprocess.SubprocessError):
            pass
    except (AttributeError, OSError, subprocess.SubprocessError):
        pass


def _cancel_probe(process: subprocess.Popen[str]) -> None:
    _terminate_and_reap(process)


def _cleanup_failed_probe(process: subprocess.Popen[str]) -> None:
    """Best-effort process cleanup that never hides the probe's public error."""
    try:
        _terminate_and_reap(process)
    except (AttributeError, OSError, UnicodeError, subprocess.SubprocessError, ProcessLookupError):
        pass


def _parse_stdout(stdout: object) -> MediaProbe:
    if not isinstance(stdout, str):
        raise InvalidProbe("ffprobe stdout must be decoded text")
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise InvalidProbe("ffprobe failed to return valid JSON") from error
    if not isinstance(payload, Mapping):
        raise InvalidProbe("ffprobe JSON root must be an object")
    return parse_probe(payload)


def probe_source(
    source: Path,
    *,
    runner: ProbeRunner = subprocess.run,
    ffprobe: str = "ffprobe",
    token: CancellationToken | None = None,
    popen_factory: PopenFactory = subprocess.Popen,
) -> MediaProbe:
    """Invoke ffprobe with an argv list, then parse its JSON result."""
    command = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)]
    if token is not None and runner is subprocess.run:
        token.raise_if_cancelled()
        process: subprocess.Popen[str] | None = None
        try:
            process = popen_factory(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, shell=False)
            while True:
                try:
                    stdout, _ = process.communicate(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    if token.cancelled:
                        _cancel_probe(process)
                        raise JobCancelled
            token.raise_if_cancelled()
            if process.returncode != 0:
                raise InvalidProbe("ffprobe exited unsuccessfully")
        except JobCancelled:
            raise
        except InvalidProbe:
            raise
        except (subprocess.SubprocessError, OSError, UnicodeError) as error:
            if process is not None:
                _cleanup_failed_probe(process)
            raise InvalidProbe("ffprobe failed to return valid JSON") from error
        return _parse_stdout(stdout)
    try:
        result = runner(command, capture_output=True, text=True, check=True, shell=False)
        stdout = getattr(result, "stdout", None)
    except (subprocess.SubprocessError, OSError, TypeError) as error:
        raise InvalidProbe("ffprobe failed to return valid JSON") from error
    return _parse_stdout(stdout)
