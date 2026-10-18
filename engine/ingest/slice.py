"""Create and materialize bounded audio-only ASR work units."""

from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path
from uuid import uuid4

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import PreparedMedia
from engine.domain import TimeRange
from engine.ingest.audio import PopenFactory, _cancel_process


def bounded_windows(duration_seconds: float) -> tuple[TimeRange, ...]:
    """Partition a positive finite duration into work units of at most 30 seconds."""
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise ValueError("duration must be positive finite seconds")
    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive finite seconds")

    windows: list[TimeRange] = []
    start = 0.0
    while start < duration:
        end = min(start + 30.0, duration)
        windows.append(TimeRange(start_seconds=start, end_seconds=end))
        start = end
    return tuple(windows)


def _seconds_argument(value: float) -> str:
    return format(value, ".15g")


def _validate_pcm16_mono_16khz(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as stream:
            valid = (
                stream.getnchannels() == 1
                and stream.getsampwidth() == 2
                and stream.getframerate() == 16_000
                and stream.getnframes() > 0
            )
    except (EOFError, OSError, wave.Error) as exc:
        raise RuntimeError("ffmpeg produced an invalid WAV slice") from exc
    if not valid:
        raise RuntimeError("ffmpeg produced an invalid WAV slice")


def slice_audio_window(
    media: PreparedMedia,
    window: TimeRange,
    work_dir: Path,
    token: CancellationToken,
    *,
    popen_factory: PopenFactory = subprocess.Popen,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Atomically materialize one bounded mono 16 kHz PCM16 WAV work unit."""
    duration = window.end_seconds - window.start_seconds
    if not math.isfinite(duration) or duration <= 0 or duration > 30:
        raise ValueError("slice window must be positive, finite, and at most 30 seconds")
    token.raise_if_cancelled()
    work_dir.mkdir(parents=True, exist_ok=True)
    start_ms = round(window.start_seconds * 1000)
    end_ms = round(window.end_seconds * 1000)
    output = work_dir / f"asr-{start_ms:012d}-{end_ms:012d}.wav"
    partial = work_dir / f".{output.stem}.{uuid4().hex}.partial.wav"
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-ss",
        _seconds_argument(window.start_seconds),
        "-i",
        str(media.audio_path),
        "-t",
        _seconds_argument(duration),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(partial),
    ]
    process = popen_factory(
        command,
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        while process.poll() is None:
            if token.cancelled:
                _cancel_process(process)
                raise JobCancelled
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                continue
        if process.wait() != 0:
            raise RuntimeError("ffmpeg audio slicing failed")
        _validate_pcm16_mono_16khz(partial)
        token.run_if_not_cancelled(lambda: partial.replace(output))
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output
