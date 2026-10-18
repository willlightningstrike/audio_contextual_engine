from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.ingest.slice import bounded_windows, slice_audio_window


def _write_pcm16_mono(path: Path, *, frames: int = 16_000) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * frames)


class CompletedProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_asr_windows_are_bounded():
    windows = bounded_windows(95.0)

    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0, 30),
        (30, 60),
        (60, 90),
        (90, 95),
    ]


@pytest.mark.parametrize("duration", [0, -1, math.inf, -math.inf, math.nan])
def test_invalid_duration_cannot_create_a_work_window(duration: float):
    with pytest.raises(ValueError, match="positive finite"):
        bounded_windows(duration)


def test_exact_boundary_has_no_zero_length_final_window():
    windows = bounded_windows(60.0)

    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0, 30),
        (30, 60),
    ]


def test_slice_reads_only_prepared_audio_with_exact_pcm_window_command(tmp_path: Path):
    prepared_audio = tmp_path / "prepared audio.wav"
    _write_pcm16_mono(prepared_audio)
    media = SimpleNamespace(audio_path=prepared_audio)
    commands: list[tuple[list[str], dict[str, object]]] = []

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        commands.append((command, kwargs))
        _write_pcm16_mono(Path(command[-1]), frames=80_000)
        return CompletedProcess()

    output = slice_audio_window(
        media,
        TimeRange(start_seconds=30, end_seconds=35),
        tmp_path / "windows",
        CancellationToken(),
        popen_factory=popen,
        ffmpeg="ffmpeg-bin",
    )

    command, kwargs = commands[0]
    assert command == [
        "ffmpeg-bin",
        "-nostdin",
        "-y",
        "-ss",
        "30",
        "-i",
        str(prepared_audio),
        "-t",
        "5",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        command[-1],
    ]
    assert kwargs["shell"] is False
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert output.parent == tmp_path / "windows"
    assert output.name.startswith("asr-")
    assert output.suffix == ".wav"
    assert output.exists()
    assert not list(output.parent.glob("*.partial.wav"))


def test_each_attempt_uses_a_unique_partial_file(tmp_path: Path):
    prepared_audio = tmp_path / "audio.wav"
    _write_pcm16_mono(prepared_audio)
    media = SimpleNamespace(audio_path=prepared_audio)
    partials: list[Path] = []

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        del kwargs
        partial = Path(command[-1])
        partials.append(partial)
        _write_pcm16_mono(partial)
        return CompletedProcess()

    window = TimeRange(start_seconds=0, end_seconds=1)
    slice_audio_window(media, window, tmp_path / "windows", CancellationToken(), popen_factory=popen)
    slice_audio_window(media, window, tmp_path / "windows", CancellationToken(), popen_factory=popen)

    assert len(set(partials)) == 2


@pytest.mark.parametrize("failure", ["nonzero", "invalid"])
def test_failed_slice_never_promotes_and_cleans_its_partial(tmp_path: Path, failure: str):
    prepared_audio = tmp_path / "audio.wav"
    _write_pcm16_mono(prepared_audio)
    media = SimpleNamespace(audio_path=prepared_audio)

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        del kwargs
        partial = Path(command[-1])
        partial.write_bytes(b"not a wav" if failure == "invalid" else b"partial")
        return CompletedProcess(returncode=1 if failure == "nonzero" else 0)

    with pytest.raises(RuntimeError, match="ffmpeg|WAV"):
        slice_audio_window(
            media,
            TimeRange(start_seconds=0, end_seconds=1),
            tmp_path / "windows",
            CancellationToken(),
            popen_factory=popen,
        )

    assert not list((tmp_path / "windows").glob("asr-*.wav"))
    assert not list((tmp_path / "windows").glob("*.partial.wav"))


def test_cancellation_terminates_ffmpeg_cleans_current_partial_and_preserves_completed_slice(
    tmp_path: Path,
):
    prepared_audio = tmp_path / "audio.wav"
    _write_pcm16_mono(prepared_audio)
    media = SimpleNamespace(audio_path=prepared_audio)
    first = slice_audio_window(
        media,
        TimeRange(start_seconds=0, end_seconds=1),
        tmp_path / "windows",
        CancellationToken(),
        popen_factory=lambda command, **kwargs: (
            _write_pcm16_mono(Path(command[-1])) or CompletedProcess()
        ),
    )
    token = CancellationToken()

    class RunningProcess(CompletedProcess):
        def __init__(self) -> None:
            super().__init__()
            self.wait_calls = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                token.cancel()
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return 0

    holder: dict[str, RunningProcess] = {}

    def popen(command: list[str], **kwargs: object) -> RunningProcess:
        del kwargs
        Path(command[-1]).write_bytes(b"partial")
        holder["process"] = RunningProcess()
        return holder["process"]

    with pytest.raises(JobCancelled):
        slice_audio_window(
            media,
            TimeRange(start_seconds=1, end_seconds=2),
            tmp_path / "windows",
            token,
            popen_factory=popen,
        )

    assert first.exists()
    assert holder["process"].terminated is True
    assert holder["process"].killed is False
    assert len(list((tmp_path / "windows").glob("asr-*.wav"))) == 1
    assert not list((tmp_path / "windows").glob("*.partial.wav"))
