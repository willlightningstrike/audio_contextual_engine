from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import ContentContext, ContainerKind, TimeRange, TranscriptSegment
from engine.ingest.audio import prepare_audio
from engine.transcribe.service import transcribe_media
from engine.transcribe.types import ScreenResult


FIXTURES = Path(__file__).parent / "fixtures" / "ffprobe"


def write_pcm_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16_000)


@pytest.fixture
def ffmpeg_tools() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg tools required")


def probe_runner(command: list[str], **kwargs: object) -> object:
    del command, kwargs
    return type("Result", (), {"stdout": (FIXTURES / "video_two_audio.json").read_text()})()


class CompletedProcess:
    def __init__(self, output: Path, returncode: int = 0) -> None:
        self.output = output
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


def test_extracts_exact_selected_stream_atomically_and_keeps_space_path_as_argv(tmp_path: Path):
    commands: list[tuple[list[str], bool]] = []

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        commands.append((command, kwargs["shell"]))
        output = Path(command[-1])
        write_pcm_wav(output)
        return CompletedProcess(output)

    prepared = prepare_audio(
        Path("source with spaces.mp4"), tmp_path, 2, None, CancellationToken(), lambda update: None,
        probe_runner=probe_runner, popen_factory=popen, ffmpeg="ffmpeg-bin",
    )

    command, shell = commands[0]
    assert shell is False
    assert command == [
        "ffmpeg-bin", "-nostdin", "-y", "-i", "source with spaces.mp4", "-map", "0:2", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp_path / "audio.partial.wav"),
    ]
    assert prepared.audio_path == tmp_path / "audio.wav"
    with wave.open(str(prepared.audio_path), "rb") as stream:
        assert stream.readframes(16_000) == b"\0\0" * 16_000
    assert not (tmp_path / "audio.partial.wav").exists()
    assert prepared.content_context_hint is ContentContext.VIDEO


def test_nonzero_ffmpeg_removes_partial_without_promoting(tmp_path: Path):
    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        output = Path(command[-1])
        output.write_bytes(b"partial")
        return CompletedProcess(output, returncode=1)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, CancellationToken(), lambda update: None,
                      probe_runner=probe_runner, popen_factory=popen)

    assert not (tmp_path / "audio.partial.wav").exists()
    assert not (tmp_path / "audio.wav").exists()


def test_new_failed_preparation_removes_only_its_stale_audio_result(tmp_path: Path):
    (tmp_path / "audio.wav").write_bytes(b"old")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("keep")

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        Path(command[-1]).write_bytes(b"partial")
        return CompletedProcess(Path(command[-1]), returncode=1)

    with pytest.raises(RuntimeError):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, CancellationToken(), lambda update: None,
                      probe_runner=probe_runner, popen_factory=popen)

    assert not (tmp_path / "audio.wav").exists()
    assert unrelated.read_text() == "keep"


def test_invalid_duplicate_probe_never_launches_or_promotes(tmp_path: Path):
    (tmp_path / "audio.wav").write_bytes(b"old")
    launches = 0
    data = json.loads((FIXTURES / "video_two_audio.json").read_text())
    data["streams"][2]["index"] = 1

    def runner(command: list[str], **kwargs: object) -> object:
        return type("Result", (), {"stdout": json.dumps(data)})()

    def popen(*args: object, **kwargs: object) -> CompletedProcess:
        nonlocal launches
        launches += 1
        raise AssertionError("ffmpeg must not launch")

    with pytest.raises(Exception):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, CancellationToken(), lambda update: None,
                      probe_runner=runner, popen_factory=popen)
    assert launches == 0
    assert not (tmp_path / "audio.wav").exists()


def test_pre_cancelled_job_does_not_probe_or_launch(tmp_path: Path):
    token = CancellationToken()
    token.cancel()
    launched = False

    def runner(*args: object, **kwargs: object) -> object:
        raise AssertionError("probe must not launch")

    def popen(*args: object, **kwargs: object) -> CompletedProcess:
        nonlocal launched
        launched = True
        raise AssertionError("ffmpeg must not launch")

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, lambda update: None,
                      probe_runner=runner, popen_factory=popen)
    assert not launched


def test_pre_cancelled_job_has_no_side_effects_and_preserves_existing_audio(tmp_path: Path):
    token = CancellationToken()
    token.cancel()
    stale_audio = tmp_path / "audio.wav"
    stale_audio.write_bytes(b"existing")
    emitted: list[object] = []

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, emitted.append,
                      probe_runner=lambda *args, **kwargs: pytest.fail("probe must not launch"),
                      popen_factory=lambda *args, **kwargs: pytest.fail("ffmpeg must not launch"))

    assert stale_audio.read_bytes() == b"existing"
    assert emitted == []


def test_emit_cancellation_prevents_ffmpeg_launch_and_output(tmp_path: Path):
    token = CancellationToken()
    launches = 0

    def popen(*args: object, **kwargs: object) -> CompletedProcess:
        nonlocal launches
        launches += 1
        raise AssertionError("ffmpeg must not launch after listener cancellation")

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, lambda update: token.cancel(),
                      probe_runner=probe_runner, popen_factory=popen)

    assert launches == 0
    assert not (tmp_path / "audio.partial.wav").exists()
    assert not (tmp_path / "audio.wav").exists()


def test_cancellation_terminates_reaps_and_removes_incomplete_partial(tmp_path: Path):
    token = CancellationToken()

    class RunningProcess(CompletedProcess):
        def __init__(self, output: Path) -> None:
            super().__init__(output)
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
        output = Path(command[-1])
        output.write_bytes(b"partial")
        holder["process"] = RunningProcess(output)
        return holder["process"]

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, lambda update: None,
                      probe_runner=probe_runner, popen_factory=popen)

    process = holder["process"]
    assert process.terminated
    assert not process.killed
    assert process.wait_calls >= 2
    assert not (tmp_path / "audio.partial.wav").exists()
    assert not (tmp_path / "audio.wav").exists()


def test_pre_cancelled_preparation_preserves_a_stale_audio_result(tmp_path: Path):
    (tmp_path / "audio.wav").write_bytes(b"old")
    token = CancellationToken()
    token.cancel()

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, lambda update: None,
                      probe_runner=probe_runner, popen_factory=lambda *args, **kwargs: pytest.fail("must not launch"))

    assert (tmp_path / "audio.wav").read_bytes() == b"old"


def test_cancellation_kills_only_when_terminate_does_not_finish(tmp_path: Path):
    token = CancellationToken()

    class StuckProcess(CompletedProcess):
        def __init__(self, output: Path) -> None:
            super().__init__(output)
            self.wait_calls = 0

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls <= 2:
                if self.wait_calls == 1:
                    token.cancel()
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return 0

    holder: dict[str, StuckProcess] = {}

    def popen(command: list[str], **kwargs: object) -> StuckProcess:
        output = Path(command[-1])
        output.write_bytes(b"partial")
        holder["process"] = StuckProcess(output)
        return holder["process"]

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, lambda update: None,
                      probe_runner=probe_runner, popen_factory=popen)

    assert holder["process"].terminated
    assert holder["process"].killed


def test_ffmpeg_does_not_create_an_undrained_stderr_pipe(tmp_path: Path):
    captured: dict[str, object] = {}

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        captured.update(kwargs)
        write_pcm_wav(Path(command[-1]))
        return CompletedProcess(Path(command[-1]))

    prepare_audio(Path("source.mp4"), tmp_path, None, None, CancellationToken(), lambda update: None,
                  probe_runner=probe_runner, popen_factory=popen)
    assert captured["stderr"] is subprocess.DEVNULL


def test_domain_validation_happens_before_partial_is_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import engine.ingest.audio as audio

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        write_pcm_wav(Path(command[-1]))
        return CompletedProcess(Path(command[-1]))

    def invalid_media(**kwargs: object) -> object:
        raise ValueError("invalid prepared media")

    monkeypatch.setattr(audio, "PreparedMedia", invalid_media)
    with pytest.raises(ValueError, match="invalid prepared media"):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, CancellationToken(), lambda update: None,
                      probe_runner=probe_runner, popen_factory=popen)
    assert not (tmp_path / "audio.partial.wav").exists()
    assert not (tmp_path / "audio.wav").exists()


@pytest.mark.parametrize("stuck", [False, True])
def test_cancellable_ffprobe_is_terminated_and_killed_only_when_needed(tmp_path: Path, stuck: bool):
    token = CancellationToken()

    class ProbeProcess:
        def __init__(self) -> None:
            self.calls = 0
            self.terminated = False
            self.killed = False

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.calls += 1
            if self.calls == 1:
                token.cancel()
                raise subprocess.TimeoutExpired("ffprobe", timeout)
            if stuck and self.calls == 2:
                raise subprocess.TimeoutExpired("ffprobe", timeout)
            return ("", "")

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            if stuck and timeout is not None:
                raise subprocess.TimeoutExpired("ffprobe", timeout)
            return 0

    process = ProbeProcess()

    def probe_popen(command: list[str], **kwargs: object) -> ProbeProcess:
        assert kwargs["shell"] is False
        return process

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, lambda update: None,
                      probe_popen_factory=probe_popen)
    assert process.terminated
    assert process.killed is stuck


def test_cancel_cleanup_reaps_when_terminate_races_with_an_exited_process(tmp_path: Path):
    token = CancellationToken()

    class RaceProcess(CompletedProcess):
        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            token.cancel()
            if timeout is not None:
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            return 0

        def terminate(self) -> None:
            raise ProcessLookupError

    process: RaceProcess | None = None

    def popen(command: list[str], **kwargs: object) -> RaceProcess:
        nonlocal process
        Path(command[-1]).write_bytes(b"partial")
        process = RaceProcess(Path(command[-1]))
        return process

    with pytest.raises(JobCancelled):
        prepare_audio(Path("source.mp4"), tmp_path, None, None, token, lambda update: None,
                      probe_runner=probe_runner, popen_factory=popen)
    assert process is not None
    assert not (tmp_path / "audio.partial.wav").exists()


@pytest.mark.parametrize(
    ("container", "requested", "expected"),
    [
        (ContainerKind.VIDEO, None, ContentContext.VIDEO),
        (ContainerKind.AUDIO, None, ContentContext.UNKNOWN),
        (ContainerKind.VIDEO, ContentContext.MUSIC, ContentContext.MUSIC),
    ],
)
def test_context_is_only_a_container_or_explicit_hint(
    tmp_path: Path, container: ContainerKind, requested: ContentContext | None, expected: ContentContext, monkeypatch: pytest.MonkeyPatch,
):
    fixture_name = "audio_only.json" if container is ContainerKind.AUDIO else "video_two_audio.json"

    def runner(command: list[str], **kwargs: object) -> object:
        return type("Result", (), {"stdout": (FIXTURES / fixture_name).read_text()})()

    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        output = Path(command[-1])
        write_pcm_wav(output)
        return CompletedProcess(output)

    prepared = prepare_audio(Path("source"), tmp_path, None, requested, CancellationToken(), lambda update: None,
                              probe_runner=runner, popen_factory=popen)
    assert prepared.content_context_hint is expected


@pytest.mark.parametrize(
    ("requested_stream", "duration", "windows"),
    [
        (None, 10.0, [(0, 10)]),
        (0, 10.0, [(0, 10)]),
        (1, 65.0, [(0, 30), (30, 60), (60, 65)]),
    ],
)
def test_selected_track_duration_bounds_real_transcription_windows(
    tmp_path: Path, ffmpeg_tools: None, requested_stream: int | None,
    duration: float, windows: list[tuple[int, int]],
):
    source = tmp_path / "different track durations.mkv"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=16000:duration=10",
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=16000:duration=65",
            "-map", "0:a", "-map", "1:a", "-c:a", "pcm_s16le", str(source),
        ],
        check=True, capture_output=True,
    )
    token = CancellationToken()
    prepared = prepare_audio(source, tmp_path / "job", requested_stream, None, token, lambda _: None)
    transcriber = SimpleNamespace(
        detect_language_probabilities=lambda *_: {"en": 1.0},
        transcribe_window=lambda *_: [TranscriptSegment(
            time_range=TimeRange(start_seconds=0, end_seconds=1),
            text="Selected track transcript", confidence=1, language="en",
        )],
    )
    screener = SimpleNamespace(screen=lambda *_: ScreenResult(speech_max=1, singing_max=0, admitted=True))

    run = transcribe_media(
        prepared, transcriber=transcriber, screener=screener,
        work_dir=tmp_path / "work", token=token,
        checkpoint=lambda _: None, on_progress=lambda *_: None,
    )

    assert run.status == "ready"
    assert prepared.duration_seconds == duration
    assert prepared.source_metadata.duration_seconds == 65
    assert [(item.time_range.start_seconds, item.time_range.end_seconds) for item in run.windows] == windows


@pytest.mark.parametrize(
    ("requested_context", "expected_context"),
    [(None, ContentContext.UNKNOWN), (ContentContext.MUSIC, ContentContext.MUSIC)],
)
def test_mp3_cover_art_keeps_audio_context(
    tmp_path: Path, ffmpeg_tools: None,
    requested_context: ContentContext | None, expected_context: ContentContext,
):
    source = tmp_path / "podcast with cover.mp3"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2", "-f", "lavfi", "-i", "color=c=red:s=32x32:d=0.04",
            "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame", "-c:v", "mjpeg",
            "-disposition:v:0", "attached_pic", str(source),
        ],
        check=True, capture_output=True,
    )

    prepared = prepare_audio(
        source, tmp_path / "job", None, requested_context, CancellationToken(), lambda _: None,
    )

    assert prepared.source_metadata.duration_seconds > 1.9
    assert prepared.container_kind is ContainerKind.AUDIO
    assert prepared.content_context_hint is expected_context


def test_unreadable_normalized_audio_is_not_promoted(tmp_path: Path):
    def popen(command: list[str], **kwargs: object) -> CompletedProcess:
        output = Path(command[-1])
        output.write_bytes(b"not a WAV file")
        return CompletedProcess(output)

    with pytest.raises(RuntimeError, match="WAV"):
        prepare_audio(
            Path("source.mp4"), tmp_path, None, None, CancellationToken(), lambda _: None,
            probe_runner=probe_runner, popen_factory=popen,
        )

    assert not (tmp_path / "audio.partial.wav").exists()
    assert not (tmp_path / "audio.wav").exists()
