"""Extract the already-selected audio stream without decoding video frames."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Callable, Protocol

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import ContentContext, PreparedMedia
from engine.events import Progress
from engine.ingest.probe import PopenFactory as ProbePopenFactory
from engine.ingest.probe import ProbeRunner, choose_audio_stream, probe_source
from engine.lifecycle import JobStage

ProgressUpdate = Progress


class PopenFactory(Protocol):
    def __call__(self, args: list[str], **kwargs: object) -> subprocess.Popen[bytes]: ...


def _context_hint(requested: ContentContext | None, is_video: bool) -> ContentContext:
    return requested if requested is not None else ContentContext.VIDEO if is_video else ContentContext.UNKNOWN


def _cancel_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=0.1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def prepare_audio(
    source: Path,
    job_dir: Path,
    requested_stream: int | None,
    requested_context: ContentContext | None,
    token: CancellationToken,
    emit: Callable[[ProgressUpdate], None],
    *,
    probe_runner: ProbeRunner = subprocess.run,
    popen_factory: PopenFactory = subprocess.Popen,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    probe_popen_factory: ProbePopenFactory = subprocess.Popen,
) -> PreparedMedia:
    """Probe and atomically normalize exactly one selected audio stream to WAV."""
    token.raise_if_cancelled()
    job_dir.mkdir(parents=True, exist_ok=True)
    partial_path = job_dir / "audio.partial.wav"
    audio_path = job_dir / "audio.wav"
    partial_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)
    probe = probe_source(source, runner=probe_runner, ffprobe=ffprobe, token=token, popen_factory=probe_popen_factory)
    selected_stream = choose_audio_stream(probe, requested_stream)
    token.raise_if_cancelled()
    command = [
        ffmpeg, "-nostdin", "-y", "-i", str(source), "-map", f"0:{selected_stream.index}", "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(partial_path),
    ]
    emit(Progress(stage=JobStage.EXTRACTING_AUDIO, percent=0, message="extracting selected audio"))
    token.raise_if_cancelled()
    process = popen_factory(command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
            raise RuntimeError("ffmpeg audio extraction failed")
        token.raise_if_cancelled()
        try:
            with wave.open(str(partial_path), "rb") as audio:
                duration_seconds = audio.getnframes() / audio.getframerate()
        except (EOFError, OSError, wave.Error) as error:
            raise RuntimeError("ffmpeg produced an invalid WAV") from error
        prepared = PreparedMedia(
            source_path=source,
            audio_path=audio_path,
            container_kind=probe.container_kind,
            content_context_hint=_context_hint(requested_context, probe.container_kind.value == "video"),
            selected_audio_stream=selected_stream.index,
            available_audio_streams=probe.audio_streams,
            duration_seconds=duration_seconds,
            source_metadata=probe.source_metadata,
        )
        token.run_if_not_cancelled(lambda: partial_path.replace(audio_path))
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return prepared
