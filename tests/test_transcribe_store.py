from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import JobRecord, JobRequest, TimeRange, TranscriptSegment
from engine.lifecycle import JobStage
from engine.store import JobStore
from engine.transcribe.service import transcribe_media
from engine.transcribe.types import ScreenResult, WindowResult


JOB_ID = UUID("87654321-4321-6789-4321-678987654321")


def _transcribing_job(store: JobStore) -> JobRecord:
    job = store.create(JobRequest(source="input.wav"))
    for stage in (
        JobStage.EXTRACTING_AUDIO,
        JobStage.DETECTING_LANGUAGE,
        JobStage.TRANSCRIBING,
    ):
        job = JobRecord.model_validate(
            {
                **job.model_dump(mode="json"),
                "status": stage,
                "progress": job.progress.advance(
                    stage=stage, percent=0, message=stage.value
                ),
            }
        )
        store.save(job)
    return job


def _window(start: float = 0, end: float = 30) -> WindowResult:
    return WindowResult(
        time_range=TimeRange(start_seconds=start, end_seconds=end),
        language="en",
        segments=(
            TranscriptSegment(
                time_range=TimeRange(start_seconds=start, end_seconds=start + 1),
                text="hello",
                confidence=0.9,
                language="en",
            ),
        ),
    )


def test_completed_window_journal_is_readable_revisioned_and_idempotent(tmp_path: Path):
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: JOB_ID)
    job = _transcribing_job(store)

    first = store.append_transcription_window(job.id, _window())
    retry = store.append_transcription_window(job.id, _window())

    assert store.load_transcription_windows(job.id) == (_window(),)
    assert first.progress.revision == job.progress.revision + 1
    assert retry.progress.revision == first.progress.revision


def test_failure_diagnostics_round_trip_through_real_job_store(tmp_path: Path):
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: JOB_ID)
    job = _transcribing_job(store)
    failure = WindowResult(
        time_range=TimeRange(start_seconds=0, end_seconds=30),
        language="en",
        error_code="MODEL_FAILURE",
        failure_stage="timestamp_validation",
        failure_reason="invalid_timestamps",
        attempt_count=2,
    )

    store.append_transcription_window(job.id, failure)

    assert store.load_transcription_windows(job.id) == (failure,)


def test_timestamp_correction_diagnostics_round_trip_through_real_job_store(
    tmp_path: Path,
):
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: JOB_ID)
    job = _transcribing_job(store)
    corrected = WindowResult.model_validate(
        {
            "time_range": {"start_seconds": 600, "end_seconds": 615},
            "language": "en",
            "segments": [
                {
                    "time_range": {
                        "start_seconds": 612.2000000000001,
                        "end_seconds": 613,
                    },
                    "text": "hello",
                    "confidence": 0.9,
                    "language": "en",
                }
            ],
            "timestamp_corrections": [
                {
                    "segment_index": 1,
                    "original_start_seconds": 12.2,
                    "corrected_start_seconds": 12.200000000000001,
                    "local_end_seconds": 13,
                }
            ],
        }
    )

    store.append_transcription_window(job.id, corrected)

    loaded = store.load_transcription_windows(job.id)
    assert loaded == (corrected,)
    assert loaded[0].timestamp_corrections[0].original_start_seconds == 12.2


def test_window_journal_survives_snapshot_failure_and_reconciles_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: JOB_ID)
    job = _transcribing_job(store)
    original = store._write_snapshot
    failed = False

    def fail_once(record: JobRecord, descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("snapshot failed")
        original(record, descriptor)

    monkeypatch.setattr(store, "_write_snapshot", fail_once)

    with pytest.raises(OSError, match="snapshot failed"):
        store.append_transcription_window(job.id, _window())

    assert store.load_transcription_windows(job.id) == (_window(),)
    assert store.load(job.id).progress.revision == job.progress.revision + 1


def test_cancellation_after_second_checkpoint_keeps_two_windows_and_starts_no_third_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = JobStore(tmp_path / "jobs", uuid_factory=lambda: JOB_ID)
    job = _transcribing_job(store)
    token = CancellationToken()
    revisions: list[int] = []

    def slice_spy(media, window, work_dir, active_token):
        active_token.raise_if_cancelled()
        return work_dir / f"{window.start_seconds}.wav"

    monkeypatch.setattr("engine.transcribe.service.slice_audio_window", slice_spy)

    class Screener:
        def screen(self, audio, active_token):
            active_token.raise_if_cancelled()
            return ScreenResult(speech_max=0.9, singing_max=0, admitted=True)

    class Transcriber:
        def __init__(self) -> None:
            self.asr_calls = 0

        def detect_language_probabilities(self, audio, window, active_token):
            active_token.raise_if_cancelled()
            return {"en": 1.0}

        def transcribe_window(self, audio, window, language, active_token):
            active_token.raise_if_cancelled()
            self.asr_calls += 1
            return [
                TranscriptSegment(
                    time_range=TimeRange(start_seconds=0, end_seconds=1),
                    text=f"window {self.asr_calls}",
                    confidence=1,
                    language="en",
                )
            ]

    transcriber = Transcriber()

    def checkpoint(result: WindowResult) -> None:
        saved = store.append_transcription_window(job.id, result)
        revisions.append(saved.progress.revision)

    def progress(completed: int, total: int) -> None:
        current = store.load(job.id)
        update = JobRecord.model_validate(
            {
                **current.model_dump(mode="json"),
                "progress": current.progress.advance(
                    stage=JobStage.TRANSCRIBING,
                    percent=completed / total * 100,
                    message=f"window {completed} of {total}",
                ),
            }
        )
        store.save(update)
        revisions.append(update.progress.revision)
        if completed == 2:
            token.cancel()

    with pytest.raises(JobCancelled):
        transcribe_media(
            SimpleNamespace(audio_path=tmp_path / "prepared.wav", duration_seconds=61),
            transcriber=transcriber,
            screener=Screener(),
            work_dir=tmp_path / "work",
            token=token,
            checkpoint=checkpoint,
            on_progress=progress,
        )

    windows = store.load_transcription_windows(job.id)
    assert len(windows) == 2
    assert transcriber.asr_calls == 2
    assert revisions == sorted(set(revisions))
    assert store.load(job.id).status is JobStage.TRANSCRIBING
    assert store.read_export(job.id) is None
