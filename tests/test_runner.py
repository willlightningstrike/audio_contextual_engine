from __future__ import annotations

import asyncio
from pathlib import Path
import threading

import pytest


class _Pipeline:
    def __init__(self, store, *, block_first: bool = False):
        self.store = store
        self.block_first = block_first
        self.started: list[str] = []
        self.worker_threads: set[int] = set()
        self.active = 0
        self.peak_active = 0
        self.lock = threading.Lock()
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def run(self, job_id, token):
        from engine.cancel import JobCancelled
        from engine.lifecycle import JobStage

        with self.lock:
            self.started.append(job_id)
            self.worker_threads.add(threading.get_ident())
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            is_first = len(self.started) == 1
        try:
            if self.block_first and is_first:
                self.first_started.set()
                while not self.release_first.wait(0.005):
                    token.raise_if_cancelled()
            token.raise_if_cancelled()
            record = self.store.load(job_id)
            progress = record.progress.advance(
                stage=JobStage.FAILED, percent=0, message="fixture_failure"
            )
            self.store.save(
                record.model_copy(
                    update={
                        "status": JobStage.FAILED,
                        "progress": progress,
                        "error": "fixture_failure",
                    }
                )
            )
        except JobCancelled:
            record = self.store.load(job_id)
            if record.status is not JobStage.CANCELLING:
                progress = record.progress.advance(
                    stage=JobStage.CANCELLING, percent=0, message="cancellation requested"
                )
                record = record.model_copy(
                    update={"status": JobStage.CANCELLING, "progress": progress}
                )
                self.store.save(record)
            progress = record.progress.advance(
                stage=JobStage.CANCELLED, percent=0, message="cancelled"
            )
            self.store.save(
                record.model_copy(
                    update={"status": JobStage.CANCELLED, "progress": progress, "error": "cancelled"}
                )
            )
        finally:
            with self.lock:
                self.active -= 1


def _request(store, name: str):
    from engine.domain import JobRequest

    return store.create(JobRequest(source=name))


@pytest.mark.asyncio
async def test_runner_executes_jobs_fifo_with_only_one_pipeline_call_active(tmp_path: Path):
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    pipeline = _Pipeline(store)
    runner = JobRunner(pipeline, store)
    jobs = [_request(store, f"{index}.wav") for index in range(3)]
    for job in jobs:
        await runner.enqueue(job.id)
    await runner.shutdown()

    assert pipeline.started == [job.id for job in jobs]
    assert pipeline.peak_active == 1
    assert len(pipeline.worker_threads) == 1
    from engine.lifecycle import JobStage

    assert all(store.load(job.id).status is JobStage.FAILED for job in jobs)


@pytest.mark.asyncio
async def test_runner_cancels_queued_job_idempotently_without_starting_it(tmp_path: Path):
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    pipeline = _Pipeline(store, block_first=True)
    runner = JobRunner(pipeline, store)
    first, second = (_request(store, f"{index}.wav") for index in range(2))
    await runner.enqueue(first.id)
    assert await asyncio.to_thread(pipeline.first_started.wait, 2)
    await runner.enqueue(second.id)
    await runner.cancel(second.id)
    await runner.cancel(second.id)
    pipeline.release_first.set()
    await runner.shutdown()

    from engine.lifecycle import JobStage

    assert pipeline.started == [first.id]
    assert store.load(second.id).status is JobStage.CANCELLED


@pytest.mark.asyncio
async def test_runner_repeatedly_cancels_active_pipeline_cooperatively(tmp_path: Path):
    from engine.lifecycle import JobStage
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    pipeline = _Pipeline(store, block_first=True)
    runner = JobRunner(pipeline, store)
    job = _request(store, "active.wav")
    await runner.enqueue(job.id)
    assert await asyncio.to_thread(pipeline.first_started.wait, 2)
    await runner.cancel(job.id)
    await runner.cancel(job.id)
    await runner.shutdown()

    assert store.load(job.id).status is JobStage.CANCELLED
    assert pipeline.peak_active == 1


@pytest.mark.asyncio
async def test_runner_recovers_each_queued_job_once_and_fails_interrupted_active_jobs(tmp_path: Path):
    from engine.domain import TimeRange, TranscriptSegment
    from engine.lifecycle import JobStage
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    queued = _request(store, "queued.wav")
    active = _request(store, "active.wav")
    checkpoint = TranscriptSegment(
        time_range=TimeRange(start_seconds=0, end_seconds=1),
        text="saved before restart",
        confidence=0.9,
        language="en",
    )
    store.append_transcript(active.id, checkpoint)
    active = store.load(active.id)
    for stage in (
        JobStage.EXTRACTING_AUDIO,
        JobStage.DETECTING_LANGUAGE,
        JobStage.TRANSCRIBING,
    ):
        progress = active.progress.advance(stage=stage, percent=40, message="interrupted")
        active = active.model_copy(update={"status": stage, "progress": progress})
        store.save(active)
    pipeline = _Pipeline(store)
    runner = JobRunner(pipeline, store)
    await runner.shutdown()

    assert pipeline.started == [queued.id]
    assert store.load(queued.id).error == "fixture_failure"
    assert store.load(active.id).error == "interrupted_by_restart"
    assert store.load(active.id).transcript == (checkpoint,)


@pytest.mark.asyncio
async def test_runner_shutdown_drains_accepted_jobs_and_rejects_later_enqueue(tmp_path: Path):
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    pipeline = _Pipeline(store, block_first=True)
    runner = JobRunner(pipeline, store)
    job = _request(store, "shutdown.wav")
    await runner.enqueue(job.id)
    assert await asyncio.to_thread(pipeline.first_started.wait, 2)
    shutdown = asyncio.create_task(runner.shutdown())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="shut down"):
        await runner.enqueue(_request(store, "late.wav").id)
    pipeline.release_first.set()
    await shutdown


@pytest.mark.asyncio
async def test_concurrent_shutdown_requests_send_only_one_worker_sentinel(tmp_path: Path):
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    runner = JobRunner(_Pipeline(store), store)
    await asyncio.gather(runner.shutdown(), runner.shutdown())

    assert not runner._worker.is_alive()
    assert runner._queue.unfinished_tasks == 0


class _MixedOutcomePipeline:
    def __init__(self, store):
        self.store = store
        self.started: list[str] = []

    def run(self, job_id, token):
        from engine.cancel import JobCancelled
        from engine.lifecycle import JobStage

        token.raise_if_cancelled()
        job = self.store.load(job_id)
        self.started.append(job.request.source.name)
        if job.request.source.name == "success.wav":
            for stage in (
                JobStage.EXTRACTING_AUDIO,
                JobStage.DETECTING_LANGUAGE,
                JobStage.TRANSCRIBING,
                JobStage.ANALYSING,
                JobStage.EXPORTING,
            ):
                job = self.store.load(job_id)
                progress = job.progress.advance(stage=stage, percent=90, message=stage.value)
                job = job.model_copy(update={"status": stage, "progress": progress})
                self.store.save(job)
            self.store.promote_result(job_id, {"id": "fixture"})
            return
        error = "non_english" if job.request.source.name == "non-english.wav" else "instrumental"
        progress = job.progress.advance(
            stage=JobStage.FAILED, percent=0, message=error
        )
        self.store.save(
            job.model_copy(update={"status": JobStage.FAILED, "progress": progress, "error": error})
        )


@pytest.mark.asyncio
async def test_batch_keeps_success_non_english_and_instrumental_outcomes_and_continues(tmp_path: Path):
    from engine.domain import JobRequest
    from engine.lifecycle import JobStage
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    pipeline = _MixedOutcomePipeline(store)
    runner = JobRunner(pipeline, store)
    children = tuple(
        store.create(JobRequest(source=name))
        for name in ("success.wav", "non-english.wav", "instrumental.wav")
    )
    batch = store.create_batch(tmp_path, tuple(job.id for job in children))
    with pytest.raises(ValueError, match="through their parent"):
        await runner.enqueue(children[0].id)
    await runner.enqueue(batch.id)
    result = await runner.wait(batch.id)
    await runner.shutdown()

    assert pipeline.started == ["success.wav", "non-english.wav", "instrumental.wav"]
    assert result.status is JobStage.COMPLETE
    assert result.success_count == 1
    assert result.failure_count == 2
    assert result.cancelled_count == 0
    assert result.all_children_succeeded is False
    assert [store.load(job.id).error for job in children] == [None, "non_english", "instrumental"]


@pytest.mark.asyncio
async def test_batch_parent_cancellation_stops_after_active_child_and_records_remaining(tmp_path: Path):
    from engine.domain import JobRequest
    from engine.lifecycle import JobStage
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    pipeline = _Pipeline(store, block_first=True)
    runner = JobRunner(pipeline, store)
    children = tuple(
        store.create(JobRequest(source=f"child-{number}.wav")) for number in range(3)
    )
    batch = store.create_batch(tmp_path, tuple(job.id for job in children))
    await runner.enqueue(batch.id)
    assert await asyncio.to_thread(pipeline.first_started.wait, 2)
    await runner.cancel(batch.id)
    await runner.cancel(batch.id)
    result = await runner.wait(batch.id)
    await runner.shutdown()

    assert pipeline.started == [children[0].id]
    assert result.status is JobStage.CANCELLED
    assert result.cancelled_count == 3
    assert all(store.load(job.id).status is JobStage.CANCELLED for job in children)


@pytest.mark.asyncio
async def test_concurrent_duplicate_enqueue_and_cancel_requests_remain_serial_and_idempotent(
    tmp_path: Path,
):
    from engine.lifecycle import JobStage
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    pipeline = _Pipeline(store)
    runner = JobRunner(pipeline, store)
    jobs = [_request(store, f"{index}.wav") for index in range(12)]

    async def enqueue_concurrently(job_id: str) -> None:
        await asyncio.to_thread(lambda: asyncio.run(runner.enqueue(job_id)))

    async def cancel_concurrently(job_id: str) -> None:
        await asyncio.to_thread(lambda: asyncio.run(runner.cancel(job_id)))

    await asyncio.gather(
        *(enqueue_concurrently(job.id) for job in jobs for _ in range(3))
    )
    await asyncio.gather(
        *(cancel_concurrently(job.id) for job in jobs[::2] for _ in range(2))
    )
    await asyncio.gather(*(runner.wait(job.id) for job in jobs))
    await runner.shutdown()

    assert pipeline.peak_active == 1
    assert len(pipeline.started) == len(set(pipeline.started))
    assert all(
        store.load(job.id).status in {JobStage.FAILED, JobStage.CANCELLED}
        for job in jobs
    )


@pytest.mark.asyncio
async def test_restart_fails_active_batch_and_cancels_unstarted_children_without_replay(
    tmp_path: Path,
):
    from engine.domain import JobRequest
    from engine.lifecycle import JobStage
    from engine.runner import JobRunner
    from engine.store import JobStore

    store = JobStore(tmp_path / "jobs")
    children = tuple(store.create(JobRequest(source=f"{i}.wav")) for i in range(2))
    batch = store.create_batch(tmp_path, tuple(job.id for job in children))
    progress = batch.progress.advance(
        stage=JobStage.BATCH_PROCESSING, percent=20, message="interrupted in child"
    )
    store.save_batch(
        batch.model_copy(
            update={
                "status": JobStage.BATCH_PROCESSING,
                "progress": progress,
                "active_child_job_id": children[0].id,
            }
        )
    )
    active = store.load(children[0].id)
    for stage in (JobStage.EXTRACTING_AUDIO, JobStage.DETECTING_LANGUAGE):
        progress = active.progress.advance(stage=stage, percent=10, message="interrupted")
        active = active.model_copy(update={"status": stage, "progress": progress})
        store.save(active)

    pipeline = _Pipeline(store)
    runner = JobRunner(pipeline, store)
    await runner.shutdown()

    assert pipeline.started == []
    recovered = store.load_batch(batch.id)
    assert recovered.status is JobStage.FAILED
    assert recovered.error == "interrupted_by_restart"
    assert recovered.failure_count == 1
    assert recovered.cancelled_count == 1
    assert store.load(children[0].id).status is JobStage.FAILED
    assert store.load(children[1].id).status is JobStage.CANCELLED
