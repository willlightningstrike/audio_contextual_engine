"""FIFO, single-threaded execution for durable jobs and batch parents."""

from __future__ import annotations

import asyncio
from queue import Queue
import threading
from typing import Protocol

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import BatchRecord, JobRecord
from engine.lifecycle import JobStage
from engine.store import JobNotFound, JobStore


class Pipeline(Protocol):
    def run(self, job_id: str, token: CancellationToken) -> None: ...


class JobRunner:
    """Run accepted work on one dedicated worker thread in FIFO order."""

    def __init__(self, pipeline: Pipeline, store: JobStore) -> None:
        self.pipeline = pipeline
        self.store = store
        self._queue: Queue[tuple[str, str] | None] = Queue()
        self._lock = threading.RLock()
        self._tokens: dict[str, CancellationToken] = {}
        self._active_children: dict[str, CancellationToken] = {}
        self._queued: set[str] = set()
        self._running: set[str] = set()
        self._events: dict[str, threading.Event] = {}
        self._closing = False
        self._closed = False
        self._shutdown_complete = threading.Event()

        jobs = store.recover()
        batches = store.recover_batches()
        self._batch_children = {
            child_id for batch in batches for child_id in batch.child_job_ids
        }
        self._worker = threading.Thread(
            target=self._work, name="contextual-engine-worker", daemon=True
        )
        self._worker.start()
        for job in jobs:
            if job.status is JobStage.QUEUED and job.id not in self._batch_children:
                self._schedule("job", job.id)
        for batch in batches:
            if batch.status is JobStage.QUEUED:
                self._schedule("batch", batch.id)

    async def enqueue(self, job_id: str) -> None:
        with self._lock:
            self._ensure_open()
            try:
                batch = self.store.load_batch(job_id)
            except JobNotFound:
                job = self.store.load(job_id)
                if job.status is not JobStage.QUEUED:
                    if job_id in self._tokens or job_id in self._queued:
                        return
                    return
                self._batch_children.update(
                    child_id
                    for existing_batch in self.store.list_batches()
                    for child_id in existing_batch.child_job_ids
                )
                if job_id in self._batch_children:
                    raise ValueError("batch children must be enqueued through their parent")
                self._schedule("job", job_id)
            else:
                if batch.status is not JobStage.QUEUED:
                    return
                self._batch_children.update(batch.child_job_ids)
                self._schedule("batch", job_id)

    async def cancel(self, job_id: str) -> None:
        with self._lock:
            try:
                batch = self.store.load_batch(job_id)
            except JobNotFound:
                self._cancel_job(job_id)
                return
            if batch.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                return
            token = self._tokens.get(job_id)
            if token is not None and job_id in self._running:
                token.cancel()
                active_id = batch.active_child_job_id
                if active_id is not None and active_id in self._active_children:
                    self._active_children[active_id].cancel()
                if batch.status is not JobStage.CANCELLING:
                    progress = batch.progress.advance(
                        stage=JobStage.CANCELLING,
                        percent=batch.progress.percent,
                        message="cancellation requested",
                    )
                    self.store.save_batch(
                        batch.model_copy(
                            update={"status": JobStage.CANCELLING, "progress": progress}
                        )
                    )
                return
            if batch.id not in self._running:
                token = self._tokens.get(batch.id)
                if token is not None:
                    token.cancel()
                self._cancel_queued_batch(batch)

    async def wait(self, item_id: str) -> JobRecord | BatchRecord:
        """Wait for an accepted queue item and return its durable terminal record."""
        with self._lock:
            event = self._events.setdefault(item_id, threading.Event())
            try:
                batch = self.store.load_batch(item_id)
            except JobNotFound:
                job = self.store.load(item_id)
                if job.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                    return job
            else:
                if batch.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                    return batch
        await asyncio.to_thread(event.wait)
        try:
            return self.store.load_batch(item_id)
        except JobNotFound:
            return self.store.load(item_id)

    async def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing:
                wait_for_owner = True
            else:
                wait_for_owner = False
                self._closing = True
        if wait_for_owner:
            await asyncio.to_thread(self._shutdown_complete.wait)
            return
        await asyncio.to_thread(self._queue.join)
        self._queue.put(None)
        await asyncio.to_thread(self._worker.join)
        with self._lock:
            self._closed = True
            self._shutdown_complete.set()

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("job runner is shut down")

    def _schedule(self, kind: str, item_id: str) -> None:
        if item_id in self._queued or item_id in self._tokens:
            return
        self._queued.add(item_id)
        self._events[item_id] = threading.Event()
        self._tokens[item_id] = CancellationToken()
        self._queue.put((kind, item_id))

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            kind, item_id = item
            with self._lock:
                self._queued.discard(item_id)
                token = self._tokens.setdefault(item_id, CancellationToken())
                if token.cancelled:
                    self._tokens.pop(item_id, None)
                    self._events.setdefault(item_id, threading.Event()).set()
                    self._queue.task_done()
                    continue
                self._running.add(item_id)
            try:
                if kind == "batch":
                    self._run_batch(item_id, token)
                else:
                    self._run_job(item_id, token)
            except BaseException as exc:
                if kind == "batch":
                    self._finish_batch_failed(item_id, type(exc).__name__)
                else:
                    self._finish_job_failed(item_id, type(exc).__name__)
            finally:
                with self._lock:
                    self._tokens.pop(item_id, None)
                    self._running.discard(item_id)
                    self._events.setdefault(item_id, threading.Event()).set()
                self._queue.task_done()

    def _run_job(self, job_id: str, token: CancellationToken) -> None:
        try:
            self.pipeline.run(job_id, token)
        except JobCancelled:
            self._finish_job_cancelled(job_id)
        except BaseException as exc:
            self._finish_job_failed(job_id, type(exc).__name__)
        else:
            record = self.store.load(job_id)
            if record.status not in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                if token.cancelled:
                    self._finish_job_cancelled(job_id)
                else:
                    self._finish_job_failed(job_id, "pipeline_returned_without_terminal_state")

    def _run_batch(self, batch_id: str, token: CancellationToken) -> None:
        with self._lock:
            batch = self.store.load_batch(batch_id)
            if batch.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                return
            cancel_before_start = batch.status is JobStage.CANCELLING or token.cancelled
            if not cancel_before_start:
                progress = batch.progress.advance(
                    stage=JobStage.BATCH_PROCESSING, percent=0, message="processing batch"
                )
                batch = batch.model_copy(
                    update={"status": JobStage.BATCH_PROCESSING, "progress": progress}
                )
                self.store.save_batch(batch)
        if cancel_before_start:
            self._cancel_remaining_children(batch_id, None)
            with self._lock:
                batch = self.store.load_batch(batch_id)
                progress = batch.progress.advance(
                    stage=JobStage.CANCELLED,
                    percent=batch.progress.percent,
                    message="cancelled before next child started",
                    completed_units=len(batch.child_outcomes),
                    total_units=len(batch.child_job_ids),
                )
                self.store.save_batch(
                    batch.model_copy(
                        update={"status": JobStage.CANCELLED, "progress": progress, "active_child_job_id": None}
                    )
                )
            return
        for child_id in batch.child_job_ids:
            if token.cancelled:
                self._cancel_remaining_children(batch_id, child_id)
                break
            with self._lock:
                child = self.store.load(child_id)
                if child.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                    already_terminal = child.status
                    child_token = None
                else:
                    already_terminal = None
                    child_token = CancellationToken()
                    self._active_children[child_id] = child_token
                    batch = self.store.load_batch(batch_id)
                    progress = batch.progress.advance(
                        stage=batch.status,
                        percent=batch.progress.percent,
                        message=f"processing child {len(batch.child_outcomes) + 1} of {len(batch.child_job_ids)}",
                        current_file=child.request.source.name if child.request else None,
                        file_index=len(batch.child_outcomes),
                        completed_units=len(batch.child_outcomes),
                        total_units=len(batch.child_job_ids),
                    )
                    self.store.save_batch(
                        batch.model_copy(
                            update={"progress": progress, "active_child_job_id": child_id}
                        )
                    )
            if already_terminal is not None:
                self._record_child_outcome(batch_id, child_id, already_terminal)
                continue
            try:
                if token.cancelled:
                    assert child_token is not None
                    child_token.cancel()
                self.pipeline.run(child_id, child_token)
            except JobCancelled:
                self._finish_job_cancelled(child_id)
            except BaseException as exc:
                self._finish_job_failed(child_id, type(exc).__name__)
            finally:
                with self._lock:
                    self._active_children.pop(child_id, None)
                    self._events.setdefault(child_id, threading.Event()).set()
            child = self.store.load(child_id)
            if child.status not in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                self._finish_job_failed(child_id, "pipeline_returned_without_terminal_state")
                child = self.store.load(child_id)
            self._record_child_outcome(batch_id, child_id, child.status)
            if token.cancelled:
                self._cancel_remaining_children(batch_id, None)
                break

        with self._lock:
            batch = self.store.load_batch(batch_id)
        if len(batch.child_outcomes) != len(batch.child_job_ids):
            self._cancel_remaining_children(batch_id, None)
        with self._lock:
            batch = self.store.load_batch(batch_id)
            terminal = JobStage.CANCELLED if token.cancelled else JobStage.COMPLETE
            progress = batch.progress.advance(
                stage=terminal,
                percent=100 if terminal is JobStage.COMPLETE else batch.progress.percent,
                message="batch processing complete" if terminal is JobStage.COMPLETE else "cancelled",
                completed_units=len(batch.child_outcomes),
                total_units=len(batch.child_job_ids),
                current_file=None,
            )
            self.store.save_batch(
                batch.model_copy(
                    update={"status": terminal, "progress": progress, "active_child_job_id": None}
                )
            )

    def _record_child_outcome(self, batch_id: str, child_id: str, outcome: JobStage) -> None:
        with self._lock:
            if outcome not in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                outcome = JobStage.FAILED
            batch = self.store.load_batch(batch_id)
            if child_id in batch.child_outcomes:
                return
            outcomes = {**batch.child_outcomes, child_id: outcome}
            completed = len(outcomes)
            progress = batch.progress.advance(
                stage=batch.status,
                percent=(100 * completed / len(batch.child_job_ids)) if batch.child_job_ids else 100,
                message=f"completed {completed} of {len(batch.child_job_ids)} children",
                completed_units=completed,
                total_units=len(batch.child_job_ids),
            )
            self.store.save_batch(
                batch.model_copy(
                    update={
                        "child_outcomes": outcomes,
                        "success_count": sum(item is JobStage.COMPLETE for item in outcomes.values()),
                        "failure_count": sum(item is JobStage.FAILED for item in outcomes.values()),
                        "cancelled_count": sum(item is JobStage.CANCELLED for item in outcomes.values()),
                        "active_child_job_id": None,
                        "progress": progress,
                    }
                )
            )

    def _cancel_remaining_children(self, batch_id: str, start_child_id: str | None) -> None:
        batch = self.store.load_batch(batch_id)
        start = batch.child_job_ids.index(start_child_id) if start_child_id is not None else 0
        for child_id in batch.child_job_ids[start:]:
            current = self.store.load_batch(batch_id)
            if child_id in current.child_outcomes:
                continue
            child = self.store.load(child_id)
            if child.status is JobStage.QUEUED:
                self._finish_job_cancelled(child_id)
                child = self.store.load(child_id)
            if child.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                self._record_child_outcome(batch_id, child_id, child.status)

    def _cancel_queued_batch(self, batch: BatchRecord) -> None:
        for child_id in batch.child_job_ids:
            child = self.store.load(child_id)
            if child.status is JobStage.QUEUED:
                self._finish_job_cancelled(child_id)
                child = self.store.load(child_id)
            self._record_child_outcome(batch.id, child_id, child.status)
        batch = self.store.load_batch(batch.id)
        progress = batch.progress.advance(
            stage=JobStage.CANCELLED,
            percent=100 if batch.child_job_ids else 0,
            message="cancelled before processing",
            completed_units=len(batch.child_outcomes),
            total_units=len(batch.child_job_ids),
        )
        self.store.save_batch(
            batch.model_copy(
                update={"status": JobStage.CANCELLED, "progress": progress, "active_child_job_id": None}
            )
        )
        self._events.setdefault(batch.id, threading.Event()).set()

    def _cancel_job(self, job_id: str) -> None:
        job = self.store.load(job_id)
        if job.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
            return
        token = self._tokens.get(job_id) or self._active_children.get(job_id)
        if job.status is JobStage.QUEUED and job_id not in self._running and job_id not in self._active_children:
            if token is not None:
                token.cancel()
            self._finish_job_cancelled(job_id)
            self._events.setdefault(job_id, threading.Event()).set()
            return
        if token is not None:
            token.cancel()

    def _finish_job_cancelled(self, job_id: str) -> None:
        record = self.store.load(job_id)
        if record.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
            return
        if record.status is not JobStage.CANCELLING:
            progress = record.progress.advance(
                stage=JobStage.CANCELLING,
                percent=record.progress.percent,
                message="cancellation requested",
            )
            record = record.model_copy(
                update={"status": JobStage.CANCELLING, "progress": progress}
            )
            self.store.save(record)
        progress = record.progress.advance(
            stage=JobStage.CANCELLED, percent=record.progress.percent, message="cancelled"
        )
        self.store.save(
            record.model_copy(
                update={"status": JobStage.CANCELLED, "progress": progress, "error": "cancelled"}
            )
        )

    def _finish_job_failed(self, job_id: str, error: str) -> None:
        record = self.store.load(job_id)
        if record.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
            return
        if record.status is JobStage.CANCELLING:
            progress = record.progress.advance(
                stage=JobStage.FAILED, percent=record.progress.percent, message=error
            )
        else:
            progress = record.progress.advance(
                stage=JobStage.FAILED, percent=record.progress.percent, message=error
            )
        self.store.save(
            record.model_copy(
                update={"status": JobStage.FAILED, "progress": progress, "error": error}
            )
        )

    def _finish_batch_failed(self, batch_id: str, error: str) -> None:
        try:
            batch = self.store.load_batch(batch_id)
            if batch.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                return
            progress = batch.progress.advance(
                stage=JobStage.FAILED,
                percent=batch.progress.percent,
                message=error,
            )
            self.store.save_batch(
                batch.model_copy(
                    update={
                        "status": JobStage.FAILED,
                        "progress": progress,
                        "active_child_job_id": None,
                        "error": error,
                    }
                )
            )
        except Exception:
            return
