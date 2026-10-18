"""Durable, descriptor-anchored persistence for local job artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import json
import os
import stat
from pathlib import Path
from threading import RLock
from typing import Callable, Iterator, Literal, Mapping, TypeVar
from pydantic import BaseModel, ConfigDict, Field
from uuid import UUID, uuid4

from engine.domain import BatchRecord, JobRecord, JobRequest, TimeRange, TimelineEntry, TranscriptSegment
from engine.events import Progress
from engine.lifecycle import JobStage
from engine.transcribe.types import WindowResult
from engine.ortb.export import serialize_content
from engine.signals.result_types import TextTimelineEntry
from engine.completion import CoverageAssessment
from engine.signals.topic_sufficiency import TopicDecision
from engine.signals.topic_verify import VerifierAttempt


class _VerifierModelIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _VerifierDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    time_range: TimeRange
    policy_version: Literal["english-transcript-verifier-v1", "english-transcript-verifier-v2"]
    model: _VerifierModelIdentity
    attempt: VerifierAttempt


_DIAGNOSTIC_KINDS = frozenset({
    "candidates", "attempt", "verifier_attempt", "result", "topic_policy", "coverage_policy",
})


class UnsafeJobId(ValueError):
    """A job identifier is not a canonical UUID."""


class UnsafeJobPath(ValueError):
    """A root, job, or artifact path was a symlink or escaped the jobs root."""


class JobNotFound(FileNotFoundError):
    """A valid job identifier has no durable snapshot."""


class JobAlreadyExists(FileExistsError):
    """A create request collided with an already-reserved UUID directory."""


class InvalidJobTransition(ValueError):
    """A public snapshot update violates the job lifecycle."""


class JobConflict(InvalidJobTransition):
    """A non-idempotent write was derived from an obsolete snapshot revision."""


class CorruptJobJournal(ValueError):
    """A partial-results journal is malformed or diverges from its snapshot."""


class CorruptJobState(ValueError):
    """A snapshot lacks the artifacts required by its durable state."""


class UnsupportedJobSchemaVersion(CorruptJobState, CorruptJobJournal):
    """A durable job or journal uses a missing or unsupported schema version."""


_PartialUnit = TypeVar(
    "_PartialUnit", TranscriptSegment, TimelineEntry, WindowResult, TextTimelineEntry
)
_TERMINAL = frozenset({JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED})
_BATCH_TERMINAL = frozenset({JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED})
_INTERRUPTED = frozenset(
    {
        JobStage.EXTRACTING_AUDIO,
        JobStage.DETECTING_LANGUAGE,
        JobStage.TRANSCRIBING,
        JobStage.ANALYSING,
        JobStage.EXPORTING,
        JobStage.CANCELLING,
    }
)
_ALLOWED = {
    JobStage.QUEUED: frozenset(
        {JobStage.EXTRACTING_AUDIO, JobStage.CANCELLING, JobStage.CANCELLED, JobStage.FAILED}
    ),
    JobStage.EXTRACTING_AUDIO: frozenset(
        {JobStage.DETECTING_LANGUAGE, JobStage.CANCELLING, JobStage.FAILED}
    ),
    JobStage.DETECTING_LANGUAGE: frozenset(
        {JobStage.TRANSCRIBING, JobStage.CANCELLING, JobStage.FAILED}
    ),
    JobStage.TRANSCRIBING: frozenset(
        {JobStage.ANALYSING, JobStage.CANCELLING, JobStage.FAILED}
    ),
    JobStage.ANALYSING: frozenset(
        {JobStage.EXPORTING, JobStage.CANCELLING, JobStage.FAILED}
    ),
    JobStage.EXPORTING: frozenset({JobStage.CANCELLING, JobStage.FAILED}),
    JobStage.CANCELLING: frozenset({JobStage.CANCELLED, JobStage.FAILED}),
}
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class JobStore:
    """Persist jobs beneath one local jobs root using anchored directory FDs."""

    def __init__(self, jobs_root: Path, *, uuid_factory: Callable[[], UUID | str] = uuid4) -> None:
        self.jobs_root = Path(jobs_root)
        self._uuid_factory = uuid_factory
        self._local_lock = RLock()

    def create(self, request: JobRequest) -> JobRecord:
        request = JobRequest.model_validate_json(request.model_dump_json())
        job_id = self._validate_job_id(str(self._uuid_factory()))
        record = JobRecord(id=job_id, request=request, evidence_mode="transcript_only")
        with self._local_lock, self._root_lock() as (root, root_fd):
            try:
                os.mkdir(job_id, dir_fd=root_fd)
            except FileExistsError as exc:
                existing_fd = self._open_job_fd(root_fd, job_id)
                os.close(existing_fd)
                raise JobAlreadyExists(f"job already exists: {job_id}") from exc
            self._fsync_directory(root, root_fd)
            try:
                with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                    self._write_snapshot(record, job_fd)
            except BaseException:
                self._cleanup_new_directory(root, root_fd, job_id)
                raise
            return record

    def create_batch(self, source_directory: Path, child_job_ids: tuple[str, ...]) -> BatchRecord:
        """Persist a batch parent after every referenced child job is durable."""
        child_job_ids = tuple(self._validate_job_id(item) for item in child_job_ids)
        if not child_job_ids or len(child_job_ids) != len(set(child_job_ids)):
            raise ValueError("a batch requires unique child jobs")
        existing_children = {
            child_id for batch in self.list_batches() for child_id in batch.child_job_ids
        }
        if existing_children.intersection(child_job_ids):
            raise InvalidJobTransition("a child job can belong to only one batch")
        for child_id in child_job_ids:
            if self.load(child_id).status is not JobStage.QUEUED:
                raise InvalidJobTransition("batch children must be queued before batch creation")
        batch_id = self._validate_job_id(str(self._uuid_factory()))
        if batch_id in child_job_ids:
            raise JobAlreadyExists(f"batch ID collides with child job: {batch_id}")
        try:
            self.load(batch_id)
        except JobNotFound:
            pass
        else:
            raise JobAlreadyExists(f"batch ID collides with a job: {batch_id}")
        record = BatchRecord(
            id=batch_id,
            source_directory=Path(source_directory),
            child_job_ids=child_job_ids,
        )
        with self._local_lock, self._batch_root_anchor(create=True) as (batch_root, batch_root_fd):
            try:
                os.mkdir(batch_id, dir_fd=batch_root_fd)
            except FileExistsError as exc:
                raise JobAlreadyExists(f"batch already exists: {batch_id}") from exc
            self._fsync_directory(batch_root, batch_root_fd)
            try:
                with self._locked_batch(batch_root, batch_root_fd, batch_id) as (_, batch_fd):
                    self._write_batch_snapshot(record, batch_fd)
            except BaseException:
                self._cleanup_new_batch(batch_root, batch_root_fd, batch_id)
                raise
            return record

    def load_batch(self, batch_id: str) -> BatchRecord:
        batch_id = self._validate_job_id(batch_id)
        with self._local_lock, self._batch_root_anchor(create=False) as (root, root_fd):
            with self._locked_batch(root, root_fd, batch_id) as (_, batch_fd):
                return self._load_batch_locked(batch_fd)

    def list_batches(self) -> tuple[BatchRecord, ...]:
        """Read all durable batch snapshots without changing recovery state."""
        with self._local_lock:
            try:
                context = self._batch_root_anchor(create=False)
                root, root_fd = context.__enter__()
            except JobNotFound:
                return ()
            try:
                with self._locked_fd(root, root_fd):
                    batches: list[BatchRecord] = []
                    for batch_id in self._stored_ids(root_fd):
                        with self._locked_batch(root, root_fd, batch_id) as (_, batch_fd):
                            batches.append(self._load_batch_locked(batch_fd))
                    return tuple(batches)
            finally:
                context.__exit__(None, None, None)

    def save_batch(self, candidate: BatchRecord) -> None:
        candidate = BatchRecord.model_validate_json(candidate.model_dump_json())
        batch_id = self._validate_job_id(candidate.id)
        with self._local_lock, self._batch_root_anchor(create=False) as (root, root_fd):
            with self._locked_batch(root, root_fd, batch_id) as (_, batch_fd):
                current = self._load_batch_locked(batch_fd)
                self._validate_batch_transition(current, candidate)
                for child_id, outcome in candidate.child_outcomes.items():
                    if child_id not in current.child_outcomes and self.load(child_id).status is not outcome:
                        raise InvalidJobTransition(
                            "batch outcome does not match child job terminal status"
                        )
                if current != candidate:
                    self._write_batch_snapshot(candidate, batch_fd)

    def recover_batches(self) -> tuple[BatchRecord, ...]:
        """Fail interrupted batches; only never-started parents remain resumable."""
        with self._local_lock:
            try:
                context = self._batch_root_anchor(create=False)
                batch_root, batch_root_fd = context.__enter__()
            except JobNotFound:
                return ()
            try:
                with self._locked_fd(batch_root, batch_root_fd):
                    recovered: list[BatchRecord] = []
                    for batch_id in self._stored_ids(batch_root_fd):
                        with self._locked_batch(batch_root, batch_root_fd, batch_id) as (_, batch_fd):
                            record = self._load_batch_locked(batch_fd)
                            if record.status in {JobStage.BATCH_PROCESSING, JobStage.CANCELLING}:
                                outcomes = dict(record.child_outcomes)
                                for child_id in record.child_job_ids:
                                    child = self.load(child_id)
                                    if child.status is JobStage.QUEUED:
                                        child_progress = child.progress.advance(
                                            stage=JobStage.CANCELLING,
                                            percent=child.progress.percent,
                                            message="batch interrupted by restart",
                                        )
                                        child = self._updated_record(
                                            child,
                                            status=JobStage.CANCELLING,
                                            progress=child_progress,
                                        )
                                        self.save(child)
                                        child_progress = child.progress.advance(
                                            stage=JobStage.CANCELLED,
                                            percent=child.progress.percent,
                                            message="batch interrupted by restart",
                                        )
                                        child = self._updated_record(
                                            child,
                                            status=JobStage.CANCELLED,
                                            progress=child_progress,
                                            error="batch_interrupted_by_restart",
                                        )
                                        self.save(child)
                                    if child.status in _TERMINAL:
                                        outcomes[child_id] = child.status
                                record = self._updated_batch(
                                    record,
                                    status=JobStage.FAILED,
                                    error="interrupted_by_restart",
                                    active_child_job_id=None,
                                    child_outcomes=outcomes,
                                    success_count=sum(item is JobStage.COMPLETE for item in outcomes.values()),
                                    failure_count=sum(item is JobStage.FAILED for item in outcomes.values()),
                                    cancelled_count=sum(item is JobStage.CANCELLED for item in outcomes.values()),
                                    progress=record.progress.advance(
                                        stage=JobStage.FAILED,
                                        percent=record.progress.percent,
                                        message="interrupted_by_restart",
                                        completed_units=len(outcomes),
                                        total_units=len(record.child_job_ids),
                                    ),
                                )
                                self._write_batch_snapshot(record, batch_fd)
                            recovered.append(record)
                    return tuple(recovered)
            finally:
                context.__exit__(None, None, None)

    def save(self, record: JobRecord) -> None:
        candidate = JobRecord.model_validate_json(record.model_dump_json())
        job_id = self._validate_job_id(candidate.id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                current = self._load_locked(job_fd)
                self._validate_public_transition(current, candidate)
                if candidate != current:
                    self._write_snapshot(candidate, job_fd)

    def load(self, job_id: str) -> JobRecord:
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                return self._load_locked(job_fd)

    def append_transcript(self, job_id: str, segment: TranscriptSegment) -> JobRecord:
        segment = TranscriptSegment.model_validate_json(segment.model_dump_json())
        return self._append_unit(job_id, segment, "transcript", "transcript.jsonl")

    def append_timeline(self, job_id: str, entry: TimelineEntry) -> JobRecord:
        entry = TimelineEntry.model_validate_json(entry.model_dump_json())
        return self._append_unit(job_id, entry, "timeline", "timeline.jsonl")

    def append_text_timeline(self, job_id: str, entry: TextTimelineEntry) -> JobRecord:
        """Durably append one role-preserving, transcript-grounded analysis unit."""
        entry = TextTimelineEntry.model_validate_json(entry.model_dump_json())
        return self._append_unit(job_id, entry, "text_timeline", "text-timeline.jsonl")

    def append_transcription_window(self, job_id: str, result: WindowResult) -> JobRecord:
        """Durably append one completed transcription work unit."""
        result = WindowResult.model_validate_json(result.model_dump_json())
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                record = self._load_locked(job_fd)
                if record.status in _TERMINAL:
                    raise InvalidJobTransition("terminal jobs cannot receive partial results")
                journal_name = "transcription-windows.jsonl"
                journal = self._read_journal(job_fd, journal_name, WindowResult)
                if any(entry["unit"] == result for entry in journal):
                    return record
                progress = record.progress.advance(
                    stage=record.status,
                    percent=record.progress.percent,
                    message=record.progress.message,
                )
                self._append_journal(
                    job_fd,
                    journal_name,
                    {
                        "schema_version": 2,
                        "sequence": len(journal) + 1,
                        "revision": progress.revision,
                        "unit": result.model_dump(mode="json"),
                    },
                )
                updated = self._updated_record(record, progress=progress)
                self._write_snapshot(updated, job_fd)
                return updated

    def load_transcription_windows(self, job_id: str) -> tuple[WindowResult, ...]:
        """Read completed window checkpoints through the anchored job descriptor."""
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                self._load_locked(job_fd)
                entries = self._read_journal(
                    job_fd, "transcription-windows.jsonl", WindowResult
                )
                return tuple(entry["unit"] for entry in entries)

    def append_analysis_diagnostic(
        self, job_id: str, kind: str, payload: Mapping[str, object]
    ) -> JobRecord:
        """Journal retrieval/classification evidence independently of promotion."""
        if kind not in _DIAGNOSTIC_KINDS or not isinstance(payload, Mapping):
            raise ValueError("invalid analysis diagnostic")
        normalized = json.loads(json.dumps(payload, allow_nan=False))
        if kind == "verifier_attempt":
            normalized = _VerifierDiagnostic.model_validate_json(
                json.dumps(normalized, allow_nan=False)
            ).model_dump(mode="json")
        if kind == "coverage_policy":
            normalized = CoverageAssessment.model_validate_json(
                json.dumps(normalized, allow_nan=False)
            ).model_dump(mode="json")
        if kind == "topic_policy":
            normalized = TopicDecision.model_validate_json(
                json.dumps(normalized, allow_nan=False)
            ).model_dump(mode="json")
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                record = self._load_locked(job_fd)
                if record.status in _TERMINAL:
                    raise InvalidJobTransition("terminal jobs cannot receive diagnostics")
                entries = self._read_diagnostics(job_fd)
                progress = record.progress.advance(
                    stage=record.status,
                    percent=record.progress.percent,
                    message=record.progress.message,
                )
                self._append_journal(
                    job_fd,
                    "analysis-diagnostics.jsonl",
                    {
                        "schema_version": 2,
                        "sequence": len(entries) + 1,
                        "revision": progress.revision,
                        "kind": kind,
                        "payload": normalized,
                    },
                )
                updated = self._updated_record(record, progress=progress)
                self._write_snapshot(updated, job_fd)
                return updated

    def load_analysis_diagnostics(self, job_id: str) -> tuple[dict[str, object], ...]:
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                self._load_locked(job_fd)
                return tuple(self._read_diagnostics(job_fd))

    def promote_result(self, job_id: str, result: Mapping[str, object]) -> JobRecord:
        if not isinstance(result, Mapping):
            raise ValueError("result must be a JSON object")
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                current = self._load_locked(job_fd)
                if current.status is not JobStage.EXPORTING:
                    raise InvalidJobTransition("only exporting jobs can promote a result")
                completed = self._updated_record(
                    current,
                    status=JobStage.COMPLETE,
                    progress=current.progress.advance(
                        stage=JobStage.COMPLETE, percent=100, message="complete"
                    ),
                    ortb_content=result,
                )
                payload = json.loads(completed.model_dump_json())["ortb_content"]
                self._atomic_write_text(
                    job_fd, "result.json", serialize_content(payload).decode("utf-8")
                )
                self._write_snapshot(completed, job_fd)
                return completed

    def read_export(self, job_id: str) -> bytes | None:
        """Consume the completed result through the locked, no-follow job descriptor.

        Export callers receive bytes rather than a pathname, avoiding a race
        between releasing the store lock and a later caller-controlled open.
        """
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                record = self._load_locked(job_fd, validate_result=False)
                if record.status is not JobStage.COMPLETE:
                    return None
                contents = self._read_text(job_fd, "result.json")
                self._validate_complete_result(record, contents)
                return contents.encode("utf-8")

    def recover(self) -> tuple[JobRecord, ...]:
        with self._local_lock:
            root_context = self._root_anchor(create=False)
            try:
                root, root_fd = root_context.__enter__()
            except JobNotFound:
                return ()
            try:
                with self._locked_fd(root, root_fd):
                    recovered: list[JobRecord] = []
                    for job_id in self._stored_ids(root_fd):
                        try:
                            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                                record = self._load_locked(job_fd)
                                if record.status in _INTERRUPTED:
                                    record = self._updated_record(
                                        record,
                                        status=JobStage.FAILED,
                                        error="interrupted_by_restart",
                                        progress=record.progress.advance(
                                            stage=JobStage.FAILED,
                                            percent=record.progress.percent,
                                            message="interrupted_by_restart",
                                        ),
                                    )
                                    self._write_snapshot(record, job_fd)
                                recovered.append(record)
                        except JobNotFound:
                            self._reclaim_empty_reservation(root, root_fd, job_id)
                            continue
                    return tuple(recovered)
            finally:
                root_context.__exit__(None, None, None)

    def _append_unit(self, job_id: str, unit: _PartialUnit, attribute: str, journal_name: str) -> JobRecord:
        job_id = self._validate_job_id(job_id)
        with self._local_lock, self._root_anchor(create=False) as (root, root_fd):
            with self._locked_job(root, root_fd, job_id) as (_, job_fd):
                record = self._load_locked(job_fd)
                if record.status in _TERMINAL:
                    raise InvalidJobTransition("terminal jobs cannot receive partial results")
                existing = getattr(record, attribute)
                stored_unit = unit.model_dump(mode="json") if isinstance(unit, TextTimelineEntry) else unit
                comparison_existing = (
                    json.loads(json.dumps(record.model_dump(mode="json")[attribute]))
                    if isinstance(unit, TextTimelineEntry)
                    else existing
                )
                if stored_unit in comparison_existing:
                    return record
                journal = self._read_journal(job_fd, journal_name, type(unit))
                progress = record.progress.advance(
                    stage=record.status, percent=record.progress.percent, message=record.progress.message
                )
                self._append_journal(
                    job_fd,
                    journal_name,
                    {
                        "schema_version": 2,
                        "sequence": len(journal) + 1,
                        "revision": progress.revision,
                        "unit": unit.model_dump(mode="json"),
                    },
                )
                updated = self._updated_record(
                    record,
                    **{attribute: existing + (stored_unit,), "progress": progress},
                )
                self._write_snapshot(updated, job_fd)
                return updated

    def _load_locked(self, job_fd: int, *, validate_result: bool = True) -> JobRecord:
        snapshot = self._read_text(job_fd, "job.json")
        try:
            raw = json.loads(snapshot)
        except json.JSONDecodeError:
            # Keep Pydantic's detailed diagnostics for malformed snapshots.
            record = JobRecord.model_validate_json(snapshot)
        else:
            if not isinstance(raw, dict):
                record = JobRecord.model_validate(raw)
            else:
                self._require_schema_version(raw.get("schema_version"), artifact="job.json")
                record = JobRecord.model_validate(raw)
        record = self._reconcile_journals(record, job_fd)
        if validate_result and record.status is JobStage.COMPLETE:
            self._require_complete_result(record, job_fd)
        return record

    def _reconcile_journals(self, record: JobRecord, job_fd: int) -> JobRecord:
        current = record
        latest_revision = current.progress.revision
        for attribute, journal_name, unit_type in (
            ("transcript", "transcript.jsonl", TranscriptSegment),
            ("timeline", "timeline.jsonl", TimelineEntry),
            ("text_timeline", "text-timeline.jsonl", TextTimelineEntry),
        ):
            entries = self._read_journal(job_fd, journal_name, unit_type)
            snapshot_units = getattr(current, attribute)
            comparison_units = (
                json.loads(json.dumps(current.model_dump(mode="json")[attribute]))
                if unit_type is TextTimelineEntry
                else snapshot_units
            )
            if len(entries) < len(snapshot_units):
                raise CorruptJobJournal(f"{journal_name} is shorter than its snapshot")
            for index, entry in enumerate(entries[: len(snapshot_units)]):
                expected = (
                    entry["unit"].model_dump(mode="json")
                    if unit_type is TextTimelineEntry
                    else entry["unit"]
                )
                if expected != comparison_units[index]:
                    raise CorruptJobJournal(f"{journal_name} diverges at sequence {index + 1}")
            if len(entries) > len(snapshot_units):
                suffix = tuple(
                    entry["unit"].model_dump(mode="json")
                    if unit_type is TextTimelineEntry
                    else entry["unit"]
                    for entry in entries[len(snapshot_units) :]
                )
                current = self._updated_record(current, **{attribute: snapshot_units + suffix})
            if entries:
                latest_revision = max(latest_revision, entries[-1]["revision"])
        window_entries = self._read_journal(
            job_fd, "transcription-windows.jsonl", WindowResult
        )
        if window_entries:
            latest_revision = max(latest_revision, window_entries[-1]["revision"])
        diagnostic_entries = self._read_diagnostics(job_fd)
        if diagnostic_entries:
            latest_revision = max(latest_revision, diagnostic_entries[-1]["revision"])
        if current != record or latest_revision != current.progress.revision:
            progress = current.progress.model_dump(mode="json")
            progress.update(revision=latest_revision, updated_at=datetime.now(UTC))
            current = self._updated_record(current, progress=Progress.model_validate(progress))
            self._write_snapshot(current, job_fd)
        return current

    def _read_diagnostics(self, job_fd: int) -> list[dict[str, object]]:
        name = "analysis-diagnostics.jsonl"
        if not self._artifact_exists(job_fd, name, required=False):
            return []
        content = self._read_text(job_fd, name)
        if content and not content.endswith("\n"):
            raise CorruptJobJournal(f"{name} ends with an incomplete record")
        entries: list[dict[str, object]] = []
        previous_revision = -1
        for sequence, line in enumerate(content.splitlines(), start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorruptJobJournal(f"{name} contains invalid JSON") from exc
            if (
                not isinstance(item, dict)
                or set(item) != {"schema_version", "sequence", "revision", "kind", "payload"}
                or item.get("schema_version") != 2
                or item.get("sequence") != sequence
                or item.get("kind") not in _DIAGNOSTIC_KINDS
                or not isinstance(item.get("payload"), dict)
                or isinstance(item.get("revision"), bool)
                or not isinstance(item.get("revision"), int)
                or item["revision"] <= previous_revision
            ):
                raise CorruptJobJournal(f"{name} has an invalid record")
            if item["kind"] == "topic_policy":
                try:
                    TopicDecision.model_validate_json(json.dumps(item["payload"], allow_nan=False))
                except ValueError as exc:
                    raise CorruptJobJournal(f"{name} has an invalid topic policy") from exc
            if item["kind"] == "coverage_policy":
                try:
                    CoverageAssessment.model_validate_json(json.dumps(item["payload"], allow_nan=False))
                except ValueError as exc:
                    raise CorruptJobJournal(f"{name} has an invalid coverage policy") from exc
            if item["kind"] == "verifier_attempt":
                try:
                    _VerifierDiagnostic.model_validate_json(json.dumps(item["payload"], allow_nan=False))
                except ValueError as exc:
                    raise CorruptJobJournal(f"{name} has an invalid verifier attempt") from exc
            previous_revision = item["revision"]
            entries.append(item)
        return entries

    @staticmethod
    def _updated_record(record: JobRecord, **updates: object) -> JobRecord:
        payload = record.model_dump(mode="json")
        payload.update(updates)
        payload["updated_at"] = datetime.now(UTC)
        return JobRecord.model_validate(payload)

    @staticmethod
    def _updated_batch(record: BatchRecord, **updates: object) -> BatchRecord:
        payload = record.model_dump(mode="json")
        payload.update(updates)
        payload["updated_at"] = datetime.now(UTC)
        return BatchRecord.model_validate(payload)

    def _validate_batch_transition(self, current: BatchRecord, candidate: BatchRecord) -> None:
        if candidate.id != current.id:
            raise InvalidJobTransition("batch IDs cannot change")
        if candidate.source_directory != current.source_directory or candidate.child_job_ids != current.child_job_ids:
            raise InvalidJobTransition("batch input identity cannot change")
        if candidate == current:
            return
        if current.status in _BATCH_TERMINAL:
            raise InvalidJobTransition("terminal batch records are immutable")
        if not set(current.child_outcomes).issubset(candidate.child_outcomes):
            raise InvalidJobTransition("batch child outcomes cannot be removed")
        for child_id, outcome in current.child_outcomes.items():
            if candidate.child_outcomes[child_id] is not outcome:
                raise InvalidJobTransition("terminal child outcomes cannot change")
        if candidate.progress.revision <= current.progress.revision:
            raise JobConflict("batch update has an obsolete progress revision")
        if candidate.progress.percent < current.progress.percent:
            raise InvalidJobTransition("batch progress cannot decrease")
        if candidate.updated_at < current.updated_at:
            raise InvalidJobTransition("batch updated_at cannot decrease")
        allowed = {
            JobStage.QUEUED: {JobStage.BATCH_PROCESSING, JobStage.CANCELLED, JobStage.FAILED},
            JobStage.BATCH_PROCESSING: {
                JobStage.BATCH_PROCESSING,
                JobStage.CANCELLING,
                JobStage.COMPLETE,
                JobStage.CANCELLED,
                JobStage.FAILED,
            },
            JobStage.CANCELLING: {JobStage.CANCELLED, JobStage.FAILED},
        }
        if candidate.status is not current.status and candidate.status not in allowed.get(current.status, set()):
            raise InvalidJobTransition(
                f"cannot transition batch from {current.status.value} to {candidate.status.value}"
            )

    def _load_batch_locked(self, batch_fd: int) -> BatchRecord:
        contents = self._read_text(batch_fd, "batch.json")
        try:
            raw = json.loads(contents)
        except json.JSONDecodeError as exc:
            raise CorruptJobState("batch snapshot is invalid JSON") from exc
        if (
            not isinstance(raw, dict)
            or isinstance(raw.get("schema_version"), bool)
            or raw.get("schema_version") != 1
        ):
            raise UnsupportedJobSchemaVersion("unsupported batch schema version")
        try:
            return BatchRecord.model_validate(raw)
        except Exception as exc:
            raise CorruptJobState("batch snapshot is malformed") from exc

    def _write_batch_snapshot(self, record: BatchRecord, batch_fd: int) -> None:
        self._atomic_write_text(batch_fd, "batch.json", record.model_dump_json())

    @contextmanager
    def _batch_root_anchor(self, *, create: bool) -> Iterator[tuple[Path, int]]:
        with self._root_anchor(create=create) as (root, root_fd):
            try:
                batch_fd = self._open_component(root_fd, ".batches")
            except FileNotFoundError:
                if not create:
                    raise JobNotFound(root / ".batches")
                try:
                    os.mkdir(".batches", dir_fd=root_fd)
                except FileExistsError:
                    pass
                self._fsync_directory(root, root_fd)
                try:
                    batch_fd = self._open_component(root_fd, ".batches")
                except OSError as exc:
                    raise UnsafeJobPath("cannot safely open batch store") from exc
            except OSError as exc:
                raise UnsafeJobPath("cannot safely open batch store") from exc
            try:
                yield root / ".batches", batch_fd
            finally:
                os.close(batch_fd)

    @contextmanager
    def _locked_batch(
        self, root: Path, root_fd: int, batch_id: str
    ) -> Iterator[tuple[Path, int]]:
        batch_fd = self._open_job_fd(root_fd, batch_id)
        try:
            with self._locked_fd(root / batch_id, batch_fd):
                yield root / batch_id, batch_fd
        finally:
            os.close(batch_fd)

    def _cleanup_new_batch(self, root: Path, root_fd: int, batch_id: str) -> None:
        try:
            batch_fd = self._open_job_fd(root_fd, batch_id)
        except (JobNotFound, UnsafeJobPath):
            return
        try:
            for name in os.listdir(batch_fd):
                self._artifact_exists(batch_fd, name, required=True)
                os.unlink(name, dir_fd=batch_fd)
        finally:
            os.close(batch_fd)
        os.rmdir(batch_id, dir_fd=root_fd)
        self._fsync_directory(root, root_fd)

    def _validate_public_transition(self, current: JobRecord, candidate: JobRecord) -> None:
        if candidate.id != current.id:
            raise InvalidJobTransition("job IDs cannot change")
        if candidate == current:
            return
        if current.status in _TERMINAL:
            raise InvalidJobTransition("terminal job records are immutable")
        if (
            candidate.transcript != current.transcript
            or candidate.timeline != current.timeline
            or candidate.text_timeline != current.text_timeline
        ):
            raise InvalidJobTransition("partial arrays are owned by append/replay only")
        if candidate.progress.revision <= current.progress.revision:
            raise JobConflict("non-idempotent update has an obsolete progress revision")
        if candidate.progress.percent < current.progress.percent:
            raise InvalidJobTransition("progress percent cannot decrease")
        if candidate.updated_at < current.updated_at:
            raise InvalidJobTransition("updated_at cannot decrease")
        if candidate.status is current.status:
            return
        if candidate.status is JobStage.COMPLETE:
            raise InvalidJobTransition("only promote_result can complete a job")
        if candidate.status not in _ALLOWED[current.status]:
            raise InvalidJobTransition(
                f"cannot transition from {current.status.value} to {candidate.status.value}"
            )

    @classmethod
    def _stored_ids(cls, root_fd: int) -> Iterator[str]:
        """Ignore unrelated entries without relaxing validation of record paths."""
        for name in sorted(os.listdir(root_fd)):
            try:
                yield cls._validate_job_id(name)
            except UnsafeJobId:
                continue

    @staticmethod
    def _validate_job_id(job_id: str) -> str:
        if not isinstance(job_id, str):
            raise UnsafeJobId("job ID must be a canonical UUID string")
        try:
            parsed = UUID(job_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise UnsafeJobId("job ID must be a canonical UUID string") from exc
        if str(parsed) != job_id:
            raise UnsafeJobId("job ID must be a canonical UUID string")
        return job_id

    @contextmanager
    def _root_anchor(self, *, create: bool) -> Iterator[tuple[Path, int]]:
        root = self.jobs_root if self.jobs_root.is_absolute() else Path.cwd() / self.jobs_root
        components = root.parts[1:]
        if not root.is_absolute() or any(component in {"", ".", ".."} for component in components):
            raise UnsafeJobPath(f"jobs root must be a normalized absolute path: {root}")
        current_path = Path(root.anchor)
        try:
            descriptor = os.open(root.anchor, _DIR_FLAGS)
        except OSError as exc:
            raise UnsafeJobPath(f"cannot safely open filesystem root: {root.anchor}") from exc
        try:
            for component in components:
                try:
                    child_descriptor = self._open_component(descriptor, component)
                except FileNotFoundError:
                    if not create:
                        raise JobNotFound(root)
                    try:
                        os.mkdir(component, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    self._fsync_directory(current_path, descriptor)
                    try:
                        child_descriptor = self._open_component(descriptor, component)
                    except OSError as exc:
                        raise UnsafeJobPath(f"cannot safely create jobs-root component: {component}") from exc
                except OSError as exc:
                    raise UnsafeJobPath(f"cannot safely open jobs-root component: {component}") from exc
                os.close(descriptor)
                descriptor = child_descriptor
                current_path /= component
            yield current_path, descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_component(parent_fd: int, name: str) -> int:
        return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)

    @contextmanager
    def _root_lock(self, *, create: bool = True) -> Iterator[tuple[Path, int]]:
        with self._root_anchor(create=create) as (root, root_fd):
            with self._locked_fd(root, root_fd):
                yield root, root_fd

    @contextmanager
    def _locked_job(self, root: Path, root_fd: int, job_id: str) -> Iterator[tuple[Path, int]]:
        path = root / job_id
        job_fd = self._open_job_fd(root_fd, job_id)
        try:
            with self._locked_fd(path, job_fd):
                yield path, job_fd
        finally:
            os.close(job_fd)

    def _open_job_fd(self, root_fd: int, job_id: str) -> int:
        try:
            return os.open(job_id, _DIR_FLAGS, dir_fd=root_fd)
        except FileNotFoundError as exc:
            raise JobNotFound(job_id) from exc
        except OSError as exc:
            raise UnsafeJobPath(f"cannot safely open job directory: {job_id}") from exc

    @contextmanager
    def _locked_fd(self, directory: Path, directory_fd: int) -> Iterator[None]:
        existed = self._artifact_exists(directory_fd, ".store.lock", required=False)
        try:
            descriptor = os.open(
                ".store.lock", os.O_RDWR | os.O_CREAT | _NOFOLLOW, 0o600, dir_fd=directory_fd
            )
        except OSError as exc:
            raise UnsafeJobPath(f"cannot safely open lock under {directory}") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise UnsafeJobPath(f"lock is not a regular file under {directory}")
            if not existed:
                self._fsync_directory(directory, directory_fd)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _write_snapshot(self, record: JobRecord, job_fd: int) -> None:
        self._atomic_write_text(job_fd, "job.json", record.model_dump_json())

    def _atomic_write_text(self, job_fd: int, name: str, contents: str) -> None:
        temporary = f".{name}.{uuid4().hex}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600, dir_fd=job_fd
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = None
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            self._artifact_exists(job_fd, name, required=False)
            os.replace(temporary, name, src_dir_fd=job_fd, dst_dir_fd=job_fd)
            os.fsync(job_fd)
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=job_fd)
            except FileNotFoundError:
                pass
            raise

    def _append_journal(self, job_fd: int, name: str, item: Mapping[str, object]) -> None:
        existed = self._artifact_exists(job_fd, name, required=False)
        try:
            descriptor = os.open(
                name, os.O_WRONLY | os.O_APPEND | os.O_CREAT | _NOFOLLOW, 0o600, dir_fd=job_fd
            )
        except OSError as exc:
            raise UnsafeJobPath(f"cannot safely open journal: {name}") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise UnsafeJobPath(f"journal is not regular: {name}")
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                descriptor = None
                handle.write(json.dumps(item, allow_nan=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if not existed:
                os.fsync(job_fd)
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _read_text(self, job_fd: int, name: str) -> str:
        self._artifact_exists(job_fd, name, required=True)
        try:
            descriptor = os.open(name, os.O_RDONLY | _NOFOLLOW, dir_fd=job_fd)
        except OSError as exc:
            raise UnsafeJobPath(f"cannot safely open artifact: {name}") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise UnsafeJobPath(f"artifact is not regular: {name}")
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = None
                return handle.read()
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _artifact_exists(self, directory_fd: int, name: str, *, required: bool) -> bool:
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            if required:
                raise JobNotFound(name)
            return False
        if stat.S_ISLNK(info.st_mode):
            raise UnsafeJobPath(f"artifact is a symlink: {name}")
        return True

    def _read_journal(
        self, job_fd: int, name: str, unit_type: type[_PartialUnit]
    ) -> tuple[dict[str, object], ...]:
        if not self._artifact_exists(job_fd, name, required=False):
            return ()
        contents = self._read_text(job_fd, name)
        if contents and not contents.endswith("\n"):
            raise CorruptJobJournal(f"{name} ends with an incomplete record")
        entries: list[dict[str, object]] = []
        last_revision = -1
        for sequence, line in enumerate(contents.splitlines(), start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorruptJobJournal(f"{name} contains invalid JSON") from exc
            if not isinstance(payload, dict):
                raise CorruptJobJournal(f"{name} has an invalid record shape")
            self._require_schema_version(payload.get("schema_version"), artifact=name)
            if set(payload) != {"schema_version", "sequence", "revision", "unit"}:
                raise CorruptJobJournal(f"{name} has an invalid record shape")
            revision = payload["revision"]
            if (
                payload["sequence"] != sequence
                or isinstance(payload["sequence"], bool)
                or not isinstance(payload["sequence"], int)
                or isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 0
                or revision <= last_revision
            ):
                raise CorruptJobJournal(f"{name} has non-monotonic sequence or revision")
            try:
                if unit_type is TextTimelineEntry:
                    unit = unit_type.model_validate_json(json.dumps(payload["unit"]))
                else:
                    unit = unit_type.model_validate(payload["unit"])
            except Exception as exc:
                raise CorruptJobJournal(f"{name} has an invalid completed unit") from exc
            entries.append(
                {
                    "schema_version": 2,
                    "sequence": sequence,
                    "revision": revision,
                    "unit": unit,
                }
            )
            last_revision = revision
        return tuple(entries)

    @staticmethod
    def _require_schema_version(value: object, *, artifact: str) -> None:
        if value != 2 or isinstance(value, bool):
            version = "missing" if value is None else repr(value)
            raise UnsupportedJobSchemaVersion(
                f"unsupported schema version for {artifact}: {version}"
            )

    def _require_complete_result(self, record: JobRecord, job_fd: int) -> None:
        if record.ortb_content is None:
            raise CorruptJobState("complete job has no durable OpenRTB content")
        try:
            contents = self._read_text(job_fd, "result.json")
        except (JobNotFound, json.JSONDecodeError) as exc:
            raise CorruptJobState("complete job has no valid durable result") from exc
        self._validate_complete_result(record, contents)

    @staticmethod
    def _validate_complete_result(record: JobRecord, contents: str) -> None:
        try:
            payload = json.loads(contents)
        except json.JSONDecodeError as exc:
            raise CorruptJobState("complete job has no valid durable result") from exc
        if payload != json.loads(record.model_dump_json())["ortb_content"]:
            raise CorruptJobState("complete result does not match its snapshot")

    def _cleanup_new_directory(self, root: Path, root_fd: int, job_id: str) -> None:
        try:
            job_fd = self._open_job_fd(root_fd, job_id)
        except (JobNotFound, UnsafeJobPath):
            return
        try:
            names = os.listdir(job_fd)
            allowed = {"job.json", ".store.lock"}
            if any(
                name not in allowed and not (name.startswith(".job.json.") and name.endswith(".tmp"))
                for name in names
            ):
                return
            for name in names:
                self._artifact_exists(job_fd, name, required=True)
                os.unlink(name, dir_fd=job_fd)
        finally:
            os.close(job_fd)
        os.rmdir(job_id, dir_fd=root_fd)
        self._fsync_directory(root, root_fd)

    def _reclaim_empty_reservation(self, root: Path, root_fd: int, job_id: str) -> bool:
        """Reclaim only a pre-snapshot directory containing lock/temp residue."""
        try:
            job_fd = self._open_job_fd(root_fd, job_id)
        except JobNotFound:
            return False
        try:
            names = os.listdir(job_fd)
            for name in names:
                if name == ".store.lock" or (
                    name.startswith(".job.json.") and name.endswith(".tmp")
                ):
                    self._artifact_exists(job_fd, name, required=True)
                    continue
                return False
            for name in names:
                os.unlink(name, dir_fd=job_fd)
        finally:
            os.close(job_fd)
        try:
            os.rmdir(job_id, dir_fd=root_fd)
        except FileNotFoundError:
            return False
        self._fsync_directory(root, root_fd)
        return True

    @staticmethod
    def _fsync_directory(directory: Path, descriptor: int) -> None:
        os.fsync(descriptor)
