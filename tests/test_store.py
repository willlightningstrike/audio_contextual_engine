from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from engine.domain import JobRecord, JobRequest, SignalSet, TimeRange, TimelineEntry, TranscriptSegment
from engine.events import Progress
from engine.lifecycle import JobStage
from engine.store import (
    CorruptJobJournal,
    CorruptJobState,
    InvalidJobTransition,
    JobAlreadyExists,
    JobConflict,
    JobStore,
    UnsafeJobId,
    UnsafeJobPath,
)


JOB_ID = UUID("12345678-1234-5678-1234-567812345678")
_ORDER = (
    JobStage.QUEUED,
    JobStage.EXTRACTING_AUDIO,
    JobStage.DETECTING_LANGUAGE,
    JobStage.TRANSCRIBING,
    JobStage.ANALYSING,
    JobStage.EXPORTING,
)


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "data" / "jobs", uuid_factory=lambda: JOB_ID)


def segment() -> TranscriptSegment:
    return TranscriptSegment(
        time_range=TimeRange(start_seconds=0, end_seconds=1), text="hello", confidence=0.9
    )


def entry() -> TimelineEntry:
    return TimelineEntry(time_range=TimeRange(start_seconds=0, end_seconds=1), signals=SignalSet())


def _test_job_directory(store: JobStore, job_id: str) -> Path:
    """Test-only artifact setup beneath the fixture's known-safe temporary root."""
    return store.jobs_root / job_id


def advance_to(store: JobStore, target: JobStage) -> JobRecord:
    job = store.create(JobRequest(source="input.wav"))
    if target is JobStage.QUEUED:
        return job
    if target in _ORDER:
        statuses = _ORDER[1 : _ORDER.index(target) + 1]
    elif target is JobStage.CANCELLING:
        statuses = (JobStage.CANCELLING,)
    elif target is JobStage.CANCELLED:
        statuses = (JobStage.CANCELLING, JobStage.CANCELLED)
    elif target is JobStage.FAILED:
        statuses = (JobStage.FAILED,)
    else:
        raise AssertionError("complete is reached only through promote_result")
    for status in statuses:
        job = JobRecord.model_validate(
            {
                **job.model_dump(mode="json"),
                "status": status,
                "progress": job.progress.advance(stage=status, percent=40, message=status.value),
            }
        )
        store.save(job)
    return job


def test_create_calls_uuid_factory_once_reserves_one_directory_and_fsyncs_jobs_root(store: JobStore, monkeypatch):
    calls = 0
    synced: list[Path] = []

    def new_id() -> UUID:
        nonlocal calls
        calls += 1
        return JOB_ID

    store = JobStore(store.jobs_root, uuid_factory=new_id)
    monkeypatch.setattr(store, "_fsync_directory", lambda path, descriptor: synced.append(path))
    job = store.create(JobRequest(source="input.wav"))

    assert calls == 1
    assert job.status is JobStage.QUEUED
    assert store.load(job.id) == job
    assert store.jobs_root.parent in synced
    assert store.jobs_root in synced
    assert store.jobs_root / job.id in synced
    assert [path for path in store.jobs_root.iterdir() if path.is_dir()] == [store.jobs_root / job.id]


def test_new_jobs_root_fsyncs_its_parent_directory_entry(store: JobStore, monkeypatch):
    synced: list[Path] = []
    original = store._fsync_directory

    def record_fsync(path: Path, descriptor: int) -> None:
        synced.append(path)
        original(path, descriptor)

    monkeypatch.setattr(store, "_fsync_directory", record_fsync)
    store.create(JobRequest(source="input.wav"))

    assert store.jobs_root.parent in synced


def test_nested_jobs_root_fsyncs_each_parent_that_gained_a_new_child(tmp_path: Path, monkeypatch):
    root = tmp_path / "base" / "data" / "jobs"
    store = JobStore(root, uuid_factory=lambda: JOB_ID)
    synced: list[Path] = []
    original = store._fsync_directory

    def record_fsync(path: Path, descriptor: int) -> None:
        synced.append(path)
        original(path, descriptor)

    monkeypatch.setattr(store, "_fsync_directory", record_fsync)
    store.create(JobRequest(source="input.wav"))

    assert {tmp_path, tmp_path / "base", tmp_path / "base" / "data"} <= set(synced)


def test_component_anchored_root_creation_cannot_follow_an_ancestor_swap(tmp_path: Path, monkeypatch):
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    relocated = tmp_path / "relocated-base"
    store = JobStore(base / "data" / "jobs", uuid_factory=lambda: JOB_ID)
    open_component = store._open_component
    swapped = False

    def swap_after_open(parent_fd: int, name: str) -> int:
        nonlocal swapped
        descriptor = open_component(parent_fd, name)
        if name == "base" and not swapped:
            base.rename(relocated)
            base.symlink_to(outside, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(store, "_open_component", swap_after_open)
    job = store.create(JobRequest(source="input.wav"))

    assert swapped
    assert (relocated / "data" / "jobs" / job.id / "job.json").exists()
    assert not (outside / "data").exists()


def test_create_collision_never_changes_existing_snapshot_or_artifacts(store: JobStore):
    original = store.create(JobRequest(source="first.wav"))
    artifact = _test_job_directory(store, original.id) / "transcript.jsonl"
    artifact.write_text('{"preserve": true}\n')
    before_snapshot = (_test_job_directory(store, original.id) / "job.json").read_bytes()

    with pytest.raises(JobAlreadyExists):
        store.create(JobRequest(source="second.wav"))

    assert (_test_job_directory(store, original.id) / "job.json").read_bytes() == before_snapshot
    assert artifact.read_text() == '{"preserve": true}\n'


def test_initial_snapshot_failure_cleans_only_newly_reserved_directory(store: JobStore, monkeypatch):
    def fail_write(*args, **kwargs):
        raise OSError("injected initial write failure")

    monkeypatch.setattr(store, "_write_snapshot", fail_write)
    with pytest.raises(OSError, match="injected"):
        store.create(JobRequest(source="input.wav"))

    assert not (store.jobs_root / str(JOB_ID)).exists()


def test_recovery_reclaims_an_empty_early_reservation_without_hiding_valid_jobs(store: JobStore):
    valid = store.create(JobRequest(source="valid.wav"))
    orphan_id = str(UUID(int=1))
    orphan = _test_job_directory(store, orphan_id)
    orphan.mkdir()

    recovered = store.recover()

    assert recovered == (valid,)
    assert not orphan.exists()
    retry = JobStore(store.jobs_root, uuid_factory=lambda: UUID(int=1)).create(
        JobRequest(source="retry.wav")
    )
    assert retry.id == orphan_id


def test_recovery_on_an_absent_root_is_non_mutating(tmp_path: Path):
    root = tmp_path / "missing" / "data" / "jobs"

    assert JobStore(root, uuid_factory=lambda: JOB_ID).recover() == ()
    assert not root.exists()


@pytest.mark.parametrize("method", ["recover", "list_batches", "recover_batches"])
def test_record_discovery_ignores_unrelated_files_and_directories(tmp_path: Path, method: str):
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source="valid.wav"))
    if method == "recover":
        root, expected = store.jobs_root, job
    else:
        expected = store.create_batch(tmp_path, (job.id,))
        root = store.jobs_root / ".batches"
    metadata = root / ".DS_Store"
    metadata.write_bytes(b"Finder metadata")
    (root / "notes.txt").write_text("not a job")
    (root / "unrelated-directory").mkdir()

    assert getattr(store, method)() == (expected,)
    assert metadata.read_bytes() == b"Finder metadata"
    assert (root / "notes.txt").read_text() == "not a job"
    assert (root / "unrelated-directory").is_dir()


@pytest.mark.parametrize("method", ["list_batches", "recover_batches"])
def test_batch_discovery_still_rejects_uuid_named_symlinks(tmp_path: Path, method: str):
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source="valid.wav"))
    store.create_batch(tmp_path, (job.id,))
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.jobs_root / ".batches" / str(UUID(int=1))).symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeJobPath):
        getattr(store, method)()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("unsafe_id", ["../x", "a/b", ".", "", "not-a-uuid"])
def test_public_id_boundaries_reject_path_traversal(store: JobStore, unsafe_id: str):
    for action in (
        lambda: store.load(unsafe_id),
        lambda: store.read_export(unsafe_id),
        lambda: store.append_transcript(unsafe_id, segment()),
        lambda: store.append_timeline(unsafe_id, entry()),
        lambda: store.promote_result(unsafe_id, {"id": "safe"}),
    ):
        with pytest.raises(UnsafeJobId):
            action()


def test_root_symlink_is_rejected_before_any_job_is_created(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root_link = tmp_path / "jobs-link"
    root_link.symlink_to(outside, target_is_directory=True)
    store = JobStore(root_link, uuid_factory=lambda: JOB_ID)

    with pytest.raises(UnsafeJobPath):
        store.create(JobRequest(source="input.wav"))

    assert list(outside.iterdir()) == []


def test_jobs_root_with_a_symlinked_ancestor_is_rejected(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    ancestor = tmp_path / "linked-ancestor"
    ancestor.symlink_to(outside, target_is_directory=True)
    store = JobStore(ancestor / "data" / "jobs", uuid_factory=lambda: JOB_ID)

    with pytest.raises(UnsafeJobPath):
        store.create(JobRequest(source="input.wav"))

    assert list(outside.iterdir()) == []


def test_uuid_named_job_directory_symlink_cannot_read_write_or_recover_outside_root(tmp_path: Path):
    root = tmp_path / "data" / "jobs"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    original = JobRecord(id=str(JOB_ID), request=JobRequest(source="outside.wav"))
    external_snapshot = outside / "job.json"
    external_snapshot.write_text(original.model_dump_json())
    (root / str(JOB_ID)).symlink_to(outside, target_is_directory=True)
    store = JobStore(root, uuid_factory=lambda: JOB_ID)

    for action in (
        lambda: store.load(str(JOB_ID)),
        lambda: store.save(original),
        store.recover,
    ):
        with pytest.raises(UnsafeJobPath):
            action()

    assert external_snapshot.read_text() == original.model_dump_json()


def test_symlinked_result_artifact_cannot_be_replaced_outside_job_directory(store: JobStore):
    exporting = advance_to(store, JobStage.EXPORTING)
    outside = store.jobs_root.parent / "outside-result.json"
    outside.write_text('{"outside": true}')
    result = _test_job_directory(store, exporting.id) / "result.json"
    result.symlink_to(outside)

    with pytest.raises(UnsafeJobPath):
        store.promote_result(exporting.id, {"id": exporting.id})

    assert outside.read_text() == '{"outside": true}'
    assert store.load(exporting.id).status is JobStage.EXPORTING


def test_snapshot_writes_use_unique_exclusive_temps_and_cleanup_after_failure(store: JobStore, monkeypatch):
    job = advance_to(store, JobStage.EXTRACTING_AUDIO)
    seen: list[Path] = []

    def fail_replace(source, destination, **kwargs):
        seen.append((source, destination, kwargs))
        raise OSError("injected replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    update = JobRecord.model_validate(
        {
            **job.model_dump(mode="json"),
            "progress": job.progress.advance(stage=job.status, percent=40, message="updated"),
        }
    )
    with pytest.raises(OSError, match="injected"):
        store.save(update)

    assert len(seen) == 1
    assert seen[0][0].startswith(".job.json.")
    assert seen[0][1] == "job.json"
    assert seen[0][2]["src_dir_fd"] == seen[0][2]["dst_dir_fd"]
    assert list(_test_job_directory(store, job.id).glob("*.tmp")) == []


def test_two_store_instances_share_an_exclusive_job_lock_for_read_and_write(store: JobStore, monkeypatch):
    job = advance_to(store, JobStage.EXTRACTING_AUDIO)
    second = JobStore(store.jobs_root, uuid_factory=lambda: UUID(int=2))
    update = JobRecord.model_validate(
        {
            **job.model_dump(mode="json"),
            "progress": job.progress.advance(stage=job.status, percent=40, message="updated"),
        }
    )
    entered = threading.Event()
    release = threading.Event()
    real_replace = os.replace

    def delayed_replace(source, destination, **kwargs):
        entered.set()
        assert release.wait(timeout=1)
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", delayed_replace)
    writer = threading.Thread(target=lambda: store.save(update))
    writer.start()
    assert entered.wait(timeout=1)
    loaded: list[JobRecord] = []
    reader = threading.Thread(target=lambda: loaded.append(second.load(job.id)))
    reader.start()
    assert reader.is_alive()
    release.set()
    writer.join(timeout=1)
    reader.join(timeout=1)

    assert loaded == [update]


def test_append_journal_entries_are_sequence_revisioned_durable_and_idempotent(store: JobStore, monkeypatch):
    job = advance_to(store, JobStage.TRANSCRIBING)
    fsyncs = 0
    real_fsync = os.fsync

    def record_fsync(descriptor: int):
        nonlocal fsyncs
        fsyncs += 1
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)

    first = store.append_transcript(job.id, segment())
    retry = store.append_transcript(job.id, segment())

    journal = json.loads((_test_job_directory(store, job.id) / "transcript.jsonl").read_text())
    assert journal["schema_version"] == 2
    assert journal["sequence"] == 1
    assert journal["revision"] == first.progress.revision
    assert journal["unit"] == segment().model_dump(mode="json")
    assert retry == first
    assert first.transcript == (segment(),)
    assert fsyncs >= 3


def test_timeline_journal_is_durable_and_round_trips_through_its_snapshot(store: JobStore):
    job = advance_to(store, JobStage.ANALYSING)

    updated = store.append_timeline(job.id, entry())

    journal = json.loads((_test_job_directory(store, job.id) / "timeline.jsonl").read_text())
    assert journal == {
        "schema_version": 2,
        "sequence": 1,
        "revision": updated.progress.revision,
        "unit": entry().model_dump(mode="json"),
    }
    assert store.load(job.id).timeline == (entry(),)


def test_role_preserving_timeline_journal_is_durable_and_idempotent(store: JobStore):
    from engine.signals.result_types import (
        AudioTranscriptProvenance,
        TextTimelineEntry,
    )
    from engine.signals.text_types import ClassificationPayload, SentimentValue

    job = advance_to(store, JobStage.ANALYSING)
    classified = TextTimelineEntry(
        time_range=TimeRange(start_seconds=0, end_seconds=1),
        payload=ClassificationPayload(
            primary_category="633",
            secondary_categories=("255",),
            keywords=(),
            entities=(),
            sentiment=SentimentValue(label="neutral", score=0),
        ),
        evidence_ranges=(TimeRange(start_seconds=0, end_seconds=1),),
        provenance=AudioTranscriptProvenance(language="en"),
    )

    store.append_text_timeline(job.id, classified)
    store.append_text_timeline(job.id, classified)

    restored = store.load(job.id)
    assert len(restored.text_timeline) == 1
    assert restored.text_timeline[0]["payload"]["primary_category"] == "633"


def test_topic_policy_diagnostic_is_validated_and_durable(store: JobStore):
    job = advance_to(store, JobStage.ANALYSING)
    policy = {
        "policy_version": "english-audio-aggregation-v2",
        "outcome": "insufficient",
        "primary_category": None,
        "transcript_seconds": 3.0,
        "audio_seconds": 30.0,
        "primary_role_support_seconds": [["633", 3.0]],
        "secondary_role_support_seconds": [],
    }
    store.append_analysis_diagnostic(job.id, "topic_policy", policy)
    assert store.load_analysis_diagnostics(job.id)[0]["payload"] == policy
    with pytest.raises(ValueError):
        store.append_analysis_diagnostic(job.id, "topic_policy", {**policy, "outcome": "eligible"})
    assert len(store.load_analysis_diagnostics(job.id)) == 1


def _verifier_envelope():
    return {
        "time_range": {"start_seconds": 0, "end_seconds": 3},
        "policy_version": "english-transcript-verifier-v1",
        "model": {"repository": "mlx-community/Qwen2.5-32B-Instruct-4bit",
                  "revision": "2938092373e5f97b95538884112085364c2da315",
                  "manifest_sha256": "a" * 64},
        "attempt": {"category_id": "633", "prompt_sha256": "b" * 64,
                    "raw": '{"status":"invalid"}', "finish_reason": "stop",
                    "seconds": 0.01, "token_count": 5, "error_code": None},
    }


def test_verifier_diagnostic_round_trips_and_is_validated_on_replay(store):
    job = advance_to(store, JobStage.ANALYSING)
    envelope = _verifier_envelope()
    store.append_analysis_diagnostic(job.id, "verifier_attempt", envelope)
    reopened = JobStore(store.jobs_root)
    assert reopened.load_analysis_diagnostics(job.id)[0]["payload"] == envelope
    path = _test_job_directory(store, job.id) / "analysis-diagnostics.jsonl"
    record = json.loads(path.read_text())
    record["payload"]["model"]["manifest_sha256"] = "bad"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(CorruptJobJournal, match="invalid verifier attempt"):
        reopened.load_analysis_diagnostics(job.id)


@pytest.mark.parametrize("defect", ["policy", "missing_model", "hash", "extra_attempt_field"])
def test_malformed_verifier_envelope_never_reaches_journal(store, defect):
    job = advance_to(store, JobStage.ANALYSING)
    envelope = _verifier_envelope()
    if defect == "policy":
        envelope["policy_version"] = "unapproved"
    elif defect == "missing_model":
        del envelope["model"]
    elif defect == "hash":
        envelope["attempt"]["prompt_sha256"] = "bad"
    else:
        envelope["attempt"]["extra"] = True
    with pytest.raises(ValueError):
        store.append_analysis_diagnostic(job.id, "verifier_attempt", envelope)
    assert store.load_analysis_diagnostics(job.id) == ()


def test_valid_durable_journal_suffix_is_reconciled_after_snapshot_crash(store: JobStore, monkeypatch):
    job = advance_to(store, JobStage.TRANSCRIBING)
    original_write = store._write_snapshot

    def fail_snapshot(record, directory):
        raise OSError("crash after journal fsync")

    monkeypatch.setattr(store, "_write_snapshot", fail_snapshot)
    with pytest.raises(OSError, match="crash"):
        store.append_transcript(job.id, segment())
    monkeypatch.setattr(store, "_write_snapshot", original_write)

    recovered = JobStore(store.jobs_root, uuid_factory=lambda: UUID(int=2)).load(job.id)

    assert recovered.transcript == (segment(),)
    assert recovered.progress.revision == job.progress.revision + 1


def test_divergent_or_corrupt_journal_prefix_is_rejected_without_cleanup(store: JobStore):
    job = advance_to(store, JobStage.TRANSCRIBING)
    store.append_transcript(job.id, segment())
    journal_path = _test_job_directory(store, job.id) / "transcript.jsonl"
    journal_path.write_text(
        '{"schema_version":2,"sequence":1,"revision":4,"unit":{"bad":"shape"}}\n'
    )

    with pytest.raises(CorruptJobJournal):
        store.load(job.id)

    assert journal_path.exists()


def test_terminal_append_is_rejected_and_terminal_snapshots_are_immutable(store: JobStore):
    cancelled = advance_to(store, JobStage.CANCELLED)
    with pytest.raises(InvalidJobTransition):
        store.append_timeline(cancelled.id, entry())
    changed = JobRecord.model_validate({**cancelled.model_dump(mode="json"), "error": "changed"})
    with pytest.raises(InvalidJobTransition):
        store.save(changed)
    store.save(cancelled)


def test_cancelled_and_failed_jobs_never_export_stale_result_artifacts(store: JobStore):
    cancelled = advance_to(store, JobStage.CANCELLED)
    (_test_job_directory(store, cancelled.id) / "result.json").write_text('{"stale": true}')
    assert store.read_export(cancelled.id) is None

    failed_store = JobStore(store.jobs_root.parent / "other-jobs", uuid_factory=lambda: UUID(int=2))
    failed = advance_to(failed_store, JobStage.FAILED)
    (_test_job_directory(failed_store, failed.id) / "result.json").write_text('{"stale": true}')
    assert failed_store.read_export(failed.id) is None


def test_public_save_enforces_graph_and_monotonic_progress_and_never_directly_completes(store: JobStore):
    queued = store.create(JobRequest(source="input.wav"))
    skipped = JobRecord.model_validate(
        {
            **queued.model_dump(mode="json"),
            "status": JobStage.ANALYSING,
            "progress": queued.progress.advance(stage=JobStage.ANALYSING, percent=40, message="skipped"),
        }
    )
    with pytest.raises(InvalidJobTransition):
        store.save(skipped)

    extracting = JobRecord.model_validate(
        {
            **queued.model_dump(mode="json"),
            "status": JobStage.EXTRACTING_AUDIO,
            "progress": queued.progress.advance(stage=JobStage.EXTRACTING_AUDIO, percent=40, message="extracting"),
        }
    )
    store.save(extracting)
    stale = JobRecord.model_validate(
        {
            **extracting.model_dump(mode="json"),
            "progress": Progress(stage=JobStage.EXTRACTING_AUDIO, percent=10, message="stale", revision=0),
        }
    )
    with pytest.raises(InvalidJobTransition):
        store.save(stale)
    complete = JobRecord(id=queued.id, status=JobStage.COMPLETE, ortb_content={"id": queued.id})
    with pytest.raises(InvalidJobTransition):
        store.save(complete)


def test_two_stores_cannot_overwrite_the_same_next_revision(store: JobStore):
    current = advance_to(store, JobStage.EXTRACTING_AUDIO)
    first = JobRecord.model_validate(
        {
            **current.model_dump(mode="json"),
            "progress": current.progress.advance(
                stage=JobStage.EXTRACTING_AUDIO, percent=40, message="first writer"
            ),
        }
    )
    second = JobRecord.model_validate(
        {
            **current.model_dump(mode="json"),
            "progress": current.progress.advance(
                stage=JobStage.EXTRACTING_AUDIO, percent=40, message="second writer"
            ),
        }
    )
    other = JobStore(store.jobs_root, uuid_factory=lambda: UUID(int=2))

    store.save(first)
    with pytest.raises(JobConflict):
        other.save(second)

    assert store.load(current.id) == first
    store.save(first)


def test_public_save_cannot_mutate_partial_arrays(store: JobStore):
    current = advance_to(store, JobStage.TRANSCRIBING)
    direct_partial = JobRecord.model_validate(
        {
            **current.model_dump(mode="json"),
            "transcript": (segment(),),
            "progress": current.progress.advance(
                stage=JobStage.TRANSCRIBING, percent=40, message="illicit partial"
            ),
        }
    )

    with pytest.raises(InvalidJobTransition):
        store.save(direct_partial)

    assert store.load(current.id).transcript == ()


def test_shorter_journal_than_snapshot_is_rejected_as_divergence(store: JobStore):
    current = advance_to(store, JobStage.TRANSCRIBING)
    store.append_transcript(current.id, segment())
    journal = _test_job_directory(store, current.id) / "transcript.jsonl"
    journal.write_text("")

    with pytest.raises(CorruptJobJournal):
        store.load(current.id)

    assert journal.exists()


def test_job_directory_swap_after_open_never_writes_through_a_symlink(store: JobStore, monkeypatch):
    current = advance_to(store, JobStage.EXTRACTING_AUDIO)
    original_dir = _test_job_directory(store, current.id)
    relocated_dir = store.jobs_root.parent / "relocated-job"
    outside = store.jobs_root.parent / "outside"
    outside.mkdir()
    update = JobRecord.model_validate(
        {
            **current.model_dump(mode="json"),
            "progress": current.progress.advance(
                stage=JobStage.EXTRACTING_AUDIO, percent=40, message="safe update"
            ),
        }
    )
    write_snapshot = store._write_snapshot
    swapped = False

    def swap_then_write(record, directory):
        nonlocal swapped
        if not swapped and record == update:
            original_dir.rename(relocated_dir)
            original_dir.symlink_to(outside, target_is_directory=True)
            swapped = True
        write_snapshot(record, directory)

    monkeypatch.setattr(store, "_write_snapshot", swap_then_write)
    store.save(update)

    assert swapped
    assert not (outside / "job.json").exists()
    assert JobRecord.model_validate_json((relocated_dir / "job.json").read_text()) == update


def test_read_export_keeps_result_consumption_anchored_after_directory_swap(store: JobStore, monkeypatch):
    exporting = advance_to(store, JobStage.EXPORTING)
    completed = store.promote_result(exporting.id, {"id": exporting.id})
    original_dir = _test_job_directory(store, completed.id)
    relocated_dir = store.jobs_root.parent / "relocated-complete-job"
    outside = store.jobs_root.parent / "outside"
    outside.mkdir()
    read_text = store._read_text
    swapped = False

    def swap_after_result_opened(directory, name):
        nonlocal swapped
        contents = read_text(directory, name)
        if name == "result.json" and not swapped:
            original_dir.rename(relocated_dir)
            original_dir.symlink_to(outside, target_is_directory=True)
            swapped = True
        return contents

    monkeypatch.setattr(store, "_read_text", swap_after_result_opened)

    assert store.read_export(completed.id) == b'{"id":"12345678-1234-5678-1234-567812345678"}'
    assert swapped
    assert not (outside / "result.json").exists()


def test_read_export_validates_and_returns_the_same_single_open_result_bytes(store: JobStore, monkeypatch):
    exporting = advance_to(store, JobStage.EXPORTING)
    completed = store.promote_result(exporting.id, {"id": exporting.id})
    result = _test_job_directory(store, completed.id) / "result.json"
    replacement = _test_job_directory(store, completed.id) / "replacement.json"
    replacement.write_text('{"id":"replacement"}')
    read_text = store._read_text
    result_reads = 0

    def replace_after_first_result_read(directory: int, name: str) -> str:
        nonlocal result_reads
        contents = read_text(directory, name)
        if name == "result.json":
            result_reads += 1
            if result_reads == 1:
                os.replace(replacement, result)
        return contents

    monkeypatch.setattr(store, "_read_text", replace_after_first_result_read)

    assert store.read_export(completed.id) == b'{"id":"12345678-1234-5678-1234-567812345678"}'
    assert result_reads == 1


def test_promote_result_is_the_only_complete_transition_and_complete_requires_result(store: JobStore):
    exporting = advance_to(store, JobStage.EXPORTING)
    completed = store.promote_result(exporting.id, {"id": exporting.id})

    assert completed.status is JobStage.COMPLETE
    assert completed.progress.stage is JobStage.COMPLETE
    assert store.read_export(completed.id) == b'{"id":"12345678-1234-5678-1234-567812345678"}'
    (_test_job_directory(store, completed.id) / "result.json").unlink()
    with pytest.raises(CorruptJobState):
        store.load(completed.id)


def test_store_export_bytes_match_shared_openrtb_serializer(store: JobStore):
    from engine.ortb.export import serialize_content

    exporting = advance_to(store, JobStage.EXPORTING)
    content = {"title": "Café", "context": 6, "language": "en"}
    store.promote_result(exporting.id, content)

    assert store.read_export(exporting.id) == serialize_content(content)


@pytest.mark.parametrize("stage", [JobStage.CANCELLED, JobStage.FAILED])
def test_cancelled_and_failed_jobs_cannot_promote_results(store: JobStore, stage: JobStage):
    job = advance_to(store, stage)

    with pytest.raises(InvalidJobTransition, match="only exporting"):
        store.promote_result(job.id, {"context": 6})

    assert store.read_export(job.id) is None
    assert not (_test_job_directory(store, job.id) / "result.json").exists()


def test_instrumental_no_vocals_failure_has_no_export(store: JobStore):
    exporting = advance_to(store, JobStage.EXPORTING)
    failed = JobRecord.model_validate(
        {
            **exporting.model_dump(mode="json"),
            "status": JobStage.FAILED,
            "error": "instrumental_no_vocals",
            "progress": exporting.progress.advance(
                stage=JobStage.FAILED, percent=exporting.progress.percent, message="instrumental_no_vocals"
            ),
        }
    )
    store.save(failed)

    with pytest.raises(InvalidJobTransition, match="only exporting"):
        store.promote_result(exporting.id, {"context": 3})

    assert store.read_export(exporting.id) is None
    assert not (_test_job_directory(store, exporting.id) / "result.json").exists()


def test_export_promotion_and_cancellation_race_has_one_durable_winner(store: JobStore):
    exporting = advance_to(store, JobStage.EXPORTING)
    cancelling = JobRecord.model_validate(
        {
            **exporting.model_dump(mode="json"),
            "status": JobStage.CANCELLING,
            "progress": exporting.progress.advance(
                stage=JobStage.CANCELLING, percent=exporting.progress.percent, message="cancelling"
            ),
        }
    )
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def cancel() -> None:
        barrier.wait()
        try:
            store.save(cancelling)
            outcomes.append("cancelled")
        except InvalidJobTransition:
            outcomes.append("cancel_rejected")

    def promote() -> None:
        barrier.wait()
        try:
            store.promote_result(exporting.id, {"context": 6})
            outcomes.append("promoted")
        except InvalidJobTransition:
            outcomes.append("promotion_rejected")

    threads = [threading.Thread(target=cancel), threading.Thread(target=promote)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(outcomes) == 2
    state = store.load(exporting.id)
    exported = store.read_export(exporting.id)
    if state.status is JobStage.COMPLETE:
        assert sorted(outcomes) == ["cancel_rejected", "promoted"]
        assert exported == b'{"context":6}'
    else:
        assert state.status is JobStage.CANCELLING
        assert sorted(outcomes) == ["cancelled", "promotion_rejected"]
        assert exported is None
        assert not (_test_job_directory(store, exporting.id) / "result.json").exists()


@pytest.mark.parametrize(
    "stage,expected_stage",
    [
        (JobStage.QUEUED, JobStage.QUEUED),
        (JobStage.EXTRACTING_AUDIO, JobStage.FAILED),
        (JobStage.DETECTING_LANGUAGE, JobStage.FAILED),
        (JobStage.TRANSCRIBING, JobStage.FAILED),
        (JobStage.ANALYSING, JobStage.FAILED),
        (JobStage.EXPORTING, JobStage.FAILED),
        (JobStage.CANCELLING, JobStage.FAILED),
        (JobStage.CANCELLED, JobStage.CANCELLED),
        (JobStage.FAILED, JobStage.FAILED),
    ],
)
def test_recovery_keeps_queued_and_terminal_jobs_but_fails_interrupted_jobs(store: JobStore, stage: JobStage, expected_stage: JobStage):
    job = advance_to(store, stage)
    journal = _test_job_directory(store, job.id) / "partial-artifact.txt"
    journal.write_text("preserve")

    recovered = store.recover()
    loaded = {record.id: record for record in recovered}[job.id]

    assert loaded.status is expected_stage
    assert journal.exists()
    if stage in _ORDER[1:] or stage is JobStage.CANCELLING:
        assert loaded.error == "interrupted_by_restart"
        assert loaded.progress.stage is JobStage.FAILED


def test_corrupt_snapshot_fails_explicitly_without_cleaning_artifacts(store: JobStore):
    job = store.create(JobRequest(source="input.wav"))
    snapshot = _test_job_directory(store, job.id) / "job.json"
    snapshot.write_text("not json")
    partial = _test_job_directory(store, job.id) / "timeline.jsonl"
    partial.write_text('{"partial": true}\n')

    with pytest.raises(ValidationError):
        store.recover()

    assert partial.exists()
