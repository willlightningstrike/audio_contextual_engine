from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest

from engine.domain import JobRecord, JobRequest
from engine.lifecycle import JobStage
from engine.store import JobStore


JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs", uuid_factory=lambda: JOB_ID)


def _job_dir(store: JobStore) -> Path:
    return store.jobs_root / str(JOB_ID)


def _advance(store: JobStore, job, stage: JobStage):
    job = JobRecord.model_validate(
        {
            **job.model_dump(mode="json"),
            "status": stage,
            "progress": job.progress.advance(stage=stage, percent=25, message=stage.value),
        }
    )
    store.save(job)
    return job


def _transcribing_job(store: JobStore):
    job = store.create(JobRequest(source="input.wav"))
    for stage in (
        JobStage.EXTRACTING_AUDIO,
        JobStage.DETECTING_LANGUAGE,
        JobStage.TRANSCRIBING,
    ):
        job = _advance(store, job, stage)
    return job


def test_new_snapshot_and_journals_declare_schema_version(store: JobStore):
    job = store.create(JobRequest(source="input.wav"))
    snapshot = json.loads((_job_dir(store) / "job.json").read_text())

    assert snapshot["schema_version"] == 2
    assert snapshot["evidence_mode"] == "transcript_only"
    assert store.load(job.id).schema_version == 2


def test_old_snapshot_is_not_silently_reinterpreted(store: JobStore):
    from engine.store import UnsupportedJobSchemaVersion

    job = store.create(JobRequest(source="input.wav"))
    path = _job_dir(store) / "job.json"
    snapshot = json.loads(path.read_text())
    del snapshot["schema_version"]
    path.write_text(json.dumps(snapshot))

    with pytest.raises(UnsupportedJobSchemaVersion, match="missing"):
        store.load(job.id)


def test_schema_v2_without_evidence_marker_retains_legacy_projection(store):
    from engine.web.snapshots import job_snapshot
    job = store.create(JobRequest(source="input.wav"))
    path = _job_dir(store) / "job.json"
    snapshot = json.loads(path.read_text())
    snapshot.pop("evidence_mode", None)
    path.write_text(json.dumps(snapshot))
    loaded = store.load(job.id)
    assert loaded.evidence_mode is None
    assert job_snapshot(store, loaded)["evidence_mode"] is None


def test_unsupported_snapshot_version_fails_explicitly(store: JobStore):
    from engine.store import UnsupportedJobSchemaVersion

    job = store.create(JobRequest(source="input.wav"))
    path = _job_dir(store) / "job.json"
    snapshot = json.loads(path.read_text())
    snapshot["schema_version"] = 999
    path.write_text(json.dumps(snapshot))

    with pytest.raises(UnsupportedJobSchemaVersion, match="999"):
        store.load(job.id)


def test_journal_reconciliation_keeps_versioned_snapshot_and_partial_data(store: JobStore):
    from engine.domain import TimeRange, TranscriptSegment

    job = _transcribing_job(store)
    store.append_transcript(
        job.id,
        TranscriptSegment(
            time_range=TimeRange(start_seconds=0, end_seconds=1),
            text="hello",
            confidence=0.9,
            language="en",
        ),
    )
    loaded = store.load(job.id)

    assert loaded.schema_version == 2
    assert loaded.transcript[0].text == "hello"


def test_old_journal_record_is_not_silently_reinterpreted(store: JobStore):
    from engine.domain import TimeRange, TranscriptSegment

    job = _transcribing_job(store)
    store.append_transcript(
        job.id,
        TranscriptSegment(
            time_range=TimeRange(start_seconds=0, end_seconds=1),
            text="hello",
            confidence=0.9,
            language="en",
        ),
    )
    journal = _job_dir(store) / "transcript.jsonl"
    line = json.loads(journal.read_text())
    del line["schema_version"]
    journal.write_text(json.dumps(line) + "\n")

    with pytest.raises(ValueError, match="unsupported schema version"):
        store.load(job.id)


def test_schema_v2_legacy_result_without_citations_has_no_new_policy_support(store: JobStore):
    from engine.domain import TimeRange, TranscriptSegment
    from engine.web.snapshots import job_snapshot

    job = _transcribing_job(store)
    path = _job_dir(store) / "job.json"
    legacy = json.loads(path.read_text())
    legacy.pop("evidence_mode")
    path.write_text(json.dumps(legacy))
    store.append_transcript(job.id, TranscriptSegment(
        time_range=TimeRange(start_seconds=0, end_seconds=2),
        text="camera review", confidence=0.9, language="en",
    ))
    store.append_analysis_diagnostic(job.id, "result", {
        "time_range": {"start_seconds": 0, "end_seconds": 2},
        "payload": {"primary_category": "633", "secondary_categories": [],
                    "keywords": [], "entities": [],
                    "sentiment": {"label": "neutral", "score": 0}},
        "attempts": [], "error_code": None,
    })
    snapshot = job_snapshot(store, store.load(job.id))
    assert snapshot["schema_version"] == 2
    assert snapshot["topic_policy"] is None
    assert snapshot["topic_abstentions"] == []
    assert snapshot["evidence_mode"] is None
    assert snapshot["verification_windows"] == []
    assert snapshot["text_timeline"][0]["payload"]["primary_category"] == "633"
