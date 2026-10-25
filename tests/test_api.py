from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
import json
from pathlib import Path
import threading

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
import pytest

from engine.domain import JobRequest
from engine.store import JobStore
from engine.web.app import create_app


@dataclass
class Settings:
    data_dir: Path
    allowed_roots: tuple[Path, ...]
    max_upload_bytes: int = 4
    allowed_suffixes: frozenset[str] = frozenset({".wav", ".mp4"})
    upload_ttl_seconds: int = 86400
    sse_heartbeat_seconds: float = 15

    @property
    def uploads_root(self) -> Path:
        return self.data_dir / "uploads"


class Runner:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.cancelled: list[str] = []

    async def enqueue(self, item_id: str) -> None:
        self.enqueued.append(item_id)

    async def cancel(self, item_id: str) -> None:
        self.cancelled.append(item_id)


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from engine.web import routes

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    media = allowed / "speech.wav"
    media.write_bytes(b"data")
    settings = Settings(data_dir=tmp_path / "data", allowed_roots=(allowed,))
    store = JobStore(settings.data_dir / "jobs")
    runner = Runner()
    monkeypatch.setattr(routes, "probe_media", lambda path: _probe(path))
    return TestClient(create_app(store=store, runner=runner, settings=settings)), store, runner, settings, media


def _probe(path: Path):
    from engine.domain import AudioStream, ContainerKind, SourceMetadata
    from engine.ingest.probe import MediaProbe

    return MediaProbe(
        container_kind=ContainerKind.AUDIO,
        audio_streams=(AudioStream(index=0, codec_name="pcm_s16le"),),
        default_audio_stream=0,
        duration_seconds=3.0,
        source_metadata=SourceMetadata(title=path.name, duration_seconds=3.0),
    )


def test_path_job_is_durable_and_enqueued_before_accepted(api) -> None:
    client, store, runner, _, media = api
    response = client.post("/api/jobs/path", json={"path": str(media), "requested_language": "en"})
    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["evidence_mode"] == "transcript_only"
    assert client.get(f"/api/jobs/{job_id}").json()["evidence_mode"] == "transcript_only"
    assert store.load(job_id).request == JobRequest(
        source=media, requested_language="en", audio_stream_index=0
    )
    assert runner.enqueued == [job_id]


def test_upload_is_staged_before_a_job_is_created_and_claimed_once(api) -> None:
    client, store, runner, _, _ = api
    uploaded = client.post("/api/uploads?filename=voice.wav", content=b"1234")
    assert uploaded.status_code == 201
    upload_id = uploaded.json()["upload_id"]
    assert (api[3].uploads_root / upload_id / "voice.wav").read_bytes() == b"1234"
    assert runner.enqueued == []

    created = client.post(
        "/api/jobs/upload",
        json={"upload_id": upload_id, "audio_stream_index": 0, "requested_language": "en"},
    )
    assert created.status_code == 202
    job_id = created.json()["id"]
    assert store.load(job_id).request.source.name == "voice.wav"
    assert runner.enqueued == [job_id]
    reused = client.post("/api/jobs/upload", json={"upload_id": upload_id})
    assert reused.status_code == 409
    assert runner.enqueued == [job_id]


def test_upload_byte_limit_rejects_oversize_body_without_staging(api) -> None:
    client, _, _, settings, _ = api
    response = client.post("/api/uploads?filename=voice.wav", content=b"12345")
    assert response.status_code == 413
    assert not settings.uploads_root.exists() or list(settings.uploads_root.iterdir()) == []


def test_upload_stream_enforces_limit_without_content_length(api) -> None:
    client, _, _, settings, _ = api
    response = client.post(
        "/api/uploads?filename=voice.wav",
        content=iter((b"123", b"45")),
        headers={"Transfer-Encoding": "chunked"},
    )
    assert response.status_code == 413
    assert not settings.uploads_root.exists() or list(settings.uploads_root.iterdir()) == []


def test_expired_upload_cannot_be_claimed(api) -> None:
    client, _, _, settings, _ = api
    uploaded = client.post("/api/uploads?filename=voice.wav", content=b"1234")
    upload_id = uploaded.json()["upload_id"]
    upload_dir = settings.uploads_root / upload_id
    expired_at = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    os.utime(upload_dir, (expired_at, expired_at))
    response = client.post("/api/jobs/upload", json={"upload_id": upload_id})
    assert response.status_code == 404
    assert not upload_dir.exists()


def test_upload_cleanup_preserves_aged_claimed_source_for_queued_job(api) -> None:
    client, store, _, settings, _ = api
    uploaded = client.post("/api/uploads?filename=voice.wav", content=b"1234").json()
    created = client.post("/api/jobs/upload", json={"upload_id": uploaded["upload_id"]})
    assert created.status_code == 202
    job = store.load(created.json()["id"])
    unclaimed = client.post("/api/uploads?filename=unused.wav", content=b"1234").json()
    unclaimed_dir = settings.uploads_root / unclaimed["upload_id"]
    expired_at = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    for directory in (job.request.source.parent, unclaimed_dir):
        os.utime(directory, (expired_at, expired_at))

    assert client.post("/api/uploads?filename=next.wav", content=b"1234").status_code == 201

    assert store.load(job.id).status.value == "queued"
    assert job.request.source.read_bytes() == b"1234"
    assert (job.request.source.parent / ".claimed").is_file()
    assert not unclaimed_dir.exists()
    assert client.post("/api/jobs/upload", json={"upload_id": uploaded["upload_id"]}).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["/media/probe-path", "/jobs/path", "/uploads", "/jobs/upload"])
async def test_cancellation_is_serviced_while_request_probe_is_blocked(api, monkeypatch, endpoint):
    from engine.web import routes

    client, store, _, _, media = api
    job = store.create(JobRequest(source=media))
    if endpoint == "/uploads":
        url, kwargs = "/api/uploads?filename=voice.wav", {"content": b"1234"}
    elif endpoint == "/jobs/upload":
        uploaded = client.post("/api/uploads?filename=voice.wav", content=b"1234").json()
        url, kwargs = "/api/jobs/upload", {"json": {"upload_id": uploaded["upload_id"]}}
    else:
        url, kwargs = f"/api{endpoint}", {"json": {"path": str(media)}}
    started, release, returned = threading.Event(), threading.Event(), threading.Event()

    def blocked_probe(path):
        started.set()
        try:
            release.wait(timeout=2)
            return _probe(path)
        finally:
            returned.set()

    monkeypatch.setattr(routes, "probe_media", blocked_probe)
    async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://testserver") as http:
        pending = asyncio.create_task(http.post(url, **kwargs))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            cancelled = await asyncio.wait_for(http.post(f"/api/jobs/{job.id}/cancel"), timeout=1)
            assert cancelled.status_code == 202
            assert not returned.is_set(), "cancellation was blocked until the unrelated probe finished"
        finally:
            release.set()
            response = await pending
        assert response.status_code in {200, 201, 202}


@pytest.mark.asyncio
async def test_concurrent_upload_job_requests_keep_claim_single_use(api, monkeypatch):
    from engine.web import routes

    client, store, runner, _, _ = api
    uploaded = client.post("/api/uploads?filename=voice.wav", content=b"1234").json()
    both_started, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    calls = 0

    def competing_probe(path):
        nonlocal calls
        with lock:
            calls += 1
            if calls == 2:
                both_started.set()
        release.wait(timeout=2)
        return _probe(path)

    monkeypatch.setattr(routes, "probe_media", competing_probe)
    async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://testserver") as http:
        pending = [asyncio.create_task(http.post("/api/jobs/upload", json={"upload_id": uploaded["upload_id"]}))
                   for _ in range(2)]
        try:
            assert await asyncio.to_thread(both_started.wait, 5)
        finally:
            release.set()
            responses = await asyncio.gather(*pending)
    assert sorted(response.status_code for response in responses) == [202, 409]
    created = next(response.json() for response in responses if response.status_code == 202)
    assert runner.enqueued == [created["id"]]
    assert [job.id for job in store.recover()] == [created["id"]]


def test_invalid_upload_stream_does_not_consume_upload_claim(api) -> None:
    client, _, runner, _, _ = api
    uploaded = client.post("/api/uploads?filename=voice.wav", content=b"1234")
    upload_id = uploaded.json()["upload_id"]
    rejected = client.post(
        "/api/jobs/upload", json={"upload_id": upload_id, "audio_stream_index": 99}
    )
    assert rejected.status_code == 422
    accepted = client.post("/api/jobs/upload", json={"upload_id": upload_id, "audio_stream_index": 0})
    assert accepted.status_code == 202
    assert runner.enqueued == [accepted.json()["id"]]


def test_path_escape_and_symlink_escape_are_rejected(api, tmp_path: Path) -> None:
    client, _, runner, _, media = api
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x")
    link = media.parent / "linked.wav"
    link.symlink_to(outside)
    for path in (str(media.parent / ".." / "outside.wav"), str(link), str(outside)):
        response = client.post("/api/media/probe-path", json={"path": path})
        assert response.status_code == 403
    assert runner.enqueued == []


def test_invalid_stream_index_is_rejected_before_job_creation(api) -> None:
    client, _, runner, _, media = api
    response = client.post(
        "/api/jobs/path", json={"path": str(media), "audio_stream_index": 99}
    )
    assert response.status_code == 422
    assert runner.enqueued == []


def test_non_english_override_is_not_supported_and_english_is_only_a_hint(api) -> None:
    client, store, runner, _, media = api
    rejected = client.post("/api/jobs/path", json={"path": str(media), "requested_language": "fr"})
    assert rejected.status_code == 422
    accepted = client.post("/api/jobs/path", json={"path": str(media), "requested_language": "en"})
    assert accepted.status_code == 202
    assert store.load(accepted.json()["id"]).request.requested_language == "en"
    assert runner.enqueued == [accepted.json()["id"]]


def test_export_requires_completion_and_partial_snapshot_keeps_diagnostics(api) -> None:
    from engine.lifecycle import JobStage

    client, store, _, _, media = api
    response = client.post("/api/jobs/path", json={"path": str(media)})
    job_id = response.json()["id"]
    partial = store.load(job_id)
    progress = partial.progress.advance(stage=JobStage.EXTRACTING_AUDIO, percent=20, message="extracting")
    store.save(partial.model_copy(update={"status": JobStage.EXTRACTING_AUDIO, "progress": progress, "error": "diagnostic"}))
    snapshot = client.get(f"/api/jobs/{job_id}")
    assert snapshot.status_code == 200
    assert snapshot.json()["error"] == "diagnostic"
    assert client.get(f"/api/jobs/{job_id}/export").status_code == 409


def test_snapshot_exposes_saved_transcription_window_diagnostics(api) -> None:
    from engine.transcribe.types import WindowResult

    client, store, _, _, media = api
    job_id = client.post("/api/jobs/path", json={"path": str(media)}).json()["id"]
    store.append_transcription_window(
        job_id,
        WindowResult(
            time_range={"start_seconds": 0, "end_seconds": 3},
            error_code="VOCAL_PRESENCE_UNCERTAIN",
            failure_stage="vocal_screening",
            failure_reason="vocal_presence_uncertain",
        ),
    )
    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    window = response.json()["transcription_windows"][0]
    assert window["error_code"] == "VOCAL_PRESENCE_UNCERTAIN"
    assert window["time_range"] == {"start_seconds": 0.0, "end_seconds": 3.0}


def test_snapshot_keeps_successful_saved_classification_when_final_timeline_is_missing(api) -> None:
    from engine.domain import TimeRange, TranscriptSegment

    client, store, _, _, media = api
    job_id = client.post("/api/jobs/path", json={"path": str(media)}).json()["id"]
    path = store.jobs_root / job_id / "job.json"
    legacy = json.loads(path.read_text())
    legacy.pop("evidence_mode")
    path.write_text(json.dumps(legacy))
    store.append_transcript(
        job_id,
        TranscriptSegment(
            time_range=TimeRange(start_seconds=0, end_seconds=3),
            text="Camera exposure controls",
            confidence=0.9,
            language="en",
        ),
    )
    store.append_analysis_diagnostic(
        job_id,
        "result",
        {
            "time_range": {"start_seconds": 0, "end_seconds": 3},
            "payload": {
                "primary_category": "633",
                "secondary_categories": [],
                "keywords": [],
                "entities": [],
                "sentiment": {"label": "neutral", "score": 0.0},
            },
            "attempts": [
                {
                    "attempt": 1,
                    "stage": "classification",
                    "prompt_sha256": "0" * 64,
                    "finish_reason": "stop",
                    "seconds": 0.1,
                    "token_count": 5,
                }
            ],
            "error_code": None,
        },
    )

    response = client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200
    timeline = response.json()["text_timeline"]
    assert len(timeline) == 1
    assert timeline[0]["payload"]["primary_category"] == "633"
    assert timeline[0]["evidence_ranges"] == [
        {"start_seconds": 0.0, "end_seconds": 3.0}
    ]


def test_versioned_snapshot_uses_only_citations_and_separates_abstention(api) -> None:
    from engine.domain import TimeRange, TranscriptSegment
    from engine.lifecycle import JobStage

    client, store, _, _, media = api
    job_id = client.post("/api/jobs/path", json={"path": str(media)}).json()["id"]
    for start, end, words in ((0, 1, "camera review"),
                              (1, 2, "camera lens"), (2, 3, "brief aside")):
        store.append_transcript(job_id, TranscriptSegment(
            time_range=TimeRange(start_seconds=start, end_seconds=end),
            text=words, confidence=0.9, language="en",
        ))
    store.append_analysis_diagnostic(job_id, "result", {
        "time_range": {"start_seconds": 0, "end_seconds": 3},
        "payload": {"primary_category": "633", "secondary_categories": [],
                    "keywords": [], "entities": [],
                    "sentiment": {"label": "neutral", "score": 0}},
        "attempts": [], "error_code": None,
        "category_citations": [{"category_id": "633", "segment_indices": [0, 1]}],
        "policy_version": "english-audio-topic-sufficiency-v1",
        "verification_policy_version": "english-transcript-verifier-v1",
    })
    store.append_analysis_diagnostic(job_id, "result", {
        "time_range": {"start_seconds": 3, "end_seconds": 4},
        "payload": None, "attempts": [],
        "error_code": "INSUFFICIENT_TOPIC_EVIDENCE",
        "category_citations": [],
        "policy_version": "english-audio-topic-sufficiency-v1",
        "verification_policy_version": "english-transcript-verifier-v1",
    })
    store.append_analysis_diagnostic(job_id, "topic_policy", {
        "policy_version": "english-audio-aggregation-v2",
        "outcome": "insufficient", "primary_category": None,
        "transcript_seconds": 3.0, "audio_seconds": 30.0,
        "primary_role_support_seconds": [["633", 2.0]],
        "secondary_role_support_seconds": [],
    })
    job = store.load(job_id)
    failed = job.progress.advance(stage=JobStage.FAILED, percent=job.progress.percent, message="insufficient")
    store.save(job.model_copy(update={"status": JobStage.FAILED, "progress": failed,
                                      "error": "INSUFFICIENT_TOPIC_EVIDENCE"}))

    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot["status"] == "failed"
    assert snapshot["error"] == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert snapshot["ortb_content"] is None
    assert [segment["text"] for segment in snapshot["transcript"]] == [
        "camera review", "camera lens", "brief aside"
    ]
    assert snapshot["topic_policy"]["outcome"] == "insufficient"
    assert snapshot["text_timeline"][0]["evidence_ranges"] == [
        {"start_seconds": 0.0, "end_seconds": 2.0}
    ]
    abstained = [{"time_range": {"start_seconds": 3, "end_seconds": 4},
                  "error_code": "INSUFFICIENT_TOPIC_EVIDENCE"}]
    assert snapshot["topic_abstentions"] == abstained
    assert snapshot["coverage"]["abstained_ranges"] == abstained
    assert snapshot["coverage"]["failed_ranges"] == []
    assert [item["status"] for item in snapshot["verification_windows"]] == ["supported", "abstained"]
    assert snapshot["verification_windows"][0]["citations"][0]["time_ranges"] == [
        {"start_seconds": 0, "end_seconds": 1}, {"start_seconds": 1, "end_seconds": 2}]
    assert client.get(f"/api/jobs/{job_id}/export").status_code == 409


def test_new_transcript_only_job_never_projects_unverified_result(api):
    client, store, _, _, media = api
    job_id = client.post("/api/jobs/path", json={"path": str(media)}).json()["id"]
    store.append_analysis_diagnostic(job_id, "result", {
        "time_range": {"start_seconds": 0, "end_seconds": 2},
        "payload": {"primary_category": "633", "secondary_categories": [],
                    "keywords": [], "entities": [],
                    "sentiment": {"label": "neutral", "score": 0}},
        "attempts": [], "error_code": None,
    })
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot["text_timeline"] == []
    assert snapshot["verification_windows"] == []


def test_export_bytes_are_identical_to_store_export_and_cancel_is_idempotent(api) -> None:
    from engine.lifecycle import JobStage

    client, store, runner, _, media = api
    created = client.post("/api/jobs/path", json={"path": str(media)})
    job_id = created.json()["id"]
    record = store.load(job_id)
    for stage, percent in (
        (JobStage.EXTRACTING_AUDIO, 10),
        (JobStage.DETECTING_LANGUAGE, 20),
        (JobStage.TRANSCRIBING, 40),
        (JobStage.ANALYSING, 80),
        (JobStage.EXPORTING, 90),
    ):
        progress = record.progress.advance(stage=stage, percent=percent, message=stage.value)
        record = record.model_copy(update={"status": stage, "progress": progress})
        store.save(record)
    result = {"id": "content", "language": "en", "context": 7}
    store.promote_result(job_id, result)
    exported = client.get(f"/api/jobs/{job_id}/export")
    assert exported.status_code == 200
    assert exported.content == store.read_export(job_id)
    assert exported.headers["content-type"].startswith("application/json")

    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 202
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 202
    assert runner.cancelled == [job_id, job_id]


def test_mutation_rejects_unexpected_origin(api) -> None:
    client, _, runner, _, _ = api
    response = client.post(
        "/api/uploads?filename=voice.wav", content=b"x", headers={"Origin": "https://evil.test"}
    )
    assert response.status_code == 403
    assert runner.enqueued == []


def test_batch_parent_has_public_snapshot_and_idempotent_cancel(api) -> None:
    client, store, runner, _, media = api
    parent = client.post("/api/batches/path", json={"path": str(media.parent)})
    assert parent.status_code == 202
    batch_id = parent.json()["id"]
    assert client.get(f"/api/batches/{batch_id}").status_code == 200
    assert store.load_batch(batch_id).child_job_ids
    assert runner.enqueued == [batch_id]
    assert client.post(f"/api/batches/{batch_id}/cancel").status_code == 202
    assert client.post(f"/api/batches/{batch_id}/cancel").status_code == 202
    assert runner.cancelled == [batch_id, batch_id]


def test_taxonomy_route_returns_names_and_paths_for_rendering_category_ids(api) -> None:
    client, _, _, _, _ = api
    response = client.get("/api/taxonomy")
    assert response.status_code == 200
    row = next(item for item in response.json() if item["id"] == "633")
    assert row == {
        "id": "633",
        "name": "Cameras and Camcorders",
        "path": ["Technology & Computing", "Consumer Electronics", "Cameras and Camcorders"],
    }


def test_dot_prefixed_upload_can_start_a_job(api) -> None:
    # Regression: staging accepted ".speech.wav", but lookup hid every
    # dot-prefixed name, so the accepted upload could never start.
    client, store, runner, _, _ = api
    uploaded = client.post("/api/uploads?filename=.speech.wav", content=b"1234")
    assert uploaded.status_code == 201

    created = client.post("/api/jobs/upload", json={"upload_id": uploaded.json()["upload_id"]})

    assert created.status_code == 202
    assert store.load(created.json()["id"]).request.source.name == ".speech.wav"
    assert runner.enqueued == [created.json()["id"]]
