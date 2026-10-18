from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from test_api import Runner, Settings, _probe
from engine.store import JobStore
from engine.web import routes
from engine.web.app import create_app
from engine.web.sse import job_events


@pytest.fixture
def sse_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    media = allowed / "one.wav"
    media.write_bytes(b"audio")
    settings = Settings(
        data_dir=tmp_path / "data",
        allowed_roots=(allowed,),
        sse_heartbeat_seconds=0.02,
    )
    store = JobStore(settings.data_dir / "jobs")
    runner = Runner()
    monkeypatch.setattr(routes, "probe_media", lambda path: _probe(path))
    return create_app(store=store, runner=runner, settings=settings), store, runner, media


@pytest.mark.asyncio
async def test_stream_starts_with_durable_snapshot_and_reconnects_at_latest_revision(sse_app) -> None:
    _, store, _, media = sse_app
    created = store.create(__import__("engine.domain", fromlist=["JobRequest"]).JobRequest(source=media))
    older = created.progress.advance(stage=created.status, percent=0, message="older")
    store.save(created.model_copy(update={"progress": older}))
    latest = store.load(created.id)
    newest = latest.progress.advance(stage=latest.status, percent=0, message="newest")
    store.save(latest.model_copy(update={"progress": newest}))

    stream = job_events(store, created.id, heartbeat_seconds=0.02)
    frame = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert frame.startswith(f"id: {newest.revision}\n")
    assert '"message":"newest"' in frame
    assert '"evidence_mode":"transcript_only"' in frame
    assert '"message":"older"' not in frame
    current = store.load(created.id)
    advanced = current.progress.advance(stage=current.status, percent=0, message="advanced")
    store.save(current.model_copy(update={"progress": advanced}))
    next_frame = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert next_frame.startswith(f"id: {advanced.revision}\n")
    assert advanced.revision > newest.revision
    assert '"message":"advanced"' in next_frame
    await stream.aclose()


@pytest.mark.asyncio
async def test_sse_emits_comment_heartbeat_without_blocking_cancel_or_worker(sse_app) -> None:
    _, store, runner, media = sse_app
    created = store.create(__import__("engine.domain", fromlist=["JobRequest"]).JobRequest(source=media))
    from engine.lifecycle import JobStage

    progress = created.progress.advance(stage=JobStage.EXTRACTING_AUDIO, percent=20, message="working")
    store.save(created.model_copy(update={"status": JobStage.EXTRACTING_AUDIO, "progress": progress}))
    stream = job_events(store, created.id, heartbeat_seconds=0.02)
    initial = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert initial.startswith("id: ")
    heartbeat = await asyncio.wait_for(stream.__anext__(), timeout=1)
    assert heartbeat == ": heartbeat\n\n"
    await runner.cancel(created.id)
    assert runner.cancelled == [created.id]
    await stream.aclose()
    assert store.load(created.id).status is JobStage.EXTRACTING_AUDIO


@pytest.mark.asyncio
async def test_terminal_stream_closes_and_snapshot_remains_polling_fallback(sse_app) -> None:
    app, store, _, media = sse_app
    created = store.create(__import__("engine.domain", fromlist=["JobRequest"]).JobRequest(source=media))
    from engine.lifecycle import JobStage

    progress = created.progress.advance(stage=JobStage.CANCELLED, percent=0, message="cancelled")
    store.save(created.model_copy(update={"status": JobStage.CANCELLED, "progress": progress}))
    client = TestClient(app)
    response = client.get(
        f"/api/jobs/{created.id}/events", headers={"Last-Event-ID": "0"}
    )
    snapshot = client.get(f"/api/jobs/{created.id}")
    blocks = response.text.strip().split("\n\n")
    assert blocks[0].startswith(f"id: {progress.revision}\ndata: ")
    assert blocks[1] == "event: terminal\ndata: {}"
    assert snapshot.json()["progress"]["revision"] == progress.revision


@pytest.mark.asyncio
async def test_reconnected_sse_matches_failed_topic_snapshot(sse_app) -> None:
    from engine.domain import JobRequest
    from engine.lifecycle import JobStage

    app, store, _, media = sse_app
    created = store.create(JobRequest(source=media))
    store.append_analysis_diagnostic(created.id, "result", {
        "time_range": {"start_seconds": 0, "end_seconds": 2},
        "payload": None, "attempts": [],
        "error_code": "INSUFFICIENT_TOPIC_EVIDENCE",
        "policy_version": "english-audio-topic-sufficiency-v1",
        "verification_policy_version": "english-transcript-verifier-v1",
        "category_citations": [],
    })
    updated = store.load(created.id)
    progress = updated.progress.advance(stage=JobStage.FAILED, percent=0, message="insufficient")
    store.save(updated.model_copy(update={"status": JobStage.FAILED,
                                  "progress": progress,
                                  "error": "INSUFFICIENT_TOPIC_EVIDENCE"}))
    client = TestClient(app)
    polled = client.get(f"/api/jobs/{created.id}").json()
    stream = job_events(store, created.id, heartbeat_seconds=0.02)
    frame = await asyncio.wait_for(stream.__anext__(), timeout=1)
    emitted = json.loads(frame.split("data: ", 1)[1])
    assert emitted == polled
    assert emitted["topic_abstentions"][0]["error_code"] == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert emitted["verification_windows"][0]["status"] == "abstained"
    assert client.get(f"/api/jobs/{created.id}/export").status_code == 409
    await stream.aclose()
