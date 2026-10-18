from __future__ import annotations

import json
import socket
import subprocess
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from engine.config import Settings
from engine.domain import ContentContext
from engine.pipeline import Pipeline
from engine.store import JobStore
from engine.web.app import create_app


class _SynchronousRunner:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.enqueued: list[str] = []

    async def enqueue(self, job_id: str) -> None:
        from engine.cancel import CancellationToken

        self.enqueued.append(job_id)
        self.pipeline.run(job_id, CancellationToken())

    async def cancel(self, job_id: str) -> None:
        del job_id


def _make_speech_video(path: Path) -> None:
    audio_path = path.with_suffix(".wav")
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=black:s=32x32:r=10:d=5", "-i", str(audio_path),
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", "-y", str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_offline_api_job_runs_real_audio_pipeline_and_exports_without_network(
    tmp_path: Path, monkeypatch,
) -> None:
    import shutil
    import engine.pipeline as pipeline_module
    from scripts.check_setup import enable_offline_mode
    from test_pipeline import _Index, _Models

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        import pytest

        pytest.skip("FFmpeg tools required")

    enable_offline_mode()
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline job attempted a network connection")
        ),
    )
    monkeypatch.setattr(
        socket.socket,
        "connect_ex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline job attempted a network connection")
        ),
    )

    media_root = tmp_path / "media"
    media_root.mkdir()
    source = media_root / "spoken.mp4"
    _make_speech_video(source)
    data_dir = tmp_path / "application-data"
    settings = Settings(data_dir=data_dir, allowed_roots=(media_root,))
    store = JobStore(settings.jobs_dir)
    pipeline = Pipeline(store, _Models(), _Index())
    prepared: list[tuple[Path, tuple[str, ...]]] = []
    original_prepare_audio = pipeline_module.prepare_audio

    def capture_prepared_audio(*args, **kwargs):
        media = original_prepare_audio(*args, **kwargs)
        workspace_files = tuple(path.name for path in media.audio_path.parent.iterdir())
        prepared.append((media.audio_path, workspace_files))
        return media

    monkeypatch.setattr(pipeline_module, "prepare_audio", capture_prepared_audio)
    runner = _SynchronousRunner(pipeline)
    client = TestClient(create_app(store=store, runner=runner, settings=settings))

    response = client.post("/api/jobs/path", json={"path": str(source), "requested_language": "en"})

    assert response.status_code == 202
    job_id = response.json()["id"]
    completed = store.load(job_id)
    assert completed.status.value == "complete"
    assert completed.resolved_content_context is ContentContext.VIDEO
    assert completed.text_timeline
    assert store.read_export(job_id) is not None
    assert runner.enqueued == [job_id]
    assert prepared and prepared[0][0].suffix == ".wav"
    assert not prepared[0][0].exists()
    assert not any(name.lower().endswith((".jpg", ".jpeg", ".png")) for name in prepared[0][1])
    exported = json.loads(store.read_export(job_id))
    assert exported
    snapshot = client.get(f"/api/jobs/{job_id}").json()
    assert snapshot["evidence_mode"] == "transcript_only"
    assert snapshot["verification_windows"][0] == {
        "time_range": {"start_seconds": 0.0, "end_seconds": 3.0},
        "verification_policy_version": "english-transcript-verifier-v2",
        "status": "supported", "error_code": None,
        "citations": [
            {"category_id": "633", "segment_indices": [0],
             "time_ranges": [{"start_seconds": 0.0, "end_seconds": 3.0}]},
            {"category_id": "255", "segment_indices": [0],
             "time_ranges": [{"start_seconds": 0.0, "end_seconds": 3.0}]},
        ],
    }
    assert client.get(f"/api/jobs/{job_id}/export").content == store.read_export(job_id)
    for url in ("/", "/static/app.js", "/static/styles.css", "/api/taxonomy"):
        assert client.get(url).status_code == 200

    # Exercise the delivered client against an actual API projection, not a JS proposal.
    from test_static_ui import _run_ui

    rendered = _run_ui(r"""
const snapshot = JSON.parse(process.argv[2]);
vm.runInContext('state.itemKind = "job"', context);
call('renderSnapshot', snapshot);
process.stdout.write(JSON.stringify({verification: document.getElementById('verification-content').textContent,
  summary: document.getElementById('summary-content').textContent,
  secondary: document.getElementById('secondary-content').textContent,
  copyHidden: document.getElementById('copy-button').hidden}));
""", snapshot)
    assert "Supported transcript evidence" in rendered["verification"]
    assert "Citations: 0:00–0:03" in rendered["verification"]
    assert "Category 633" in rendered["summary"]
    assert "Category 255" in rendered["secondary"]
    assert rendered["copyHidden"] is False
