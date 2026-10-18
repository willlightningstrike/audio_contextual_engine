import json
import hashlib
import pytest

from scripts.verify_topic_support import load_saved_case, run_probe


@pytest.fixture(autouse=True)
def model_metadata(tmp_path, monkeypatch):
    from scripts import verify_topic_support
    path = tmp_path / "test-model-manifest.json"
    path.write_text(json.dumps({"model": {
        "repository": "test-only/fake-generator",
        "revision": "test-only-revision",
    }}), encoding="utf-8")
    monkeypatch.setattr(verify_topic_support, "TEXT_MANIFEST", path)
    return path


@pytest.fixture
def saved(tmp_path):
    interval = {"start_seconds": 0.0, "end_seconds": 4.0}
    unit = {"time_range": interval, "segments": [{"time_range": interval,
            "text": "Camera settings", "confidence": 0.9, "language": "en"}]}
    (tmp_path / "transcription-windows.jsonl").write_text(json.dumps({"unit": unit}) + "\n")
    candidates = [{"taxonomy_id": id, "name": "Camera", "path": ["Technology", id], "similarity": 0.8}
                  for id in ("633", "514")]
    result = {"time_range": interval, "payload": None, "error_code": "INSUFFICIENT_TOPIC_EVIDENCE",
              "attempts": [{"attempt": 1, "prompt_sha256": "a" * 64}]}
    records = [{"kind": "candidates", "payload": {"time_range": interval, "candidates": candidates}},
               {"kind": "result", "payload": result}]
    (tmp_path / "analysis-diagnostics.jsonl").write_text("\n".join(map(json.dumps, records)))
    return tmp_path


def test_load_preserves_order_and_status(saved):
    run, results, candidates = load_saved_case(saved)
    assert run.status == "ready"
    assert results[0].error_code == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert [c.taxonomy_id for c in candidates[0].candidates] == ["633", "514"]


@pytest.mark.parametrize("keep", [0, 1])
def test_missing_records_rejected(saved, keep):
    path = saved / "analysis-diagnostics.jsonl"
    path.write_text(path.read_text().splitlines()[keep])
    with pytest.raises(ValueError):
        load_saved_case(saved)


def test_duplicate_record_rejected(saved):
    path = saved / "analysis-diagnostics.jsonl"
    path.write_text(path.read_text() + "\n" + path.read_text().splitlines()[0])
    with pytest.raises(ValueError):
        load_saved_case(saved)


def test_partial_asr_is_preserved(saved):
    path = saved / "transcription-windows.jsonl"
    failed = {"unit": {"time_range": {"start_seconds": 4, "end_seconds": 8},
                       "error_code": "ASR_FAILED"}}
    path.write_text(path.read_text() + json.dumps(failed) + "\n")
    run, _, _ = load_saved_case(saved)
    assert run.status == "partial"
    assert run.windows[-1].error_code == "ASR_FAILED"


def test_saved_success_deserializes_strict_json_arrays(saved):
    path = saved / "analysis-diagnostics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["payload"].update(error_code=None, payload={"primary_category": "633",
        "secondary_categories": [], "keywords": [], "entities": [],
        "sentiment": {"label": "neutral", "score": 0.0}})
    path.write_text("\n".join(map(json.dumps, records)))
    _, results, _ = load_saved_case(saved)
    assert results[0].payload.secondary_categories == ()


@pytest.mark.parametrize("sidecar", [False, True])
def test_existing_evidence_prevents_factory(tmp_path, sidecar):
    out = tmp_path / "report.json"
    target = out.with_suffix(".attempts.jsonl") if sidecar else out
    target.write_text("historical evidence")
    with pytest.raises(FileExistsError):
        run_probe(cases=(), out=out, generator_factory=lambda: pytest.fail("loaded"))
    assert target.read_text() == "historical evidence"


def test_rows_and_durable_attempts(saved):
    from engine.signals.text_types import GenerationResult
    class Generator:
        def generate_json(self, prompt, schema, **kwargs):
            # The checkpoint must already exist before any model response parsing.
            assert (saved / "report.attempts.jsonl").exists()
            return GenerationResult(raw=json.dumps({"status": "supported", "category_id":
                schema["oneOf"][0]["properties"]["category_id"]["const"], "segment_indices": [0]}),
                finish_reason="stop", seconds=0.01, token_count=10)
    report = run_probe(cases=[("control", saved)], out=saved / "report.json", generator_factory=Generator)
    row = report["cases"][0]
    assert {"case", "window", "proposed_ids", "verified_ids", "attempts", "error", "non_blind"} <= row.keys()
    assert row["non_blind"] is True
    assert row["verified_ids"] == ["633"]
    attempt = json.loads((saved / "report.attempts.jsonl").read_text())
    assert attempt["attempt"]["raw"] == row["attempts"][0]["raw"]
    assert len(attempt["attempt"]["prompt_sha256"]) == 64


def test_factory_failure_records_every_case(saved):
    def fail():
        raise RuntimeError("preflight failed")
    report = run_probe(cases=[("one", saved), ("two", saved)], out=saved / "failure.json", generator_factory=fail)
    assert [row["case"] for row in report["cases"]] == ["one", "two"]
    assert all(row["error"] == "MODEL_PREFLIGHT_ERROR" and row["verified_ids"] == [] for row in report["cases"])


def test_missing_model_metadata_records_preflight_without_calling_factory(saved, monkeypatch):
    from scripts import verify_topic_support as probe
    missing_manifest = saved / "missing-model-manifest.json"
    monkeypatch.setattr(probe, "TEXT_MANIFEST", missing_manifest)
    calls = []
    def factory():
        calls.append("called")
        raise RuntimeError("factory must not run without metadata")
    out = saved / "missing-metadata.json"
    report = run_probe(cases=[("control", saved)], out=out, generator_factory=factory)
    assert json.loads(out.read_text()) == report
    assert report["model"] is None
    assert report["preflight_error"].startswith("FileNotFoundError:")
    assert str(missing_manifest) in report["preflight_error"]
    assert report["cases"][0]["error"] == "MODEL_PREFLIGHT_ERROR"
    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["verified_ids"] == []
    assert report["cases"][0]["attempts"] == []
    assert out.with_suffix(".attempts.jsonl").read_text() == ""
    assert calls == []


@pytest.mark.parametrize("content,error_type", [
    pytest.param('{"model":', "JSONDecodeError", id="malformed-json"),
    pytest.param('{}', "KeyError", id="missing-model"),
    pytest.param('{"model":{"revision":"test-only-revision"}}', "KeyError", id="missing-repository"),
    pytest.param('{"model":{"repository":"test-only/fake-generator"}}', "KeyError", id="missing-revision"),
])
def test_invalid_model_metadata_records_preflight_without_calling_factory(saved, monkeypatch, content, error_type):
    from scripts import verify_topic_support as probe
    manifest = saved / "invalid-model-manifest.json"
    manifest.write_text(content, encoding="utf-8")
    monkeypatch.setattr(probe, "TEXT_MANIFEST", manifest)
    calls = []
    def factory():
        calls.append("called")
        raise RuntimeError("factory must not run with invalid metadata")
    out = saved / "invalid-metadata.json"
    report = run_probe(cases=[("one", saved), ("two", saved)], out=out, generator_factory=factory)
    assert json.loads(out.read_text()) == report
    assert report["model"] is None
    assert report["preflight_error"].startswith(f"{error_type}:")
    assert [row["case"] for row in report["cases"]] == ["one", "two"]
    for row in report["cases"]:
        assert row["error"] == "MODEL_PREFLIGHT_ERROR"
        assert row["status"] == "failed"
        assert row["verified_ids"] == []
        assert row["attempts"] == []
    assert out.with_suffix(".attempts.jsonl").read_text() == ""
    assert calls == []


def test_probe_hashes_consumed_snapshots_when_factory_mutates_sources(saved):
    from engine.signals.text_types import GenerationResult
    filenames = ("transcription-windows.jsonl", "analysis-diagnostics.jsonl")
    original = {name: (saved / name).read_bytes() for name in filenames}
    class Generator:
        def generate_json(self, prompt, schema, **kwargs):
            assert "Camera settings" in prompt
            assert "Replacement transcript" not in prompt
            assert schema["oneOf"][0]["properties"]["category_id"]["const"] == "633"
            return GenerationResult(raw='{"status":"supported","category_id":"633","segment_indices":[0]}',
                                    finish_reason="stop", seconds=0.01, token_count=10)
    def factory():
        (saved / filenames[0]).write_text(original[filenames[0]].decode().replace("Camera settings", "Replacement transcript"))
        (saved / filenames[1]).write_text(original[filenames[1]].decode().replace("633", "255"))
        return Generator()
    report = run_probe(cases=[("control", saved)], out=saved / "snapshot.json", generator_factory=factory)
    assert report["cases"][0]["verified_ids"] == ["633"]
    assert report["inputs"][0]["source_sha256"] == {
        name: hashlib.sha256(content).hexdigest() for name, content in original.items()}


def test_factory_failure_preserves_captured_model_identity(saved, monkeypatch):
    from scripts import verify_topic_support as probe
    manifest = saved / "manifest.json"
    content = b'{"model":{"repository":"test/repo","revision":"fixed-revision"}}'
    manifest.write_bytes(content)
    monkeypatch.setattr(probe, "TEXT_MANIFEST", manifest)
    def fail():
        manifest.write_text('{"model":{"repository":"replacement/repo","revision":"replacement"}}')
        raise RuntimeError("factory failed")
    report = run_probe(cases=[("control", saved)], out=saved / "identity.json", generator_factory=fail)
    assert report["model"] == {"manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "repository": "test/repo", "revision": "fixed-revision"}
    assert report["preflight_error"] == "RuntimeError: factory failed"
    assert report["cases"][0]["error"] == "MODEL_PREFLIGHT_ERROR"
    assert report["cases"][0]["verified_ids"] == []


@pytest.mark.parametrize("before_attempt", [True, False])
def test_cancellation_finalizes_all_declared_windows_without_more_inference(saved, before_attempt):
    from engine.cancel import JobCancelled
    from engine.signals.text_types import GenerationResult
    for filename in ("transcription-windows.jsonl", "analysis-diagnostics.jsonl"):
        path = saved / filename
        records = [json.loads(line) for line in path.read_text().splitlines()]
        copies = json.loads(json.dumps(records))
        for record in copies:
            payload = record.get("unit", record.get("payload"))
            payload["time_range"] = {"start_seconds": 100.0, "end_seconds": 104.0}
            for segment in payload.get("segments", []):
                segment["time_range"] = payload["time_range"]
        path.write_text("\n".join(map(json.dumps, records + copies)))
    class Generator:
        calls = 0
        def generate_json(self, prompt, schema, *, token, **kwargs):
            self.calls += 1
            if before_attempt:
                raise JobCancelled()
            token.cancel()
            return GenerationResult(raw=json.dumps({"status": "supported", "category_id":
                schema["oneOf"][0]["properties"]["category_id"]["const"], "segment_indices": [0]}),
                finish_reason="stop", seconds=0.01, token_count=10)
    generator = Generator()
    out = saved / "cancelled.json"
    report = run_probe(cases=[("one", saved), ("two", saved)], out=out,
                       generator_factory=lambda: generator)
    assert json.loads(out.read_text()) == report
    rows = report["cases"]
    assert len(rows) == 4
    assert [row["case"] for row in rows] == ["one", "one", "two", "two"]
    assert rows[0]["error"] == "JOB_CANCELLED"
    assert rows[0]["status"] == "cancelled"
    assert all(row["status"] == "not_attempted" and row["error"] == "NOT_ATTEMPTED_AFTER_CANCELLATION"
               for row in rows[1:])
    assert all(row["verified_ids"] == [] for row in rows)
    assert generator.calls == 1
    journal = out.with_suffix(".attempts.jsonl").read_text().splitlines()
    assert len(rows[0]["attempts"]) == len(journal) == (0 if before_attempt else 1)
    if journal:
        assert rows[0]["attempts"][0] == json.loads(journal[0])["attempt"]
