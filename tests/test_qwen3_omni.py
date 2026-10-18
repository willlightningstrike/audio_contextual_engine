from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import wave

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange


def _load_script(name: str):
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True)
    return path.resolve()


def _wav(tmp_path: Path, seconds: float = 1.0) -> Path:
    path = tmp_path / "window.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path.resolve()


def _inventory_record(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _candidate_snapshot(tmp_path: Path) -> Path:
    snapshot = _snapshot(tmp_path, "qwen3_omni")
    for name, payload in {
        "config.json": b"{}",
        "processor_config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors.index.json": b"{}",
        "model-00001-of-00001.safetensors": b"weights",
    }.items():
        (snapshot / name).write_bytes(payload)
    return snapshot


def _retrieval_candidates() -> list[dict[str, object]]:
    ids = [
        "150",
        "151",
        "179",
        "181",
        "153",
        "154",
        "155",
        "158",
        "159",
        "160",
        "177",
        "178",
        "1",
        "2",
        "3",
    ]
    return [
        {
            "taxonomy_id": taxonomy_id,
            "name": f"candidate-{taxonomy_id}",
            "path": [f"candidate-{taxonomy_id}"],
            "similarity": 1.0 - index / 100,
            "provenance": ["transcript"],
        }
        for index, taxonomy_id in enumerate(ids)
    ]


def _baseline_manifest(tmp_path: Path) -> Path:
    from engine.models.registry import MODEL_SPECS

    models: dict[str, dict[str, object]] = {}
    for role, spec in MODEL_SPECS.items():
        snapshot = _snapshot(tmp_path, role)
        model_file = snapshot / "model.safetensors"
        model_file.write_bytes(f"{role}-weights".encode())
        models[role] = {
            **spec,
            "local_path": str(snapshot),
            "files": [_inventory_record(model_file, snapshot)],
        }
    path = tmp_path / "model-manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "models": models}))
    return path


def test_qwen3_omni_uses_one_resident_model_and_native_text_audio_streams(
    tmp_path: Path,
):
    from engine.models.qwen3_omni import Qwen3OmniAdapter

    snapshot = _snapshot(tmp_path, "candidate")
    wav = _wav(tmp_path)
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_omni_moe"))
    processor = object()
    load_calls: list[Path] = []
    template_calls: list[dict[str, object]] = []
    stream_calls: list[dict[str, object]] = []

    def load(path: Path):
        load_calls.append(path)
        return model, processor

    def template(got_processor, config, prompt, **kwargs):
        template_calls.append(
            {
                "processor": got_processor,
                "config": config,
                "prompt": prompt,
                **kwargs,
            }
        )
        return f"formatted:{prompt}"

    def stream(got_model, got_processor, prompt, **kwargs):
        stream_calls.append(
            {
                "model": got_model,
                "processor": got_processor,
                "prompt": prompt,
                **kwargs,
            }
        )
        yield SimpleNamespace(text="answer")

    adapter = Qwen3OmniAdapter(
        snapshot,
        load_fn=load,
        chat_template_fn=template,
        stream_generate_fn=stream,
    )

    assert adapter.generate_text(
        "text request", max_tokens=31, token=CancellationToken()
    ) == "answer"
    assert adapter.generate_audio(
        wav,
        TimeRange(start_seconds=0, end_seconds=30),
        "audio request",
        max_tokens=47,
        token=CancellationToken(),
    ) == "answer"
    assert adapter.model is model
    assert load_calls == [snapshot]
    assert template_calls == [
        {
            "processor": processor,
            "config": model.config,
            "prompt": "text request",
            "num_audios": 0,
            "enable_thinking": False,
        },
        {
            "processor": processor,
            "config": model.config,
            "prompt": "audio request",
            "num_audios": 1,
            "enable_thinking": False,
        },
    ]
    assert stream_calls == [
        {
            "model": model,
            "processor": processor,
            "prompt": "formatted:text request",
            "audio": None,
            "image": None,
            "video": None,
            "max_tokens": 31,
            "temperature": 0.0,
            "verbose": False,
            "skip_special_tokens": True,
        },
        {
            "model": model,
            "processor": processor,
            "prompt": "formatted:audio request",
            "audio": str(wav),
            "image": None,
            "video": None,
            "max_tokens": 47,
            "temperature": 0.0,
            "verbose": False,
            "skip_special_tokens": True,
        },
    ]


def test_qwen3_omni_checks_cancellation_around_streaming_and_closes_generator(
    tmp_path: Path,
):
    from engine.models.qwen3_omni import Qwen3OmniAdapter

    snapshot = _snapshot(tmp_path, "candidate")
    model = SimpleNamespace(config=object())
    processor = object()
    token = CancellationToken()
    events: list[str] = []

    def stream(*_args, **_kwargs):
        try:
            yield SimpleNamespace(text="first")
            token.cancel()
            yield SimpleNamespace(text="second")
        finally:
            events.append("closed")

    adapter = Qwen3OmniAdapter(
        snapshot,
        load_fn=lambda _path: (model, processor),
        chat_template_fn=lambda *_args, **_kwargs: "formatted",
        stream_generate_fn=stream,
    )

    with pytest.raises(JobCancelled):
        adapter.generate_text("request", max_tokens=2, token=token)
    assert events == ["closed"]

    cancelled = CancellationToken()
    cancelled.cancel()
    with pytest.raises(JobCancelled):
        adapter.generate_text("request", max_tokens=2, token=cancelled)

    completion_token = CancellationToken()

    def cancel_on_completion(*_args, **_kwargs):
        yield SimpleNamespace(text="complete")
        completion_token.cancel()

    completion_adapter = Qwen3OmniAdapter(
        snapshot,
        load_fn=lambda _path: (model, processor),
        chat_template_fn=lambda *_args, **_kwargs: "formatted",
        stream_generate_fn=cancel_on_completion,
    )
    with pytest.raises(JobCancelled):
        completion_adapter.generate_text(
            "request", max_tokens=2, token=completion_token
        )


def test_qwen3_omni_rechecks_cancellation_after_load_and_chat_template(
    tmp_path: Path,
):
    from engine.models.qwen3_omni import Qwen3OmniAdapter

    snapshot = _snapshot(tmp_path, "candidate")
    model = SimpleNamespace(config=object())
    after_load = CancellationToken()
    chat_calls: list[str] = []

    def load(_path):
        after_load.cancel()
        return model, object()

    adapter = Qwen3OmniAdapter(
        snapshot,
        load_fn=load,
        chat_template_fn=lambda *_args, **_kwargs: chat_calls.append("chat")
        or "formatted",
        stream_generate_fn=lambda *_args, **_kwargs: iter(()),
    )
    with pytest.raises(JobCancelled):
        adapter.generate_text("request", max_tokens=2, token=after_load)
    assert chat_calls == []

    after_template = CancellationToken()
    stream_calls: list[str] = []

    def template(*_args, **_kwargs):
        after_template.cancel()
        return "formatted"

    adapter = Qwen3OmniAdapter(
        snapshot,
        load_fn=lambda _path: (model, object()),
        chat_template_fn=template,
        stream_generate_fn=lambda *_args, **_kwargs: stream_calls.append("stream")
        or iter(()),
    )
    with pytest.raises(JobCancelled):
        adapter.generate_text("request", max_tokens=2, token=after_template)
    assert stream_calls == []


def test_qwen3_omni_rejects_invalid_chunks_and_unbounded_or_non_wav_audio(
    tmp_path: Path,
):
    from engine.models.qwen3_omni import Qwen3OmniAdapter

    snapshot = _snapshot(tmp_path, "candidate")
    model = SimpleNamespace(config=object())
    adapter = Qwen3OmniAdapter(
        snapshot,
        load_fn=lambda _path: (model, object()),
        chat_template_fn=lambda *_args, **_kwargs: "formatted",
        stream_generate_fn=lambda *_args, **_kwargs: iter([SimpleNamespace(text=1)]),
    )

    with pytest.raises(ValueError, match="invalid Qwen3-Omni output"):
        adapter.generate_text("request", max_tokens=2, token=CancellationToken())
    with pytest.raises(ValueError, match="30 seconds"):
        adapter.generate_audio(
            _wav(tmp_path, 30.1),
            TimeRange(start_seconds=0, end_seconds=30),
            "request",
            max_tokens=2,
            token=CancellationToken(),
        )
    not_wav = tmp_path / "audio.mp3"
    not_wav.write_bytes(b"not a wav")
    with pytest.raises(ValueError, match="normalized WAV"):
        adapter.generate_audio(
            not_wav,
            TimeRange(start_seconds=0, end_seconds=1),
            "request",
            max_tokens=2,
            token=CancellationToken(),
        )


def test_qwen3_manifest_is_candidate_bound_and_registry_selects_qwen3_once(
    tmp_path: Path,
):
    from engine.models.registry import (
        ModelRegistry,
        QWEN3_OMNI_SPEC,
        load_verified_model_paths,
    )

    baseline_path = _baseline_manifest(tmp_path)
    baseline = json.loads(baseline_path.read_text())
    candidate = _candidate_snapshot(tmp_path)
    baseline["candidate"] = "qwen3_omni"
    baseline["models"]["text_audio"] = {
        **QWEN3_OMNI_SPEC,
        "local_path": str(candidate),
        "files": [
            _inventory_record(path, candidate)
            for path in sorted(candidate.iterdir())
        ],
    }
    manifest = tmp_path / "qwen3-omni-manifest.json"
    manifest.write_text(json.dumps(baseline))

    paths = load_verified_model_paths(manifest, candidate="qwen3_omni")
    calls: list[Path] = []
    qwen3 = object()
    registry = ModelRegistry(
        paths,
        candidate="qwen3_omni",
        load_qwen3_omni=lambda path: calls.append(path) or qwen3,
    )

    assert registry.text_generator is qwen3
    assert registry.audio_understanding is qwen3
    assert calls == [candidate]
    with pytest.raises(ValueError, match="candidate"):
        load_verified_model_paths(manifest, candidate="qwen2_audio")

    tokenizer = candidate / "tokenizer.json"
    tokenizer.unlink()
    baseline["models"]["text_audio"]["files"] = [
        record
        for record in baseline["models"]["text_audio"]["files"]
        if record["path"] != "tokenizer.json"
    ]
    manifest.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="incomplete"):
        load_verified_model_paths(manifest, candidate="qwen3_omni")


def test_fetch_qwen3_reuses_verified_baseline_and_writes_separate_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetch_models = _load_script("fetch_models")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    baseline_path = _baseline_manifest(models_dir)
    baseline_before = baseline_path.read_bytes()
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"]))
        _candidate_snapshot(destination.parent)
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    manifest = fetch_models.fetch(models_dir, candidate="qwen3_omni")

    assert baseline_path.read_bytes() == baseline_before
    assert calls == [
        {
            "repo_id": "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
            "revision": "93b3cbddd65ed4babff8f22fba491cdba7a21778",
            "local_dir": str((models_dir / "qwen3_omni").resolve()),
        }
    ]
    assert manifest["candidate"] == "qwen3_omni"
    assert (
        models_dir / "qwen3-omni-manifest.json"
    ).read_text() == json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    assert manifest["models"]["asr"] == json.loads(baseline_before)["models"]["asr"]
    assert (
        manifest["models"]["text_audio"]["revision"]
        == "93b3cbddd65ed4babff8f22fba491cdba7a21778"
    )


def test_fetch_qwen3_can_manifest_an_existing_download_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetch_models = _load_script("fetch_models")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _baseline_manifest(models_dir)
    candidate = _candidate_snapshot(models_dir)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("existing candidate must not be downloaded again")
            )
        ),
    )

    manifest = fetch_models.fetch(
        models_dir, candidate="qwen3_omni", download_candidate=False
    )

    assert manifest["models"]["text_audio"]["local_path"] == str(candidate)


def test_runtime_measurements_distinguish_pre_cancel_from_live_acknowledgement():
    runtime = _load_script("benchmark_runtime")
    clock_values = iter([1.0, 1.1, 2.0, 2.4, 2.5, 2.8])

    pre = runtime.measure_pre_cancelled(
        lambda token, _started: token.raise_if_cancelled(),
        clock=lambda: next(clock_values),
    )

    def live(token, started):
        started()
        while not token.cancelled:
            pass
        token.raise_if_cancelled()

    active = runtime.measure_active_cancellation(
        live,
        clock=lambda: next(clock_values),
    )

    assert pre == {
        "status": "passed",
        "call_started": False,
        "acknowledgement_latency_seconds": pytest.approx(0.1),
    }
    assert active == {
        "status": "passed",
        "call_started": True,
        "first_activity_latency_seconds": pytest.approx(0.4),
        "acknowledgement_latency_seconds": pytest.approx(0.3),
    }


def test_runtime_checkpoint_is_atomic_and_stdout_redacts_private_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    runtime = _load_script("benchmark_runtime")
    output = tmp_path / "runtime.json"
    checkpoint = runtime.RuntimeCheckpoint(output, candidate="qwen3_omni")

    checkpoint.record(
        "active_audio",
        {
            "status": "failed",
            "error_code": "ValueError",
            "reason": "private path /private/audio.wav and raw output",
        },
    )
    runtime.print_safe_runtime_result(json.loads(output.read_text()))

    payload = json.loads(output.read_text())
    stdout = capsys.readouterr().out
    assert payload["status"] == "running"
    assert payload["completed_measurements"] == 1
    assert payload["measurements"]["active_audio"]["reason"].startswith("private")
    assert json.loads(stdout) == {
        "candidate": "qwen3_omni",
        "completed_measurements": 1,
        "status": "running",
    }
    assert "private" not in stdout


def test_runtime_checkpoint_recovers_after_a_failed_atomic_record(
    tmp_path: Path,
):
    runtime = _load_script("benchmark_runtime")
    output = tmp_path / "runtime.json"
    checkpoint = runtime.RuntimeCheckpoint(output, candidate="qwen3_omni")
    checkpoint.record("first", {"status": "passed"})
    prior = output.read_text()

    with pytest.raises(TypeError):
        checkpoint.record("invalid", {"status": "running", "value": object()})

    assert output.read_text() == prior
    checkpoint.record("second", {"status": "passed"})
    payload = json.loads(output.read_text())
    assert set(payload["measurements"]) == {"first", "second"}


def test_model_smoke_exercises_realistic_metadata_prompts_without_raw_output(
    tmp_path: Path,
):
    smoke = _load_script("model_smoke")
    contracts = _load_script("benchmark_contracts")
    wav = _wav(tmp_path)
    candidates = _retrieval_candidates()
    calls: list[object] = []

    class TextAudio:
        def generate_text(self, prompt, *, max_tokens, token):
            assert "museum attractions" in prompt
            assert all(str(item["taxonomy_id"]) in prompt for item in candidates)
            calls.append(("text", prompt, max_tokens))
            return json.dumps(
                {
                    "taxonomy_ids": ["155"],
                    "keywords": [],
                    "entities": [],
                    "sentiment": {"label": "neutral", "score": 0.0},
                }
            )

        def generate_audio(self, audio, window, prompt, *, max_tokens, token):
            assert audio == wav
            calls.append(("audio", prompt, max_tokens))
            return json.dumps(
                {
                    "is_music": False,
                    "vocal_presence": "unknown",
                    "genres": [],
                    "moods": [],
                }
            )

    def retrieve(texts, token):
        calls.append(("retrieve", texts))
        return candidates

    result = smoke.exercise_realistic_metadata(
        TextAudio(),
        wav,
        TimeRange(start_seconds=0, end_seconds=30),
        transcript="museum attractions",
        retrieve=retrieve,
    )

    assert calls[0] == ("audio", contracts.ACOUSTIC_JSON_PROMPT, 160)
    assert calls[1] == ("retrieve", [("transcript", "museum attractions")])
    assert calls[2][0] == "text"
    assert calls[2][2] == 320
    assert result["text"]["status"] == "passed"
    assert result["audio"]["status"] == "passed"
    assert result["retrieval"]["status"] == "passed"
    assert result["text"]["candidate_count"] == 15
    assert "response" not in json.dumps(result)
    assert result["text"]["output_utf8_bytes"] > 0


def test_model_smoke_checkpoints_each_prompt_before_a_later_call_fails(
    tmp_path: Path,
):
    smoke = _load_script("model_smoke")
    wav = _wav(tmp_path)
    isolation_seen: list[str] = []
    realistic_seen: list[str] = []

    class IsolationModel:
        def generate_text(self, prompt, *, max_tokens, token):
            return '{"request":"text-first"}'

        def generate_audio(self, *_args, **_kwargs):
            raise RuntimeError("later audio failure")

    with pytest.raises(RuntimeError, match="later audio"):
        smoke.exercise_prompt_isolation(
            IsolationModel(),
            wav,
            TimeRange(start_seconds=0, end_seconds=30),
            on_measurement=lambda name, _result: isolation_seen.append(name),
        )
    assert isolation_seen == ["isolation_text_1", "isolation_audio_1"]

    class RealisticModel:
        def generate_audio(self, *_args, **_kwargs):
            raise RuntimeError("later audio failure")

    result = smoke.exercise_realistic_metadata(
        RealisticModel(),
        wav,
        TimeRange(start_seconds=0, end_seconds=30),
        transcript="actual transcript",
        retrieve=lambda _texts, _token: _retrieval_candidates(),
        on_measurement=lambda name, _result: realistic_seen.append(name),
    )
    assert realistic_seen == [
        "realistic_audio",
        "realistic_retrieval",
        "realistic_text",
    ]
    assert result["audio"]["error_kind"] == "execution"
    assert result["retrieval"]["status"] == "unattempted"
    assert result["text"]["status"] == "unattempted"


def test_model_smoke_classifies_schema_failures_and_keeps_independent_controls_running(
    tmp_path: Path,
):
    smoke = _load_script("model_smoke")
    wav = _wav(tmp_path)
    calls: list[str] = []

    class TextAudio:
        def generate_audio(self, *_args, **_kwargs):
            calls.append("audio")
            return '{"is_music":"false","vocal_presence":"unknown","genres":[],"moods":[]}'

        def generate_text(self, *_args, **_kwargs):
            calls.append("text")
            return '{"taxonomy_ids":[]}'

    result = smoke.exercise_realistic_metadata(
        TextAudio(),
        wav,
        TimeRange(start_seconds=0, end_seconds=30),
        transcript="actual transcript",
        retrieve=lambda texts, _token: calls.append(f"retrieve:{texts[0][1]}")
        or _retrieval_candidates(),
    )

    assert calls == ["audio", "retrieve:actual transcript", "text"]
    assert result["audio"]["status"] == "failed"
    assert result["audio"]["error_kind"] == "schema"
    assert result["retrieval"]["status"] == "passed"
    assert result["text"]["status"] == "failed"
    assert result["text"]["error_kind"] == "schema"


def test_runtime_cancellation_suite_skips_failed_model_but_measures_independent_asr():
    smoke = _load_script("model_smoke")
    called: list[str] = []
    actions = {
        "asr": lambda _token, _started: called.append("asr"),
        "text": lambda _token, _started: called.append("text"),
        "audio": lambda _token, _started: called.append("audio"),
    }

    result = smoke.measure_cancellation_suite(
        actions,
        unavailable_modes={
            "text": "prior_execution_failure",
            "audio": "prior_execution_failure",
        },
        pre_measure=lambda action: action(None, None) or {"status": "passed"},
        active_measure=lambda action: action(None, None) or {"status": "passed"},
    )

    assert called == ["asr", "asr"]
    assert result["pre_cancelled"]["text"] == {
        "status": "unattempted",
        "reason_code": "prior_execution_failure",
    }
    assert result["active"]["audio"] == {
        "status": "unattempted",
        "reason_code": "prior_execution_failure",
    }


def test_model_smoke_checkpoints_failed_basic_model_measurements():
    smoke = _load_script("model_smoke")
    seen: list[tuple[str, dict[str, object]]] = []

    with pytest.raises(RuntimeError, match="model failure"):
        smoke._timed_model_call(
            "asr_1",
            lambda: (_ for _ in ()).throw(RuntimeError("model failure")),
            on_measurement=lambda name, result: seen.append((name, result)),
        )

    assert seen == [
        (
            "asr_1",
            {
                "status": "failed",
                "error_code": "RuntimeError",
                "reason": "model failure",
            },
        )
    ]


@pytest.mark.parametrize("script_name", ["model_smoke", "accuracy_harness"])
def test_candidate_cli_defaults_to_qwen2_and_accepts_qwen3(
    script_name: str,
):
    script = _load_script(script_name)

    required = ["--manifest", "/private/labels.json"]
    assert script.parse_args(required).candidate == "qwen2_audio"
    assert (
        script.parse_args([*required, "--candidate", "qwen3_omni"]).candidate
        == "qwen3_omni"
    )
    with pytest.raises(SystemExit):
        script.parse_args([*required, "--candidate", "unsupported"])


def test_model_smoke_cli_routes_qwen3_candidate_and_keeps_stdout_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    smoke = _load_script("model_smoke")
    labels = tmp_path / "private-labels.json"
    labels.write_text("{}")
    output = tmp_path / "runtime.json"
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        smoke,
        "run",
        lambda *args: calls.append(args)
        or {"status": "passed", "reason": "private raw result"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_smoke.py",
            "--manifest",
            str(labels),
            "--candidate",
            "qwen3_omni",
            "--output",
            str(output),
        ],
    )

    assert smoke.main() == 0

    assert calls[0][2] == "qwen3_omni"
    assert json.loads(capsys.readouterr().out) == {
        "candidate": "qwen3_omni",
        "completed_measurements": 0,
        "status": "passed",
    }
    assert "private raw result" in output.read_text()


def test_accuracy_cli_routes_qwen3_manifest_registry_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    harness = _load_script("accuracy_harness")
    labels = tmp_path / "private-labels.json"
    labels.write_text("{}")
    output = tmp_path / "accuracy.json"
    selected: list[tuple[Path, str]] = []
    monkeypatch.setattr(harness, "_load_json", lambda _path: {"thresholds": {}})
    monkeypatch.setattr(harness, "validate_manifest", lambda _payload: [])
    monkeypatch.setattr(
        harness,
        "load_verified_model_paths",
        lambda path, *, candidate: selected.append((path, candidate)) or object(),
    )
    monkeypatch.setattr(
        harness,
        "ModelRegistry",
        lambda _paths, *, candidate: SimpleNamespace(candidate=candidate),
    )
    monkeypatch.setattr(harness, "MlxObservationRunner", lambda _registry: object())
    monkeypatch.setattr(harness, "generate_observations", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(harness, "score", lambda *_args: {"status": "passed"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy_harness.py",
            "--manifest",
            str(labels),
            "--models-dir",
            str(tmp_path / "models"),
            "--candidate",
            "qwen3_omni",
            "--output",
            str(output),
        ],
    )

    assert harness.main() == 0

    assert selected == [
        (tmp_path / "models" / "qwen3-omni-manifest.json", "qwen3_omni")
    ]
    assert json.loads(output.read_text())["candidate"] == "qwen3_omni"
    assert "private" not in capsys.readouterr().out


def test_qwen3_adapter_runtime_evidence_uses_audio_language_and_vision_tensors(
    tmp_path: Path,
):
    from engine.models.qwen3_omni import Qwen3OmniAdapter

    class FakeMlxArray:
        __module__ = "mlx.core"

    weight = FakeMlxArray()
    model = SimpleNamespace(
        config=object(),
        parameters=lambda: {"weight": weight},
        thinker=SimpleNamespace(
            audio_tower=SimpleNamespace(conv2d1=SimpleNamespace(weight=weight)),
            language_model=SimpleNamespace(
                model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight))
            ),
            vision_tower=SimpleNamespace(
                patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=weight))
            ),
        ),
    )
    adapter = Qwen3OmniAdapter(
        _snapshot(tmp_path, "candidate"),
        load_fn=lambda _path: (model, object()),
    )

    evidence = adapter.runtime_evidence()

    assert evidence["parameter_leaves_are_mlx"] is True
    assert set(evidence["tensor_types"]) == {
        "audio_tower_input_weight",
        "language_model_embedding_weight",
        "vision_tower_input_weight",
    }
    assert evidence["tensors_are_mlx"] is True
