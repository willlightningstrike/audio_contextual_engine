from __future__ import annotations

from pathlib import Path

import pytest


def test_text_identity_hashes_exact_verified_manifest_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    from dataclasses import FrozenInstanceError
    from engine.models import registry

    content = b"test local shard"
    weights = {"model.safetensors": {"bytes": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()}}
    monkeypatch.setattr(registry, "QWEN25_WEIGHT_FILES", weights)
    snapshot = tmp_path / "text"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(content)
    manifest = tmp_path / "text-manifest.json"
    payload = {"schema_version": 1, "model": {
        "repository": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "revision": "2938092373e5f97b95538884112085364c2da315",
        "local_path": str(snapshot.resolve()),
        "files": [{"path": "model.safetensors", **weights["model.safetensors"]}],
    }}
    exact_bytes = (json.dumps(payload, indent=3) + "\n\n").encode()
    manifest.write_bytes(exact_bytes)
    monkeypatch.setattr(registry, "load_verified_asr_path", lambda _: snapshot.resolve())
    monkeypatch.setattr(registry, "load_verified_embedding_path", lambda _: snapshot.resolve())
    models = registry.EnglishPipelineModelRegistry(
        asr_manifest=tmp_path / "unused-asr", embedding_manifest=tmp_path / "unused-embedder",
        text_manifest=manifest, yamnet_path=snapshot.resolve())
    identity = models.text_model_identity
    assert identity.repository == "mlx-community/Qwen2.5-32B-Instruct-4bit"
    assert identity.revision == "2938092373e5f97b95538884112085364c2da315"
    assert identity.manifest_sha256 == hashlib.sha256(exact_bytes).hexdigest()
    assert identity.manifest_sha256 != hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    with pytest.raises(FrozenInstanceError):
        identity.revision = "other"


def test_settings_respects_explicit_data_directory_without_changing_home(
    tmp_path: Path,
) -> None:
    from engine.config import Settings

    original_home = Path.home()
    data_dir = tmp_path / "engine-data"
    settings = Settings.from_environment(
        environ={"CONTEXTUAL_ENGINE_DATA_DIR": str(data_dir)}
    )

    assert settings.data_dir == data_dir
    assert Path.home() == original_home
    assert settings.jobs_dir == data_dir / "jobs"
    assert settings.uploads_root == data_dir / "uploads"
    assert settings.bind_host == "127.0.0.1"
    assert {".mkv", ".webm"}.issubset(settings.allowed_suffixes)


@pytest.mark.parametrize(
    ("system", "version", "machine", "memory_bytes", "expected"),
    [
        ("Linux", "14.0", "arm64", 64 * 1024**3, "macOS 14"),
        ("Darwin", "13.6", "arm64", 64 * 1024**3, "macOS 14"),
        ("Darwin", "14.0", "x86_64", 64 * 1024**3, "Apple Silicon"),
        ("Darwin", "14.0", "arm64", 63 * 1024**3, "64 GiB"),
    ],
)
def test_platform_preflight_rejects_unsupported_hosts(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    version: str,
    machine: str,
    memory_bytes: int,
    expected: str,
) -> None:
    from scripts import check_setup

    monkeypatch.setattr(
        check_setup,
        "_host_identity",
        lambda: (system, version, machine, memory_bytes),
    )

    failures = check_setup._platform_failures()

    assert any(expected.casefold() in failure.casefold() for failure in failures)


def test_missing_model_assets_fail_locally_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socket

    from engine.config import Settings
    from scripts import check_setup

    monkeypatch.setattr(check_setup, "_platform_failures", lambda: ())
    monkeypatch.setattr(check_setup, "_runtime_failures", lambda: ())
    monkeypatch.setattr(check_setup, "_tool_failures", lambda: ())
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight attempted network access")
        ),
    )
    monkeypatch.setattr(
        socket.socket,
        "connect_ex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline preflight attempted network access")
        ),
    )

    settings = Settings(data_dir=tmp_path / "app-data")
    failures = check_setup.check_setup(settings)

    assert any("production model manifest" in failure.casefold() for failure in failures)
    assert any("taxonomy index" in failure.casefold() for failure in failures)
    assert settings.jobs_dir.is_dir()
    assert settings.uploads_root.is_dir()


def test_production_manifest_requires_exact_english_phase1_roles(tmp_path: Path) -> None:
    import hashlib
    import json

    from engine.models.registry import MODEL_SPECS, load_verified_production_paths

    models: dict[str, dict[str, object]] = {}
    expected_paths: dict[str, Path] = {}
    for role in ("asr", "embeddings"):
        snapshot = tmp_path / role
        snapshot.mkdir()
        content = (role + " snapshot").encode()
        (snapshot / "config.json").write_bytes(content)
        models[role] = {
            **MODEL_SPECS[role],
            "local_path": str(snapshot.resolve()),
            "files": [
                {
                    "path": "config.json",
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ],
        }
        expected_paths[role] = snapshot.resolve()
    manifest = tmp_path / "english-phase1-manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "bundle": "english_phase1", "models": models})
    )

    assert load_verified_production_paths(manifest) == expected_paths

    models["text_audio"] = dict(models["asr"])
    manifest.write_text(
        json.dumps({"schema_version": 1, "bundle": "english_phase1", "models": models})
    )
    with pytest.raises(ValueError, match="exactly the required production roles"):
        load_verified_production_paths(manifest)


def test_yamnet_manifest_requires_pinned_source_conversion_and_exact_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from engine.models import registry

    source = dict(registry.YAMNET_SOURCE_PROVENANCE)
    files = {
        "weights.npz": b"converted weights",
        "frontend.npz": b"frozen frontend",
        "LICENSE": b"Apache License",
    }
    expected = {
        name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in files.items()
    }
    monkeypatch.setattr(registry, "YAMNET_ASSET_FILES", expected)
    artifact_dir = tmp_path / "yamnet"
    artifact_dir.mkdir()
    records = []
    for name, content in files.items():
        (artifact_dir / name).write_bytes(content)
        records.append({"path": name, **expected[name]})
    manifest = tmp_path / "yamnet-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "yamnet_mlx_v1",
                "converter": "yamnet-tf-reference-to-mlx-npz-v1",
                "source": source,
                "artifact": {"local_path": str(artifact_dir.resolve()), "files": records},
            }
        )
    )

    assert registry.load_verified_yamnet_path(manifest) == artifact_dir.resolve()

    payload = json.loads(manifest.read_text())
    payload["converter"] = "unknown"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source or conversion provenance"):
        registry.load_verified_yamnet_path(manifest)

    source["revision"] = "unreviewed"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "yamnet_mlx_v1",
                "source": source,
                "artifact": {"local_path": str(artifact_dir.resolve()), "files": records},
            }
        )
    )
    with pytest.raises(ValueError, match="source or conversion provenance"):
        registry.load_verified_yamnet_path(manifest)


def test_runtime_preflight_checks_pins_forbidden_distributions_and_offline_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import check_setup

    monkeypatch.setattr(check_setup, "_package_pins", lambda: {"mlx": "0.32.0"})
    monkeypatch.setattr(check_setup, "PINNED_IMPORTS", {"mlx": ("mlx.core",)})
    monkeypatch.setattr(
        check_setup,
        "_installed_versions",
        lambda: {"mlx": "0.32.0", "tensorflow": "2.19.1"},
    )
    monkeypatch.setattr(check_setup.importlib.util, "find_spec", lambda _name: None)
    for key, value in check_setup.OFFLINE_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)

    failures = check_setup._runtime_failures()

    assert any("forbidden runtime distributions" in item.casefold() for item in failures)
    assert any("runtime import" in item.casefold() for item in failures), failures
    assert not any("offline operation" in item.casefold() for item in failures)

    monkeypatch.setattr(check_setup, "_installed_versions", lambda: {"mlx": "0.31.0"})
    monkeypatch.delenv("HF_HUB_OFFLINE")
    failures = check_setup._runtime_failures()
    assert any("version mismatch" in item.casefold() for item in failures)
    assert any("offline operation" in item.casefold() for item in failures)


def test_standards_manifest_is_verified_before_server_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from scripts import check_setup

    standards = tmp_path / "data" / "standards"
    standards.mkdir(parents=True)
    (standards / "manifest.json").write_text(
        json.dumps(
            {
                "iab_content_taxonomy_version": "3.0",
                "sha256": "0" * 64,
                "context_constants": {
                    "video": 1,
                    "music": 3,
                    "other": 6,
                    "unknown": 7,
                },
            }
        )
    )
    (standards / "content-taxonomy-3.0.tsv").write_text("not the pinned taxonomy")
    monkeypatch.setattr(check_setup, "ROOT", tmp_path)

    failures = check_setup._verify_standards()

    assert failures
    assert any("standards" in failure.casefold() for failure in failures)


def test_setup_script_requires_prepared_yamnet_and_uses_english_bundle_only() -> None:
    script = Path(__file__).parents[1] / "scripts" / "setup.sh"
    text = script.read_text(encoding="utf-8")

    assert "--yamnet-assets" in text
    assert "--candidate english_phase1" in text
    assert "--model-manifest" in text
    assert "--no-diagnostics" in text
    assert "scripts.check_setup" in text
    assert "tensorflow" not in text.casefold()
    assert "qwen2_audio" not in text
    assert "qwen3_omni" not in text


def test_preflight_accepts_only_complete_local_production_assets_without_legacy_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib
    import json
    import socket

    import numpy as np

    from engine.config import Settings
    from engine.models import registry
    from engine.signals.taxonomy import build_taxonomy_index
    from engine.standards import load_taxonomy
    from scripts import check_setup

    monkeypatch.setattr(
        check_setup,
        "_host_identity",
        lambda: ("Darwin", "15.0", "arm64", 64 * 1024**3),
    )
    monkeypatch.setattr(check_setup.shutil, "which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(check_setup, "_installed_versions", lambda: check_setup._package_pins())
    for key, value in check_setup.OFFLINE_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline preflight attempted network access")
        ),
    )
    monkeypatch.setattr(
        socket.socket,
        "connect_ex",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("offline preflight attempted network access")
        ),
    )

    settings = Settings(data_dir=tmp_path / "installed")
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True)
    model_roles: dict[str, dict[str, object]] = {}
    for role, spec in registry.PRODUCTION_MODEL_SPECS.items():
        snapshot = models_dir / role
        snapshot.mkdir()
        data = f"{role}-snapshot".encode()
        (snapshot / "config.json").write_bytes(data)
        weights = f"{role}-weights".encode()
        (snapshot / "weights.safetensors").write_bytes(weights)
        model_roles[role] = {
            **spec,
            "local_path": str(snapshot.resolve()),
            "files": [
                {
                    "path": "config.json",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
                {
                    "path": "weights.safetensors",
                    "bytes": len(weights),
                    "sha256": hashlib.sha256(weights).hexdigest(),
                },
            ],
        }
    (models_dir / "english-phase1-manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "bundle": "english_phase1", "models": model_roles}
        )
    )

    text_dir = models_dir / "qwen25_text"
    text_dir.mkdir()
    text_data = b"only a small test shard"
    (text_dir / "model-00001-of-00004.safetensors").write_bytes(text_data)
    monkeypatch.setattr(
        registry,
        "QWEN25_WEIGHT_FILES",
        {
            "model-00001-of-00004.safetensors": {
                "bytes": len(text_data),
                "sha256": hashlib.sha256(text_data).hexdigest(),
            }
        },
    )
    (models_dir / "qwen25-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {
                    **registry.QWEN25_TEXT_SPEC,
                    "local_path": str(text_dir.resolve()),
                    "files": [
                        {
                            "path": "model-00001-of-00004.safetensors",
                            "bytes": len(text_data),
                            "sha256": hashlib.sha256(text_data).hexdigest(),
                        }
                    ],
                },
            }
        )
    )

    yamnet_dir = models_dir / "yamnet"
    yamnet_dir.mkdir()
    yamnet_files = {
        name: (name + "-fixture").encode()
        for name in ("weights.npz", "frontend.npz", "LICENSE")
    }
    yamnet_records = []
    yamnet_pins = {}
    for name, data in yamnet_files.items():
        (yamnet_dir / name).write_bytes(data)
        record = {
            "path": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        yamnet_records.append(record)
        yamnet_pins[name] = {"bytes": record["bytes"], "sha256": record["sha256"]}
    monkeypatch.setattr(registry, "YAMNET_ASSET_FILES", yamnet_pins)
    (models_dir / "yamnet-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "yamnet_mlx_v1",
                "converter": registry.YAMNET_CONVERTER_VERSION,
                "source": registry.YAMNET_SOURCE_PROVENANCE,
                "artifact": {"local_path": str(yamnet_dir.resolve()), "files": yamnet_records},
            }
        )
    )

    class TestEmbedder:
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.ones((len(texts), 2), dtype=np.float32)

    taxonomy_path = (
        Path(__file__).parents[1] / "data" / "standards" / "content-taxonomy-3.0.tsv"
    )
    build_taxonomy_index(
        settings.taxonomy_index_dir,
        taxonomy_path=taxonomy_path,
        taxonomy=load_taxonomy(),
        embedder=TestEmbedder(),
        embedding_revision=registry.MODEL_SPECS["embeddings"]["revision"],
    )

    assert check_setup.check_setup(settings) == ()
    assert not (models_dir / "text_audio").exists()
    assert not (models_dir / "qwen2_audio").exists()
    assert not (models_dir / "qwen3_omni").exists()

    asr_weights = models_dir / "asr" / "weights.safetensors"
    original_weights = asr_weights.read_bytes()
    asr_weights.unlink()
    failures = check_setup.check_setup(settings)
    assert any("production model manifest verification failed" in item.casefold() for item in failures)
    assert not (models_dir / "qwen2_audio").exists()

    asr_weights.write_bytes(original_weights + b"changed")
    failures = check_setup.check_setup(settings)
    assert any("production model manifest verification failed" in item.casefold() for item in failures)

    index_manifest = settings.taxonomy_index_dir / "manifest.json"
    index_payload = json.loads(index_manifest.read_text())
    index_payload["embedding_revision"] = "unreviewed"
    index_manifest.write_text(json.dumps(index_payload))
    failures = check_setup.check_setup(settings)
    assert any("taxonomy index provenance verification failed" in item.casefold() for item in failures)


def test_serve_returns_before_app_construction_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from scripts import serve

    monkeypatch.setattr(
        serve,
        "check_setup",
        lambda _settings: ("missing local model manifest",),
    )
    monkeypatch.setattr(
        serve,
        "build_production_app",
        lambda _settings: (_ for _ in ()).throw(
            AssertionError("the app must not be constructed before preflight passes")
        ),
    )

    assert serve.main() == 1
    output = capsys.readouterr()
    assert "missing local model manifest" in output.err
    assert all(
        __import__("os").environ.get(key) == value
        for key, value in serve.OFFLINE_ENVIRONMENT.items()
    )


def test_production_app_serves_and_shuts_down_worker_with_pinned_fastapi(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from engine.config import Settings
    from engine.models import registry
    from engine import runner as runner_module
    from engine.signals import taxonomy
    from scripts.serve import build_production_app

    # Replace only heavyweight model/index loading, retaining the real app and worker lifecycle.
    monkeypatch.setattr(registry, "EnglishPipelineModelRegistry", lambda **_: SimpleNamespace(embedder=object()))
    monkeypatch.setattr(registry, "load_verified_yamnet_path", lambda _: tmp_path)
    monkeypatch.setattr(taxonomy, "load_taxonomy_index", lambda *_, **__: object())
    workers = []
    real_runner = runner_module.JobRunner
    def make_runner(*args, **kwargs):
        worker = real_runner(*args, **kwargs)
        workers.append(worker)
        return worker
    monkeypatch.setattr(runner_module, "JobRunner", make_runner)
    try:
        app = build_production_app(Settings(data_dir=tmp_path))
        with TestClient(app) as client:
            assert client.get("/").status_code == 200
        with pytest.raises(RuntimeError, match="shut down"):
            asyncio.run(app.state.runner.enqueue("not-enqueued-after-shutdown"))
    finally:
        for worker in workers:
            asyncio.run(worker.shutdown())
