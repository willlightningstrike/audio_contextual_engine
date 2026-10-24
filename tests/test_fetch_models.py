from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "fetch_models.py"
    spec = importlib.util.spec_from_file_location("fetch_models_explicit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_requires_explicit_candidate_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_models = _load_script()
    monkeypatch.setattr(sys, "argv", ["fetch_models.py"])
    monkeypatch.setattr(
        fetch_models,
        "fetch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model setup must not start without --candidate")
        ),
    )
    monkeypatch.setattr(
        fetch_models,
        "fetch_text_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("text-model setup must not start without --candidate")
        ),
    )

    with pytest.raises(SystemExit) as raised:
        fetch_models.main()

    assert raised.value.code == 2


def test_fetch_requires_explicit_candidate_before_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch_models = _load_script()
    monkeypatch.setattr(
        fetch_models,
        "_fetch_baseline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy model setup must be explicitly selected")
        ),
    )

    with pytest.raises(TypeError, match="candidate"):
        fetch_models.fetch(tmp_path / "models")

    assert not (tmp_path / "models").exists()


def test_cli_explicit_qwen36_reuses_existing_snapshot_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import engine.models.registry as registry

    fetch_models = _load_script()
    models_dir = tmp_path / "models"
    snapshot = models_dir / "qwen36_text"
    snapshot.mkdir(parents=True)
    expected: dict[str, dict[str, object]] = {}
    for number in range(1, 5):
        name = f"model-{number:05d}-of-00004.safetensors"
        payload = f"weight-{number}".encode()
        (snapshot / name).write_bytes(payload)
        expected[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    monkeypatch.setattr(registry, "QWEN36_WEIGHT_FILES", expected)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("existing explicit Qwen3.6 snapshot must not download")
            )
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_models.py",
            "--models-dir",
            str(models_dir),
            "--candidate",
            "qwen36_text",
            "--reuse-downloaded-candidate",
        ],
    )

    assert fetch_models.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "candidate": "qwen36_text",
        "manifest": str((models_dir / "qwen36-manifest.json").resolve()),
        "roles": ["text"],
    }
    assert json.loads((models_dir / "qwen36-manifest.json").read_text())["model"][
        "local_path"
    ] == str(snapshot.resolve())


def test_english_phase1_bundle_downloads_only_selected_model_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import engine.models.registry as registry

    fetch_models = _load_script()
    yamnet_source = tmp_path / "prepared-yamnet"
    artifact_dir = yamnet_source / "assets"
    artifact_dir.mkdir(parents=True)
    source_files = {
        "weights.npz": b"weights",
        "frontend.npz": b"frontend",
        "LICENSE": b"license",
    }
    yamnet_pins: dict[str, dict[str, object]] = {}
    records = []
    for name, content in source_files.items():
        (artifact_dir / name).write_bytes(content)
        record = {
            "path": name,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        records.append(record)
        yamnet_pins[name] = {"bytes": record["bytes"], "sha256": record["sha256"]}
    monkeypatch.setattr(registry, "YAMNET_ASSET_FILES", yamnet_pins)
    monkeypatch.setattr(fetch_models, "YAMNET_ASSET_FILES", yamnet_pins)
    (yamnet_source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "yamnet_mlx_v1",
                "converter": registry.YAMNET_CONVERTER_VERSION,
                "source": registry.YAMNET_SOURCE_PROVENANCE,
                "artifact": {"local_path": str(artifact_dir.resolve()), "files": records},
            }
        )
    )
    text_pins = {
        "model-00001-of-00004.safetensors": {
            "bytes": 10,
            "sha256": hashlib.sha256(b"text shard").hexdigest(),
        }
    }
    monkeypatch.setattr(registry, "QWEN36_WEIGHT_FILES", text_pins)
    downloads: list[tuple[str, str]] = []

    def downloader(*, repo_id: str, revision: str, local_dir: str) -> str:
        downloads.append((repo_id, revision))
        target = Path(local_dir)
        target.mkdir(parents=True)
        if repo_id == registry.QWEN36_TEXT_SPEC["repository"]:
            (target / "model-00001-of-00004.safetensors").write_bytes(b"text shard")
        else:
            (target / "config.json").write_text(repo_id)
        return str(target)

    result = fetch_models.fetch_english_phase1(
        tmp_path / "models", yamnet_assets=yamnet_source, downloader=downloader
    )

    expected = [
        (spec["repository"], spec["revision"])
        for spec in registry.PRODUCTION_MODEL_SPECS.values()
    ] + [(registry.QWEN36_TEXT_SPEC["repository"], registry.QWEN36_TEXT_SPEC["revision"])]
    assert downloads == expected
    assert set(result["models"]) == {"asr", "embeddings"}
    assert result["bundle"] == "english_phase1"
    assert (tmp_path / "models" / "yamnet-manifest.json").is_file()
    stray = tmp_path / "models" / "yamnet" / "old-conversion.npz"
    stray.write_bytes(b"stale converted file")
    with pytest.raises(ValueError, match="unexpected files"):
        fetch_models._install_yamnet_assets(tmp_path / "models", yamnet_source)


EMBED_4B = {
    "repository": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
    "revision": "b5d88f1fe49b50d2ac01b4692ca2d387f14f9c72",
}


def _production_models_dir(tmp_path: Path, fetch_models) -> tuple[Path, bytes]:
    """An installed bundle whose embeddings entry still pins Qwen3-Embedding-0.6B."""
    import engine.models.registry as registry

    models_dir = tmp_path / "models"
    for role, content in (("asr", b"asr weights"), ("embeddings", b"old")):
        (models_dir / role).mkdir(parents=True)
        (models_dir / role / "model.safetensors").write_bytes(content)
    legacy = {
        "repository": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "revision": "6c3ae70858513f1a78e9cdca3cae330d9075cd2a",
    }
    manifest = {
        "schema_version": 1,
        "bundle": "english_phase1",
        "models": {
            "asr": {**registry.MODEL_SPECS["asr"], "local_path": str((models_dir / "asr").resolve()),
                    "files": fetch_models._inventory(models_dir / "asr")},
            "embeddings": {**legacy, "local_path": str((models_dir / "embeddings").resolve()),
                           "files": fetch_models._inventory(models_dir / "embeddings")},
        },
    }
    path = models_dir / "english-phase1-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return models_dir, path.read_bytes()


def _embedding_pins(monkeypatch: pytest.MonkeyPatch, fetch_models, content: bytes) -> None:
    import engine.models.registry as registry

    pins = {"model.safetensors": {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}
    monkeypatch.setattr(registry, "EMBEDDING_WEIGHT_FILES", pins)
    monkeypatch.setattr(fetch_models, "EMBEDDING_WEIGHT_FILES", pins, raising=False)


def _writer(content: bytes):
    def downloader(*, repo_id: str, revision: str, local_dir: str) -> str:
        assert (repo_id, revision) == (EMBED_4B["repository"], EMBED_4B["revision"])
        target = Path(local_dir)
        target.mkdir(parents=True)
        (target / "model.safetensors").write_bytes(content)
        (target / "config.json").write_text("{}")
        return str(target)
    return downloader


def test_embeddings_spec_pins_qwen3_embedding_4b() -> None:
    import engine.models.registry as registry

    assert registry.MODEL_SPECS["embeddings"] == EMBED_4B
    assert registry.PRODUCTION_MODEL_SPECS["embeddings"] == EMBED_4B
    assert registry.EMBEDDING_WEIGHT_FILES == {"model.safetensors": {
        "bytes": 2_262_632_177,
        "sha256": "cd373d0bc76a3464a69b036ac6beca217a0d668bdf901ff3b05a7d7997d2d607"}}


def test_embedding_refresh_swaps_in_verified_snapshot_and_keeps_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import engine.models.registry as registry

    fetch_models = _load_script()
    models_dir, old_bytes = _production_models_dir(tmp_path, fetch_models)
    _embedding_pins(monkeypatch, fetch_models, b"new 4b weights")

    manifest = fetch_models.fetch_embedding_model(models_dir, downloader=_writer(b"new 4b weights"))

    old = json.loads(old_bytes)
    assert manifest["models"]["embeddings"]["revision"] == EMBED_4B["revision"]
    assert manifest["models"]["asr"] == old["models"]["asr"]
    assert (models_dir / "embeddings" / "model.safetensors").read_bytes() == b"new 4b weights"
    assert not (models_dir / "embeddings.staging").exists()
    assert not (models_dir / "embeddings.previous").exists()
    paths = registry.load_verified_production_paths(models_dir / "english-phase1-manifest.json")
    assert paths["embeddings"] == (models_dir / "embeddings").resolve()


def test_embedding_refresh_accepts_manifest_still_pinning_0_6b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch_models = _load_script()
    models_dir, old_bytes = _production_models_dir(tmp_path, fetch_models)
    assert "0.6B" in json.loads(old_bytes)["models"]["embeddings"]["repository"]
    _embedding_pins(monkeypatch, fetch_models, b"w")

    manifest = fetch_models.fetch_embedding_model(models_dir, downloader=_writer(b"w"))

    assert manifest["models"]["embeddings"]["repository"].endswith("4B-4bit-DWQ")


def test_embedding_refresh_failure_leaves_old_snapshot_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch_models = _load_script()
    models_dir, old_bytes = _production_models_dir(tmp_path, fetch_models)
    _embedding_pins(monkeypatch, fetch_models, b"new")

    def broken(*, repo_id: str, revision: str, local_dir: str) -> str:
        Path(local_dir).mkdir(parents=True)
        (Path(local_dir) / "model.safetensors").write_bytes(b"ne")
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        fetch_models.fetch_embedding_model(models_dir, downloader=broken)

    assert (models_dir / "embeddings" / "model.safetensors").read_bytes() == b"old"
    assert (models_dir / "english-phase1-manifest.json").read_bytes() == old_bytes
    assert not (models_dir / "embeddings.staging").exists()


def test_embedding_refresh_rejects_wrong_weight_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch_models = _load_script()
    models_dir, old_bytes = _production_models_dir(tmp_path, fetch_models)
    _embedding_pins(monkeypatch, fetch_models, b"genuine")

    with pytest.raises(ValueError, match="embedding"):
        fetch_models.fetch_embedding_model(models_dir, downloader=_writer(b"tampered"))

    assert (models_dir / "embeddings" / "model.safetensors").read_bytes() == b"old"
    assert (models_dir / "english-phase1-manifest.json").read_bytes() == old_bytes
    assert not (models_dir / "embeddings.staging").exists()
