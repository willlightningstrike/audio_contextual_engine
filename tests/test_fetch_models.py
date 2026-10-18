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


def test_cli_explicit_qwen25_reuses_existing_snapshot_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import engine.models.registry as registry

    fetch_models = _load_script()
    models_dir = tmp_path / "models"
    snapshot = models_dir / "qwen25_text"
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
    monkeypatch.setattr(registry, "QWEN25_WEIGHT_FILES", expected)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("existing explicit Qwen2.5 snapshot must not download")
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
            "qwen25_text",
            "--reuse-downloaded-candidate",
        ],
    )

    assert fetch_models.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "candidate": "qwen25_text",
        "manifest": str((models_dir / "qwen25-manifest.json").resolve()),
        "roles": ["text"],
    }
    assert json.loads((models_dir / "qwen25-manifest.json").read_text())["model"][
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
    monkeypatch.setattr(registry, "QWEN25_WEIGHT_FILES", text_pins)
    downloads: list[tuple[str, str]] = []

    def downloader(*, repo_id: str, revision: str, local_dir: str) -> str:
        downloads.append((repo_id, revision))
        target = Path(local_dir)
        target.mkdir(parents=True)
        if repo_id == registry.QWEN25_TEXT_SPEC["repository"]:
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
    ] + [(registry.QWEN25_TEXT_SPEC["repository"], registry.QWEN25_TEXT_SPEC["revision"])]
    assert downloads == expected
    assert set(result["models"]) == {"asr", "embeddings"}
    assert result["bundle"] == "english_phase1"
    assert (tmp_path / "models" / "yamnet-manifest.json").is_file()
    stray = tmp_path / "models" / "yamnet" / "old-conversion.npz"
    stray.write_bytes(b"stale converted file")
    with pytest.raises(ValueError, match="unexpected files"):
        fetch_models._install_yamnet_assets(tmp_path / "models", yamnet_source)
