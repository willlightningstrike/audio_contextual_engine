from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_prepare_script():
    path = Path(__file__).parents[1] / "scripts" / "prepare_yamnet.py"
    spec = importlib.util.spec_from_file_location("prepare_yamnet_explicit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_downloads_are_hash_pinned_and_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = _load_prepare_script()
    contents = {
        "params.py": b"params source",
        "features.py": b"features source",
        "yamnet.py": b"model source",
        "LICENSE": b"license",
        "yamnet.h5": b"checkpoint",
    }
    prepare.REFERENCE_FILES = {
        name: {
            "url": f"https://example.invalid/{name}",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in contents.items()
    }
    calls: list[str] = []

    def urlopen(url: str, *, timeout: float):
        assert timeout > 0
        calls.append(url)
        return io.BytesIO(contents[Path(url).name])

    monkeypatch.setattr(prepare, "urlopen", urlopen)
    reference = tmp_path / "reference"

    assert prepare.download_reference(reference) == reference
    assert calls == [entry["url"] for entry in prepare.REFERENCE_FILES.values()]
    assert prepare.download_reference(reference) == reference
    assert len(calls) == len(prepare.REFERENCE_FILES)

    (reference / "yamnet.h5").write_bytes(b"changed")
    with pytest.raises(ValueError, match="YAMNet reference file hash mismatch"):
        prepare.download_reference(reference)


def test_prepare_writes_a_verified_manifest_without_tensorflow_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import engine.models.registry as registry

    prepare = _load_prepare_script()
    assets = {
        "weights.npz": b"fake weights",
        "frontend.npz": b"fake frontend",
        "LICENSE": b"LICENSE",
    }
    expected = {
        name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in assets.items()
    }
    monkeypatch.setattr(registry, "YAMNET_ASSET_FILES", expected)
    prepare.YAMNET_ASSET_FILES = expected
    source_files = {
        name: {
            "url": f"https://example.invalid/{name}",
            "bytes": len(name.encode()),
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        for name in ("params.py", "features.py", "yamnet.py", "LICENSE", "yamnet.h5")
    }
    prepare.REFERENCE_FILES = source_files
    reference = tmp_path / "reference"
    reference.mkdir()
    for name in source_files:
        (reference / name).write_bytes(name.encode())

    def export(_reference: Path, asset_dir: Path) -> None:
        (asset_dir / "weights.npz").write_bytes(assets["weights.npz"])
        (asset_dir / "frontend.npz").write_bytes(assets["frontend.npz"])

    output = tmp_path / "prepared"
    manifest_path = prepare.prepare_yamnet(
        output, reference_dir=reference, converter=export
    )

    assert manifest_path == output / "manifest.json"
    assert registry.load_verified_yamnet_path(manifest_path) == output / "assets"


def test_prepare_refuses_to_overwrite_an_existing_artifact_directory(tmp_path: Path) -> None:
    prepare = _load_prepare_script()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user data")

    with pytest.raises(FileExistsError, match="output already exists"):
        prepare.prepare_yamnet(output)

    assert (output / "keep.txt").read_text() == "user data"
