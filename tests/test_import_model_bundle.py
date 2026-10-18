from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import import_model_bundle as importer


def _bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    components = {name: source / name for name in
                  ("asr", "embeddings", "qwen25_text", "yamnet", "taxonomy-index")}
    for name, directory in components.items():
        directory.mkdir()
        (directory / "asset").write_bytes(name.encode())
    marker = {"files": [{"path": "asset", "sha256": "preserved", "bytes": 3}],
              "repository": "pinned", "revision": "pinned"}
    payloads = {
        importer.MANIFEST_NAMES[0]: {"models": {name: {**marker, "local_path": str(components[name])}
                                              for name in ("asr", "embeddings")}},
        importer.MANIFEST_NAMES[1]: {"model": {**marker, "local_path": str(components["qwen25_text"])}},
        importer.MANIFEST_NAMES[2]: {"artifact": {**marker, "local_path": str(components["yamnet"])}},
    }
    for name, payload in payloads.items():
        (source / name).write_text(json.dumps(payload))

    def verify(root):
        docs = {name: json.loads((root / name).read_text()) for name in importer.MANIFEST_NAMES}
        items = {**docs[importer.MANIFEST_NAMES[0]]["models"],
                 "qwen25_text": docs[importer.MANIFEST_NAMES[1]]["model"],
                 "yamnet": docs[importer.MANIFEST_NAMES[2]]["artifact"]}
        for role, item in items.items():
            assert item["files"] == marker["files"]
            assert Path(item["local_path"]) == root / role
            assert (root / role / "asset").read_bytes() == role.encode()
        return {role: root / role for role in components}

    monkeypatch.setattr(importer, "_verified_paths", verify)
    monkeypatch.setattr(importer, "_copy_directory", shutil.copytree)
    return source


def test_import_rebases_paths_preserves_assets_and_leaves_source_intact(tmp_path, monkeypatch):
    source = _bundle(tmp_path, monkeypatch)
    original = (source / importer.MANIFEST_NAMES[0]).read_bytes()
    target = importer.import_bundle(source, tmp_path / "project/data/models")
    importer._verified_paths(target)
    assert (source / importer.MANIFEST_NAMES[0]).read_bytes() == original
    assert all(not path.is_symlink() for path in target.rglob("*"))


def test_import_refuses_existing_destination_before_reading_source(tmp_path):
    target = tmp_path / "models"
    target.mkdir()
    (target / "keep").write_text("user data")
    with pytest.raises(FileExistsError):
        importer.import_bundle(tmp_path / "missing", target)
    assert (target / "keep").read_text() == "user data"


def test_copy_failure_does_not_publish_partial_bundle(tmp_path, monkeypatch):
    source = _bundle(tmp_path, monkeypatch)

    def fail_copy(*_args):
        raise OSError("copy failed")

    monkeypatch.setattr(importer, "_copy_directory", fail_copy)
    target = tmp_path / "models"
    with pytest.raises(OSError, match="copy failed"):
        importer.import_bundle(source, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".model-import-*"))


def test_import_rejects_destination_inside_external_model_asset(tmp_path, monkeypatch):
    source = _bundle(tmp_path, monkeypatch)
    external = tmp_path / "external-model"
    external.mkdir()
    monkeypatch.setattr(importer, "_verified_paths", lambda _: {"asr": external})
    with pytest.raises(ValueError, match="outside every source asset"):
        importer.import_bundle(source, external / "new-bundle")
    assert not (external / "new-bundle").exists()
