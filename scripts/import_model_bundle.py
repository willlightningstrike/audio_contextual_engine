"""Import a verified local model bundle without downloads or worktree links."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any

from engine.models.registry import (
    MODEL_SPECS,
    load_verified_production_paths,
    load_verified_text_path,
    load_verified_yamnet_path,
)
from engine.signals.taxonomy import load_taxonomy_index
from engine.standards import load_taxonomy


MANIFEST_NAMES = (
    "english-phase1-manifest.json", "qwen36-manifest.json", "yamnet-manifest.json",
)


def _verified_paths(root: Path) -> dict[str, Path]:
    paths = load_verified_production_paths(root / MANIFEST_NAMES[0])
    paths["qwen36_text"] = load_verified_text_path(root / MANIFEST_NAMES[1])
    paths["yamnet"] = load_verified_yamnet_path(root / MANIFEST_NAMES[2])

    class NoopEmbedder:
        def encode(self, texts: list[str]) -> Any:
            raise AssertionError("bundle import must not run inference")

    load_taxonomy_index(
        root / "taxonomy-index",
        taxonomy_path=Path(__file__).resolve().parents[1] / "data/standards/content-taxonomy-3.1.tsv",
        taxonomy=load_taxonomy(), embedder=NoopEmbedder(),
        expected_embedding_revision=MODEL_SPECS["embeddings"]["revision"],
    )
    index = root / "taxonomy-index"
    if any(path.is_symlink() for path in index.rglob("*")) or index.is_symlink():
        raise ValueError("taxonomy index must not contain symlinks")
    paths["taxonomy-index"] = index
    return paths


def _copy_directory(source: Path, destination: Path) -> None:
    if platform.system() == "Darwin":
        # APFS clones are independent regular files sharing unchanged disk blocks.
        result = subprocess.run(
            ["/bin/cp", "-cR", str(source), str(destination)], capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        if destination.exists():
            shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _write_manifests(payloads: dict[str, Any], location: Path, paths_root: Path) -> None:
    documents = deepcopy(payloads)
    for role, item in documents[MANIFEST_NAMES[0]]["models"].items():
        item["local_path"] = str(paths_root / role)
    documents[MANIFEST_NAMES[1]]["model"]["local_path"] = str(paths_root / "qwen36_text")
    documents[MANIFEST_NAMES[2]]["artifact"]["local_path"] = str(paths_root / "yamnet")
    for name, payload in documents.items():
        (location / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def import_bundle(source: Path, destination: Path) -> Path:
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists; refusing to overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    source = source.expanduser().resolve(strict=True)
    if destination.is_relative_to(source):
        raise ValueError("destination must be outside the source bundle")
    paths = _verified_paths(source)
    if any(destination.is_relative_to(path) for path in paths.values()):
        raise ValueError("destination must be outside every source asset directory")
    payloads = {name: json.loads((source / name).read_text()) for name in MANIFEST_NAMES}
    with tempfile.TemporaryDirectory(prefix=".model-import-", dir=destination.parent) as directory:
        staging = Path(directory).resolve()
        for role, path in paths.items():
            _copy_directory(path, staging / role)
        _write_manifests(payloads, staging, staging)
        _verified_paths(staging)
        _write_manifests(payloads, staging, destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"destination appeared during import: {destination}")
        staging.rename(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--target-models", type=Path, default=Path("data/models"))
    args = parser.parse_args()
    target = import_bundle(args.source_models, args.target_models)
    print(json.dumps({"models_dir": str(target), "downloads": False, "symlinked_models": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
