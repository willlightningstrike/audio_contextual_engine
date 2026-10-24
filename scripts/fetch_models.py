"""Fetch immutable local model snapshots and write a cryptographic manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from engine.models.registry import (
    CANDIDATES,
    EMBEDDING_WEIGHT_FILES,
    MODEL_SPECS,
    PRODUCTION_MODEL_SPECS,
    QWEN3_OMNI_SPEC,
    QWEN36_TEXT_SPEC,
    YAMNET_CONVERTER_VERSION,
    YAMNET_ASSET_FILES,
    YAMNET_SOURCE_PROVENANCE,
    candidate_manifest_path,
    load_verified_asr_path,
    load_verified_model_paths,
    load_verified_production_paths,
    load_verified_text_path,
    load_verified_yamnet_path,
    text_manifest_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(snapshot: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise RuntimeError(f"download produced an empty snapshot: {snapshot}")
    return files


def _write_manifest(target: Path, manifest: dict[str, Any]) -> None:
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def _fetch_baseline(root: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    models: dict[str, Any] = {}
    for role, spec in MODEL_SPECS.items():
        destination = root / role
        result = Path(
            snapshot_download(
                repo_id=spec["repository"],
                revision=spec["revision"],
                local_dir=str(destination),
            )
        ).resolve(strict=True)
        if result != destination.resolve():
            raise RuntimeError(f"snapshot destination mismatch for {role}: {result}")
        models[role] = {
            "repository": spec["repository"],
            "revision": spec["revision"],
            "local_path": str(result),
            "files": _inventory(result),
        }
    return {"schema_version": 1, "models": models}


def _fetch_qwen3(root: Path, *, download_candidate: bool) -> dict[str, Any]:
    baseline_path = candidate_manifest_path(root, "qwen2_audio")
    load_verified_model_paths(baseline_path, candidate="qwen2_audio")
    baseline = json.loads(baseline_path.read_text())
    destination = (root / "qwen3_omni").resolve()
    if download_candidate:
        from huggingface_hub import snapshot_download

        result = Path(
            snapshot_download(
                repo_id=QWEN3_OMNI_SPEC["repository"],
                revision=QWEN3_OMNI_SPEC["revision"],
                local_dir=str(destination),
            )
        ).resolve(strict=True)
        if result != destination:
            raise RuntimeError(
                f"snapshot destination mismatch for qwen3_omni: {result}"
            )
    elif not destination.is_dir():
        raise FileNotFoundError(
            f"existing Qwen3-Omni snapshot is missing: {destination}"
        )
    files = _inventory(destination)
    file_names = {record["path"] for record in files}
    required = {
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    if not required.issubset(file_names) or not any(
        str(name).endswith(".safetensors") for name in file_names
    ):
        raise RuntimeError("Qwen3-Omni snapshot is missing required model files")
    models = {
        "asr": baseline["models"]["asr"],
        "embeddings": baseline["models"]["embeddings"],
        "text_audio": {
            **QWEN3_OMNI_SPEC,
            "local_path": str(destination),
            "files": files,
        },
    }
    return {"schema_version": 1, "candidate": "qwen3_omni", "models": models}


def fetch_text_model(
    models_dir: Path,
    *,
    download: bool = True,
    downloader: Any | None = None,
) -> dict[str, Any]:
    """Download or register only the pinned local Qwen3.6 text snapshot."""
    root = Path(models_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = (root / "qwen36_text").resolve()
    if download:
        active_downloader = downloader
        if active_downloader is None:
            from huggingface_hub import snapshot_download

            active_downloader = snapshot_download

        result = Path(
            active_downloader(
                repo_id=QWEN36_TEXT_SPEC["repository"],
                revision=QWEN36_TEXT_SPEC["revision"],
                local_dir=str(destination),
            )
        ).resolve(strict=True)
        if result != destination:
            raise RuntimeError(f"snapshot destination mismatch for Qwen3.6: {result}")
    elif not destination.is_dir():
        raise FileNotFoundError(f"existing Qwen3.6 snapshot is missing: {destination}")
    manifest = {
        "schema_version": 1,
        "model": {
            **QWEN36_TEXT_SPEC,
            "local_path": str(destination),
            "files": _inventory(destination),
        },
    }
    target = text_manifest_path(root)
    _write_manifest(target, manifest)
    try:
        load_verified_text_path(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return manifest


def _install_yamnet_assets(models_dir: Path, source_assets: Path) -> Path:
    """Copy only the already-converted, verified MLX assets into model storage."""
    source_dir = load_verified_yamnet_path(Path(source_assets) / "manifest.json")
    target_dir = models_dir / "yamnet"
    manifest_path = models_dir / "yamnet-manifest.json"
    if target_dir.exists():
        if target_dir.is_symlink() or not target_dir.is_dir():
            raise ValueError("existing YAMNet destination is not a safe directory")
        actual_names = {
            path.relative_to(target_dir).as_posix()
            for path in target_dir.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual_names != set(YAMNET_ASSET_FILES):
            raise ValueError("existing YAMNet destination contains unexpected files")
        existing = {
            name: {
                "bytes": (target_dir / name).stat().st_size,
                "sha256": _sha256(target_dir / name),
            }
            for name in YAMNET_ASSET_FILES
            if (target_dir / name).is_file() and not (target_dir / name).is_symlink()
        }
        if existing != YAMNET_ASSET_FILES:
            raise ValueError("existing YAMNet destination is not the pinned asset set")
    else:
        staging = Path(tempfile.mkdtemp(prefix="yamnet-install-", dir=models_dir))
        try:
            for name in YAMNET_ASSET_FILES:
                shutil.copyfile(source_dir / name, staging / name)
            staging.rename(target_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    target_dir = target_dir.resolve(strict=True)
    target_manifest = {
        "schema_version": 1,
        "kind": "yamnet_mlx_v1",
        "converter": YAMNET_CONVERTER_VERSION,
        "source": YAMNET_SOURCE_PROVENANCE,
        "artifact": {"local_path": str(target_dir), "files": _inventory(target_dir)},
    }
    _write_manifest(manifest_path, target_manifest)
    load_verified_yamnet_path(manifest_path)
    return target_dir


def fetch_english_phase1(
    models_dir: Path,
    *,
    yamnet_assets: Path,
    download: bool = True,
    downloader: Any | None = None,
) -> dict[str, Any]:
    """Install only ASR, embeddings, Qwen3.6 and verified MLX YAMNet assets."""
    root = Path(models_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Validate the separately produced YAMNet artifact before starting large fetches.
    load_verified_yamnet_path(Path(yamnet_assets) / "manifest.json")
    active_downloader = downloader
    if download and active_downloader is None:
        from huggingface_hub import snapshot_download

        active_downloader = snapshot_download

    models: dict[str, Any] = {}
    for role, spec in PRODUCTION_MODEL_SPECS.items():
        destination = (root / role).resolve()
        if download:
            assert active_downloader is not None
            result = Path(
                active_downloader(
                    repo_id=spec["repository"],
                    revision=spec["revision"],
                    local_dir=str(destination),
                )
            ).resolve(strict=True)
            if result != destination:
                raise RuntimeError(f"snapshot destination mismatch for {role}: {result}")
        elif not destination.is_dir():
            raise FileNotFoundError(f"existing {role} snapshot is missing: {destination}")
        models[role] = {
            **spec,
            "local_path": str(destination),
            "files": _inventory(destination),
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "bundle": "english_phase1",
        "models": models,
    }
    manifest_path = root / "english-phase1-manifest.json"
    _write_manifest(manifest_path, manifest)
    load_verified_production_paths(manifest_path)
    fetch_text_model(root, download=download, downloader=active_downloader)
    _install_yamnet_assets(root, yamnet_assets)
    return manifest


def fetch_embedding_model(
    models_dir: Path,
    *,
    downloader: Any | None = None,
) -> dict[str, Any]:
    """Replace only the installed embedder: stage, verify pinned weights, then swap."""
    root = Path(models_dir).expanduser().resolve()
    manifest_path = root / "english-phase1-manifest.json"
    # Only ASR is checked: the embeddings entry may still pin the previous embedder.
    load_verified_asr_path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    spec = PRODUCTION_MODEL_SPECS["embeddings"]
    destination = root / "embeddings"
    staging = root / "embeddings.staging"
    previous = root / "embeddings.previous"
    if staging.exists():
        shutil.rmtree(staging)
    active_downloader = downloader
    if active_downloader is None:
        from huggingface_hub import snapshot_download

        active_downloader = snapshot_download
    try:
        result = Path(
            active_downloader(
                repo_id=spec["repository"],
                revision=spec["revision"],
                local_dir=str(staging),
            )
        ).resolve(strict=True)
        if result != staging.resolve():
            raise RuntimeError(f"snapshot destination mismatch for embeddings: {result}")
        files = _inventory(staging)
        by_path = {record["path"]: record for record in files}
        for name, pinned in EMBEDDING_WEIGHT_FILES.items():
            record = by_path.get(name)
            if record is None or (record["bytes"], record["sha256"]) != (pinned["bytes"], pinned["sha256"]):
                raise ValueError(f"embedding snapshot does not match the pinned weights: {name}")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.rename(previous)
    staging.rename(destination)
    # Inventory paths are relative to the snapshot, so they survive the rename.
    manifest["models"]["embeddings"] = {**spec, "local_path": str(destination), "files": files}
    _write_manifest(manifest_path, manifest)
    load_verified_production_paths(manifest_path)
    shutil.rmtree(previous, ignore_errors=True)
    return manifest


def fetch(
    models_dir: Path,
    *,
    candidate: str,
    download_candidate: bool = True,
) -> dict[str, Any]:
    root = models_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if candidate not in CANDIDATES:
        raise ValueError(f"unsupported model candidate: {candidate}")
    manifest = (
        _fetch_baseline(root)
        if candidate == "qwen2_audio"
        else _fetch_qwen3(root, download_candidate=download_candidate)
    )
    target = candidate_manifest_path(root, candidate)
    _write_manifest(target, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument(
        "--candidate",
        choices=(*CANDIDATES, "qwen36_text", "english_phase1", "embeddings"),
        required=True,
    )
    parser.add_argument(
        "--yamnet-assets",
        type=Path,
        help="prepared YAMNet MLX assets from the separate reference conversion operation",
    )
    parser.add_argument(
        "--reuse-downloaded-candidate",
        action="store_true",
        help="inventory an existing Qwen3-Omni directory without downloading it again",
    )
    args = parser.parse_args()
    if args.candidate == "english_phase1":
        if args.yamnet_assets is None:
            parser.error("--candidate english_phase1 requires --yamnet-assets")
        manifest = fetch_english_phase1(
            args.models_dir,
            yamnet_assets=args.yamnet_assets,
            download=not args.reuse_downloaded_candidate,
        )
        manifest_path = args.models_dir.resolve() / "english-phase1-manifest.json"
        roles = sorted(manifest["models"])
    elif args.candidate == "embeddings":
        manifest = fetch_embedding_model(args.models_dir)
        manifest_path = args.models_dir.resolve() / "english-phase1-manifest.json"
        roles = ["embeddings"]
    elif args.candidate == "qwen36_text":
        manifest = fetch_text_model(
            args.models_dir,
            download=not args.reuse_downloaded_candidate,
        )
        manifest_path = text_manifest_path(args.models_dir.resolve())
        roles = ["text"]
    else:
        manifest = fetch(
            args.models_dir,
            candidate=args.candidate,
            download_candidate=not args.reuse_downloaded_candidate,
        )
        manifest_path = candidate_manifest_path(args.models_dir.resolve(), args.candidate)
        roles = sorted(manifest["models"])
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "manifest": str(manifest_path),
                "roles": roles,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
