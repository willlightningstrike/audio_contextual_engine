"""Explicit developer-only conversion of pinned official YAMNet inputs to MLX."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable
from urllib.request import urlopen


YAMNET_REVISION = "c14bf9ad91962cf189f9f58db2132c06247fcd53"
YAMNET_SOURCE_PROVENANCE = {
    "repository": "tensorflow/models",
    "revision": YAMNET_REVISION,
    "checkpoint_url": "https://storage.googleapis.com/audioset/yamnet.h5",
    "checkpoint_bytes": 15_296_092,
    "checkpoint_sha256": "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744",
    "tensorflow_version": "2.19.1",
    "tf_keras_version": "2.19.0",
}
YAMNET_ASSET_FILES = {
    "weights.npz": {
        "bytes": 15_038_944,
        "sha256": "8dacef0002f1b4eec29af9f40c0e2dc76fe76dde38e8572dab06d4ef94d9acdb",
    },
    "frontend.npz": {
        "bytes": 67_892,
        "sha256": "f9bde8b7c1f30f0d733278aab88bda6431d3c610a423af8677a9c2ac4cb83601",
    },
    "LICENSE": {
        "bytes": 11_512,
        "sha256": "5b17814bf0de8cf65069bc6d7cc38cff19fcaa864d243423ad3ef3db01b52385",
    },
}
CONVERTER_VERSION = "yamnet-tf-reference-to-mlx-npz-v1"
_SOURCE_ROOT = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    f"{YAMNET_REVISION}/research/audioset/yamnet"
)
REFERENCE_FILES = {
    "params.py": {
        "url": f"{_SOURCE_ROOT}/params.py",
        "bytes": 1_847,
        "sha256": "925bb1e62461016031f98aea09aeac28975dd516f5747513767de5d1b06b6145",
    },
    "features.py": {
        "url": f"{_SOURCE_ROOT}/features.py",
        "bytes": 7_490,
        "sha256": "e6cd53f81d072c7c43be4c7fff2b9dd0c5ccc7d64f2fbcfc85b44013d6d2ed5e",
    },
    "yamnet.py": {
        "url": f"{_SOURCE_ROOT}/yamnet.py",
        "bytes": 5_541,
        "sha256": "7ef3df32b7ecb782490b5a04d7581a23cfcf701dbf476bc4d03deefe22cdb040",
    },
    "LICENSE": {
        "url": f"https://raw.githubusercontent.com/tensorflow/models/{YAMNET_REVISION}/LICENSE",
        "bytes": 11_512,
        "sha256": "5b17814bf0de8cf65069bc6d7cc38cff19fcaa864d243423ad3ef3db01b52385",
    },
    "yamnet.h5": {
        "url": YAMNET_SOURCE_PROVENANCE["checkpoint_url"],
        "bytes": YAMNET_SOURCE_PROVENANCE["checkpoint_bytes"],
        "sha256": YAMNET_SOURCE_PROVENANCE["checkpoint_sha256"],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, record: dict[str, Any], *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"YAMNet {label} is missing or unsafe: {path.name}")
    if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
        raise ValueError(f"YAMNet {label} hash mismatch: {path.name}")


def _inventory(directory: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    records = []
    for name in names:
        path = directory / name
        _verify_file(path, YAMNET_ASSET_FILES[name], label="converted asset")
        records.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def download_reference(
    reference_dir: Path,
    *,
    downloader: Callable[..., Any] | None = None,
) -> Path:
    """Download or reuse only the source files pinned to the reviewed commit."""
    root = Path(reference_dir)
    if root.is_symlink():
        raise ValueError("YAMNet reference directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    active_downloader = downloader or urlopen
    for name, record in REFERENCE_FILES.items():
        target = root / name
        if target.exists():
            _verify_file(target, record, label="reference file")
            continue
        if not record["url"].startswith("https://"):
            raise ValueError("YAMNet reference downloads must use HTTPS")
        temporary = root / f".{name}.download"
        try:
            with closing(active_downloader(record["url"], timeout=60)) as response:
                content = response.read(record["bytes"] + 1)
            if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"YAMNet reference file hash mismatch: {name}")
            temporary.write_bytes(content)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return root


def _export_reference(reference_dir: Path, asset_dir: Path) -> None:
    """Run the official reference model only in this explicit converter process."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(reference_dir))
    try:
        import numpy as np
        import tensorflow as tf
        import tf_keras

        if tf.__version__ != YAMNET_SOURCE_PROVENANCE["tensorflow_version"]:
            raise RuntimeError(f"expected TensorFlow {YAMNET_SOURCE_PROVENANCE['tensorflow_version']}")
        if tf_keras.__version__ != YAMNET_SOURCE_PROVENANCE["tf_keras_version"]:
            raise RuntimeError(f"expected tf-keras {YAMNET_SOURCE_PROVENANCE['tf_keras_version']}")
        import params
        import yamnet

        model = yamnet.yamnet_frames_model(params.Params())
        model.load_weights(str(reference_dir / "yamnet.h5"))
        weights = {variable.name: variable.numpy() for variable in model.weights}
        if len(weights) != len(model.weights):
            raise RuntimeError("YAMNet checkpoint contains duplicate tensor names")
        np.savez(asset_dir / "weights.npz", **weights)
        mel = tf.signal.linear_to_mel_weight_matrix(
            64, 257, 16_000, 125.0, 7_500.0
        ).numpy()
        hann = tf.signal.hann_window(400, periodic=True).numpy()
        np.savez(asset_dir / "frontend.npz", mel=mel, hann=hann)
    finally:
        try:
            sys.path.remove(str(reference_dir))
        except ValueError:
            pass


def prepare_yamnet(
    output_dir: Path,
    *,
    reference_dir: Path | None = None,
    downloader: Callable[..., Any] | None = None,
    converter: Callable[[Path, Path], None] | None = None,
) -> Path:
    """Create a hash-verified asset folder and provenance manifest atomically."""
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"YAMNet output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".yamnet-prepare-", dir=destination.parent))
    try:
        reference = (
            download_reference(staging / "reference", downloader=downloader)
            if reference_dir is None
            else Path(reference_dir)
        )
        for name, record in REFERENCE_FILES.items():
            _verify_file(reference / name, record, label="reference file")
        assets = staging / "assets"
        assets.mkdir()
        (converter or _export_reference)(reference, assets)
        shutil.copyfile(reference / "LICENSE", assets / "LICENSE")
        records = _inventory(assets, tuple(YAMNET_ASSET_FILES))
        payload = {
            "schema_version": 1,
            "kind": "yamnet_mlx_v1",
            "converter": CONVERTER_VERSION,
            "source": YAMNET_SOURCE_PROVENANCE,
            "artifact": {
                "local_path": str(destination / "assets"),
                "files": records,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explicit developer-only YAMNet conversion; TensorFlow is never an app dependency."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="reuse a directory containing verified official source files and yamnet.h5",
    )
    args = parser.parse_args()
    manifest = prepare_yamnet(args.output, reference_dir=args.reference_dir)
    print(json.dumps({"manifest": str(manifest), "converter": CONVERTER_VERSION}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
