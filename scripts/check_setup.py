"""Fail-fast, local-only validation before the service is constructed."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

from engine.config import Settings


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DISTRIBUTIONS = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "sentence-transformers",
        "tensorflow",
        "tensorflow-macos",
        "tensorflow-cpu",
        "mlx-cpu",
        "mlx-cuda",
    }
)
PINNED_IMPORTS = {
    "fastapi": ("fastapi",),
    "mlx": ("mlx.core",),
    "mlx-audio": ("mlx_audio",),
    "mlx-embeddings": ("mlx_embeddings",),
    "mlx-lm": ("mlx_lm",),
    "mlx-vlm": ("mlx_vlm",),
    "numpy": ("numpy",),
    "outlines": ("outlines",),
    "outlines-core": ("outlines_core",),
    "pydantic": ("pydantic",),
    "python-multipart": ("multipart",),
    "transformers": ("transformers",),
    "uvicorn": ("uvicorn",),
}
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
MINIMUM_MEMORY_BYTES = 64 * 1024**3


def enable_offline_mode() -> None:
    """Prevent model or tokenizer libraries from reaching the network."""
    os.environ.update(OFFLINE_ENVIRONMENT)


def _physical_memory_bytes() -> int:
    if platform.system() != "Darwin":
        return 0
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _host_identity() -> tuple[str, str, str, int]:
    return (
        platform.system(),
        platform.mac_ver()[0],
        platform.machine().lower(),
        _physical_memory_bytes(),
    )


def _platform_failures() -> tuple[str, ...]:
    system, version, machine, memory_bytes = _host_identity()
    failures: list[str] = []
    if system != "Darwin":
        failures.append("Contextual Engine requires macOS 14 or newer on Apple Silicon.")
    else:
        try:
            major = int(version.split(".", maxsplit=1)[0])
        except (AttributeError, ValueError):
            major = 0
        if major < 14:
            failures.append("Contextual Engine requires macOS 14 or newer.")
        if machine != "arm64":
            failures.append("Contextual Engine requires an Apple Silicon arm64 host.")
    if memory_bytes < MINIMUM_MEMORY_BYTES:
        failures.append("Contextual Engine requires at least 64 GiB of physical memory.")
    return tuple(failures)


def _package_pins() -> dict[str, str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    pins: dict[str, str] = {}
    for requirement in dependencies:
        name, separator, version = requirement.partition("==")
        if separator and name and version:
            pins[_normalize_distribution(name)] = version
    return pins


def _normalize_distribution(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name and version:
            versions[_normalize_distribution(name)] = version
    return versions


def _runtime_failures() -> tuple[str, ...]:
    installed = _installed_versions()
    failures = []
    pins = _package_pins()
    for name, expected in pins.items():
        actual = installed.get(name)
        if actual is None:
            failures.append(f"Pinned runtime package is missing: {name}=={expected}.")
        elif actual != expected:
            failures.append(
                f"Pinned runtime package version mismatch: {name}=={expected}; found {actual}."
            )
    for distribution, module_names in PINNED_IMPORTS.items():
        if distribution not in pins:
            continue
        for module_name in module_names:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                available = False
            if not available:
                failures.append(
                    f"Pinned runtime import is unavailable: {module_name} ({distribution})."
                )
    forbidden = sorted(set(installed) & {
        _normalize_distribution(name) for name in FORBIDDEN_DISTRIBUTIONS
    })
    if forbidden:
        failures.append(
            "Forbidden runtime distributions are installed: " + ", ".join(forbidden) + "."
        )
    if any(os.environ.get(key) != value for key, value in OFFLINE_ENVIRONMENT.items()):
        failures.append("Model libraries are not configured for offline operation.")
    return tuple(failures)


def _tool_failures() -> tuple[str, ...]:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if not missing:
        return ()
    return ("Required media tools are missing from PATH: " + ", ".join(missing) + ".",)


def _verify_standards() -> tuple[str, ...]:
    manifest_path = ROOT / "data" / "standards" / "manifest.json"
    taxonomy_path = ROOT / "data" / "standards" / "content-taxonomy-3.0.tsv"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(taxonomy_path.read_bytes()).hexdigest()
    except (OSError, ValueError, TypeError) as exc:
        return (f"Vendored standards verification failed: {exc}.",)

    expected_constants = {"video": 1, "music": 3, "other": 6, "unknown": 7}
    if (
        manifest.get("iab_content_taxonomy_version") != "3.0"
        or manifest.get("openrtb_cattax_content_3_0") != 7
        or manifest.get("context_constants") != expected_constants
        or manifest.get("sha256") != digest
    ):
        return ("Vendored standards manifest, checksum, or context constants do not match.",)
    return ()


def _check_writable_directory(path: Path, *, label: str) -> tuple[str, ...]:
    try:
        if path.is_symlink():
            return (f"{label} must not be a symlink: {path}.",)
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not path.is_dir():
            return (f"{label} is not a directory: {path}.",)
        with tempfile.TemporaryFile(dir=path):
            pass
    except OSError as exc:
        return (f"{label} is not writable: {path} ({exc}).",)
    return ()


def _verify_local_assets(settings: Settings) -> tuple[str, ...]:
    model_dir = settings.models_dir
    failures: list[str] = []
    production_manifest = model_dir / "english-phase1-manifest.json"
    text_manifest = model_dir / "qwen25-manifest.json"
    yamnet_manifest = model_dir / "yamnet-manifest.json"
    for label, manifest in (
        ("Production model manifest", production_manifest),
        ("Qwen2.5 model manifest", text_manifest),
        ("YAMNet asset manifest", yamnet_manifest),
    ):
        if not manifest.is_file() or manifest.is_symlink():
            failures.append(f"{label} is missing or unsafe: {manifest}.")

    if production_manifest.is_file() and not production_manifest.is_symlink():
        try:
            from engine.models.registry import load_verified_production_paths

            load_verified_production_paths(production_manifest)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"Production model manifest verification failed: {exc}.")
    if text_manifest.is_file() and not text_manifest.is_symlink():
        try:
            from engine.models.registry import load_verified_text_path

            load_verified_text_path(text_manifest)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"Qwen2.5 model manifest verification failed: {exc}.")
    if yamnet_manifest.is_file() and not yamnet_manifest.is_symlink():
        try:
            from engine.models.registry import load_verified_yamnet_path

            load_verified_yamnet_path(yamnet_manifest)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            failures.append(f"YAMNet asset manifest verification failed: {exc}.")

    index_manifest = settings.taxonomy_index_dir / "manifest.json"
    if not index_manifest.is_file() or index_manifest.is_symlink():
        failures.append(f"Taxonomy index manifest is missing or unsafe: {index_manifest}.")
    else:
        try:
            from engine.models.registry import MODEL_SPECS
            from engine.signals.taxonomy import load_taxonomy_index
            from engine.standards import load_taxonomy

            class NoopEmbedder:
                def encode(self, texts: list[str]) -> Any:
                    del texts
                    raise AssertionError("setup validation must not encode text")

            load_taxonomy_index(
                settings.taxonomy_index_dir,
                taxonomy_path=ROOT / "data" / "standards" / "content-taxonomy-3.0.tsv",
                taxonomy=load_taxonomy(),
                embedder=NoopEmbedder(),
                expected_embedding_revision=MODEL_SPECS["embeddings"]["revision"],
            )
        except (
            ImportError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(f"Taxonomy index provenance verification failed: {exc}.")
    return tuple(failures)


def check_setup(settings: Settings) -> tuple[str, ...]:
    """Return local setup errors; never downloads or contacts a model service."""
    failures = [
        *_platform_failures(),
        *_runtime_failures(),
        *_tool_failures(),
        *_verify_standards(),
    ]
    failures.extend(_check_writable_directory(settings.data_dir, label="Data directory"))
    failures.extend(_check_writable_directory(settings.jobs_dir, label="Job directory"))
    failures.extend(_check_writable_directory(settings.uploads_root, label="Upload directory"))
    failures.extend(_verify_local_assets(settings))
    if settings.bind_host not in {"127.0.0.1", "localhost", "::1"}:
        failures.append("The service bind address must be loopback-only.")
    if not 1 <= settings.port <= 65_535:
        failures.append("The service port must be between 1 and 65535.")
    return tuple(failures)


def main() -> int:
    enable_offline_mode()
    failures = check_setup(Settings.from_environment())
    if failures:
        print("Contextual Engine setup failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Contextual Engine setup passed; local assets verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
