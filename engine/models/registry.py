"""Lazy ownership of the three resident, local-only model adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Generic, TypeVar

from engine.models import require_resolved_local_directory
from engine.models.asr import MlxAsrAdapter
from engine.models.embeddings import MlxEmbeddingAdapter
from engine.models.qwen3_omni import Qwen3OmniAdapter
from engine.models.text_audio import MlxTextAudioAdapter
from engine.models.text import MlxConstrainedTextAdapter
from engine.models.vocals import MlxYamnetAdapter


T = TypeVar("T")


MODEL_SPECS = {
    "asr": {
        "repository": "mlx-community/whisper-large-v3-turbo-asr-fp16",
        "revision": "624c19c9af5603fa73b83bce14d4aeea96156d18",
    },
    "embeddings": {
        "repository": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "revision": "6c3ae70858513f1a78e9cdca3cae330d9075cd2a",
    },
    "text_audio": {
        "repository": "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
        "revision": "c65570002626f41b4dc08b7b54f42f99f3e82e7f",
    },
}
PRODUCTION_MODEL_SPECS = {role: dict(MODEL_SPECS[role]) for role in ("asr", "embeddings")}
YAMNET_SOURCE_PROVENANCE = {
    "repository": "tensorflow/models",
    "revision": "c14bf9ad91962cf189f9f58db2132c06247fcd53",
    "checkpoint_url": "https://storage.googleapis.com/audioset/yamnet.h5",
    "checkpoint_bytes": 15_296_092,
    "checkpoint_sha256": "13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744",
    "tensorflow_version": "2.19.1",
    "tf_keras_version": "2.19.0",
}
YAMNET_CONVERTER_VERSION = "yamnet-tf-reference-to-mlx-npz-v1"
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

QWEN3_OMNI_SPEC = {
    "repository": "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
    "revision": "93b3cbddd65ed4babff8f22fba491cdba7a21778",
}
CANDIDATES = ("qwen2_audio", "qwen3_omni")

QWEN25_TEXT_SPEC = {
    "repository": "mlx-community/Qwen2.5-32B-Instruct-4bit",
    "revision": "2938092373e5f97b95538884112085364c2da315",
}
QWEN25_WEIGHT_FILES = {
    "model-00001-of-00004.safetensors": {
        "bytes": 5_366_582_717,
        "sha256": "3187f89267bdebe362410a3b23c2767d9d0707f4ffbbf7a945e5cd0abf535a21",
    },
    "model-00002-of-00004.safetensors": {
        "bytes": 5_335_712_920,
        "sha256": "31547d7a3e6583eea8bcc1b7230680f8132143c43c19cab3eab75f6b31d33e33",
    },
    "model-00003-of-00004.safetensors": {
        "bytes": 5_366_641_934,
        "sha256": "d00aec153b1ea6540446c1769ef76004e4ee11855844919f48434f0d4f77b33f",
    },
    "model-00004-of-00004.safetensors": {
        "bytes": 2_362_540_888,
        "sha256": "4dae2dd0355ac7ad5440ccd4d5f60a08084cda924c058402b979654359150743",
    },
}


def text_manifest_path(models_dir: Path) -> Path:
    return Path(models_dir) / "qwen25-manifest.json"


def model_specs(candidate: str) -> dict[str, dict[str, str]]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unsupported model candidate: {candidate}")
    specs = {role: dict(spec) for role, spec in MODEL_SPECS.items()}
    if candidate == "qwen3_omni":
        specs["text_audio"] = dict(QWEN3_OMNI_SPEC)
    return specs


def candidate_manifest_path(models_dir: Path, candidate: str) -> Path:
    if candidate not in CANDIDATES:
        raise ValueError(f"unsupported model candidate: {candidate}")
    name = "model-manifest.json" if candidate == "qwen2_audio" else "qwen3-omni-manifest.json"
    return Path(models_dir) / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_snapshot_files(snapshot: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for directory, subdirectories, names in os.walk(snapshot, followlinks=False):
        current = Path(directory)
        for name in [*subdirectories, *names]:
            if (current / name).is_symlink():
                raise ValueError(f"model snapshot contains a symlink: {current / name}")
        for name in names:
            path = current / name
            if not path.is_file():
                raise ValueError(f"model snapshot contains a non-regular file: {path}")
            files[path.relative_to(snapshot).as_posix()] = path
    return files


def _load_manifest_payload(
    manifest_path: Path, *, manifest_name: str
) -> dict[str, Any]:
    requested_path = Path(manifest_path)
    if requested_path.is_symlink():
        raise ValueError("model manifest must not be a symlink")
    try:
        path = requested_path.resolve(strict=True)
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {manifest_name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {manifest_name}")
    return payload


def _parse_snapshot_inventory(
    item: dict[str, Any], *, label: str
) -> tuple[Path, dict[str, dict[str, Any]]]:
    snapshot = Path(item.get("local_path", ""))
    if (
        not snapshot.is_absolute()
        or snapshot.is_symlink()
        or not snapshot.is_dir()
        or snapshot.resolve(strict=True) != snapshot
    ):
        raise ValueError(f"model manifest {label} path is not a resolved local directory")
    inventory = item.get("files")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError(f"model manifest {label} has no file inventory")
    expected_files: dict[str, dict[str, Any]] = {}
    for record in inventory:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("path"), str)
            or not isinstance(record.get("bytes"), int)
            or isinstance(record.get("bytes"), bool)
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
        ):
            raise ValueError(f"model manifest {label} has malformed file inventory")
        relative = Path(record["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in expected_files
        ):
            raise ValueError(f"model manifest {label} has unsafe or duplicate file path")
        expected_files[relative.as_posix()] = record
    return snapshot, expected_files


def _verify_snapshot_inventory(
    snapshot: Path,
    expected_files: dict[str, dict[str, Any]],
    *,
    label: str,
) -> None:
    actual_files = _safe_snapshot_files(snapshot)
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(expected_files))
        raise ValueError(
            f"model snapshot inventory mismatch for {label}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, record in expected_files.items():
        actual = actual_files[relative]
        if actual.stat().st_size != record["bytes"]:
            raise ValueError(f"model snapshot size mismatch for {label}/{relative}")
        if _sha256(actual) != record["sha256"]:
            raise ValueError(f"model snapshot SHA-256 mismatch for {label}/{relative}")


def _load_verified_single_role(manifest_path: Path, role: str) -> Path:
    payload = _load_manifest_payload(manifest_path, manifest_name="model manifest")
    models = payload.get("models")
    if (
        payload.get("schema_version") != 1
        or not isinstance(models, dict)
    ):
        raise ValueError("invalid model manifest")
    item = models.get(role)
    expected = MODEL_SPECS[role]
    if (
        not isinstance(item, dict)
        or item.get("repository") != expected["repository"]
        or item.get("revision") != expected["revision"]
    ):
        raise ValueError(f"model manifest does not pin the required {role} revision")
    snapshot, expected_files = _parse_snapshot_inventory(item, label=role)
    _verify_snapshot_inventory(snapshot, expected_files, label=role)
    return snapshot


def load_verified_embedding_path(manifest_path: Path) -> Path:
    """Verify and expose only the pinned embedder from a legacy full manifest."""
    return _load_verified_single_role(manifest_path, "embeddings")


def load_verified_asr_path(manifest_path: Path) -> Path:
    """Verify and expose only pinned ASR without selecting deleted legacy models."""
    return _load_verified_single_role(manifest_path, "asr")


def load_verified_text_path(manifest_path: Path) -> Path:
    """Verify the pinned local Qwen2.5 snapshot without selecting legacy audio models."""
    payload = _load_manifest_payload(
        manifest_path, manifest_name="Qwen2.5 model manifest"
    )
    item = payload.get("model")
    if (
        payload.get("schema_version") != 1
        or not isinstance(item, dict)
        or item.get("repository") != QWEN25_TEXT_SPEC["repository"]
        or item.get("revision") != QWEN25_TEXT_SPEC["revision"]
    ):
        raise ValueError("model manifest does not pin the required Qwen2.5 revision")
    snapshot, expected_files = _parse_snapshot_inventory(item, label="Qwen2.5")
    if set(QWEN25_WEIGHT_FILES) - set(expected_files):
        raise ValueError("Qwen2.5 manifest is missing one or more pinned weight shards")
    for name, pinned in QWEN25_WEIGHT_FILES.items():
        record = expected_files[name]
        if record["bytes"] != pinned["bytes"] or record["sha256"] != pinned["sha256"]:
            raise ValueError(f"Qwen2.5 pinned weight metadata mismatch for {name}")
    _verify_snapshot_inventory(snapshot, expected_files, label="Qwen2.5")
    return snapshot


def load_verified_production_paths(manifest_path: Path) -> dict[str, Path]:
    """Verify the exact ASR/embedder roles in the current production bundle."""
    payload = _load_manifest_payload(
        manifest_path, manifest_name="English Phase 1 production model manifest"
    )
    models = payload.get("models")
    if (
        payload.get("schema_version") != 1
        or payload.get("bundle") != "english_phase1"
        or not isinstance(models, dict)
        or set(models) != set(PRODUCTION_MODEL_SPECS)
    ):
        raise ValueError("manifest must contain exactly the required production roles")
    verified: dict[str, Path] = {}
    for role, expected in PRODUCTION_MODEL_SPECS.items():
        item = models[role]
        if (
            not isinstance(item, dict)
            or item.get("repository") != expected["repository"]
            or item.get("revision") != expected["revision"]
        ):
            raise ValueError(f"production manifest does not pin the required {role} revision")
        snapshot, inventory = _parse_snapshot_inventory(item, label=role)
        _verify_snapshot_inventory(snapshot, inventory, label=role)
        verified[role] = snapshot
    return verified


def load_verified_yamnet_path(manifest_path: Path) -> Path:
    """Verify the converted YAMNet files and their pinned upstream provenance."""
    payload = _load_manifest_payload(manifest_path, manifest_name="YAMNet asset manifest")
    artifact = payload.get("artifact")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "yamnet_mlx_v1"
        or payload.get("converter") != YAMNET_CONVERTER_VERSION
        or payload.get("source") != YAMNET_SOURCE_PROVENANCE
        or not isinstance(artifact, dict)
    ):
        raise ValueError("YAMNet manifest has unexpected source or conversion provenance")
    snapshot, inventory = _parse_snapshot_inventory(artifact, label="YAMNet")
    if set(inventory) != set(YAMNET_ASSET_FILES):
        raise ValueError("YAMNet manifest must contain exactly the verified assets and license")
    for name, pinned in YAMNET_ASSET_FILES.items():
        if inventory[name].get("bytes") != pinned["bytes"] or inventory[name].get("sha256") != pinned["sha256"]:
            raise ValueError(f"YAMNet manifest does not pin the expected {name}")
    _verify_snapshot_inventory(snapshot, inventory, label="YAMNet")
    return snapshot


def load_verified_model_paths(
    manifest_path: Path, *, candidate: str = "qwen2_audio"
) -> ModelPaths:
    """Load only the complete, hash-verified snapshots recorded by fetch_models."""
    expected_specs = model_specs(candidate)
    requested_path = Path(manifest_path)
    if requested_path.is_symlink():
        raise ValueError("model manifest must not be a symlink")
    path = requested_path.resolve(strict=True)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("models"), dict):
        raise ValueError("invalid model manifest")
    manifest_candidate = payload.get("candidate")
    if candidate == "qwen2_audio":
        if manifest_candidate not in {None, "qwen2_audio"}:
            raise ValueError("model manifest does not match selected candidate")
    elif manifest_candidate != candidate:
        raise ValueError("model manifest does not match selected candidate")
    models = payload["models"]
    if set(models) != set(expected_specs):
        raise ValueError("model manifest roles do not match required model set")
    verified: dict[str, Path] = {}
    for role, expected in expected_specs.items():
        item = models[role]
        if not isinstance(item, dict) or item.get("repository") != expected["repository"] or item.get("revision") != expected["revision"]:
            raise ValueError(f"model manifest does not pin the required {role} revision")
        snapshot = Path(item.get("local_path", ""))
        if not snapshot.is_absolute() or snapshot.is_symlink() or not snapshot.is_dir() or snapshot.resolve(strict=True) != snapshot:
            raise ValueError(f"model manifest {role} path is not a resolved local directory")
        inventory = item.get("files")
        if not isinstance(inventory, list) or not inventory:
            raise ValueError(f"model manifest {role} has no file inventory")
        expected_files: dict[str, dict[str, Any]] = {}
        for record in inventory:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str) or not isinstance(record.get("bytes"), int) or not isinstance(record.get("sha256"), str):
                raise ValueError(f"model manifest {role} has malformed file inventory")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in expected_files:
                raise ValueError(f"model manifest {role} has unsafe or duplicate file path")
            expected_files[relative.as_posix()] = record
        actual_files = _safe_snapshot_files(snapshot)
        if set(actual_files) != set(expected_files):
            missing = sorted(set(expected_files) - set(actual_files))
            unexpected = sorted(set(actual_files) - set(expected_files))
            raise ValueError(f"model snapshot inventory mismatch for {role}; missing={missing}, unexpected={unexpected}")
        for relative, record in expected_files.items():
            actual = actual_files[relative]
            if actual.stat().st_size != record["bytes"]:
                raise ValueError(f"model snapshot size mismatch for {role}/{relative}")
            if _sha256(actual) != record["sha256"]:
                raise ValueError(f"model snapshot SHA-256 mismatch for {role}/{relative}")
        if candidate == "qwen3_omni" and role == "text_audio":
            required = {
                "config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors.index.json",
            }
            if not required.issubset(actual_files) or not any(
                name.endswith(".safetensors") for name in actual_files
            ):
                raise ValueError("Qwen3-Omni candidate snapshot is incomplete")
        verified[role] = snapshot
    return ModelPaths(**verified)


@dataclass(frozen=True)
class ModelPaths:
    asr: Path
    embeddings: Path
    text_audio: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "asr", require_resolved_local_directory(self.asr))
        object.__setattr__(self, "embeddings", require_resolved_local_directory(self.embeddings))
        object.__setattr__(self, "text_audio", require_resolved_local_directory(self.text_audio))


class _LazyModel(Generic[T]):
    def __init__(self, loader: Callable[[], T]) -> None:
        self._loader = loader
        self._value: T | None = None
        self._lock = Lock()

    def get(self) -> T:
        with self._lock:
            if self._value is None:
                self._value = self._loader()
            return self._value

    def release(self, *, clear: bool = True) -> bool:
        with self._lock:
            if self._value is None:
                return False
            release = getattr(self._value, "release", None)
            if callable(release):
                release()
            if clear:
                self._value = None
            return True


class TextModelRegistry:
    """Own exactly one verified constrained-text adapter at a time."""

    def __init__(
        self,
        path: Path,
        *,
        load_text: Callable[[Path], MlxConstrainedTextAdapter] = MlxConstrainedTextAdapter,
    ) -> None:
        self.path = require_resolved_local_directory(path)
        self._generator = _LazyModel(lambda: load_text(self.path))

    @property
    def generator(self) -> MlxConstrainedTextAdapter:
        return self._generator.get()

    def release(self) -> bool:
        return self._generator.release()


class ModelRegistry:
    """Own one instance of every adapter, including one shared text/audio model."""

    def __init__(
        self,
        paths: ModelPaths,
        *,
        load_asr: Callable[[Path], MlxAsrAdapter] = MlxAsrAdapter,
        load_embedder: Callable[[Path], MlxEmbeddingAdapter] = MlxEmbeddingAdapter,
        load_text_audio: Callable[[Path], MlxTextAudioAdapter] = MlxTextAudioAdapter,
        load_qwen3_omni: Callable[[Path], Qwen3OmniAdapter] = Qwen3OmniAdapter,
        candidate: str = "qwen2_audio",
    ) -> None:
        if candidate not in CANDIDATES:
            raise ValueError(f"unsupported model candidate: {candidate}")
        self.paths = paths
        self._asr = _LazyModel(lambda: load_asr(paths.asr))
        self._embedder = _LazyModel(lambda: load_embedder(paths.embeddings))
        text_audio_loader = (
            load_text_audio if candidate == "qwen2_audio" else load_qwen3_omni
        )
        self._text_audio = _LazyModel(lambda: text_audio_loader(paths.text_audio))

    @property
    def asr(self) -> MlxAsrAdapter:
        return self._asr.get()

    @property
    def embedder(self) -> MlxEmbeddingAdapter:
        return self._embedder.get()

    @property
    def text_generator(self) -> MlxTextAudioAdapter | Qwen3OmniAdapter:
        return self._text_audio.get()

    @property
    def audio_understanding(self) -> MlxTextAudioAdapter | Qwen3OmniAdapter:
        return self._text_audio.get()


@dataclass(frozen=True)
class TextModelIdentity:
    repository: str
    revision: str
    manifest_sha256: str


class EnglishPipelineModelRegistry:
    """Own the four verified MLX roles required by the English pipeline."""

    def __init__(
        self,
        *,
        asr_manifest: Path,
        embedding_manifest: Path,
        text_manifest: Path,
        yamnet_path: Path,
        load_asr: Callable[[Path], MlxAsrAdapter] = MlxAsrAdapter,
        load_embedder: Callable[[Path], MlxEmbeddingAdapter] = MlxEmbeddingAdapter,
        load_screener: Callable[[Path], MlxYamnetAdapter] = MlxYamnetAdapter,
        load_text: Callable[[Path], MlxConstrainedTextAdapter] = MlxConstrainedTextAdapter,
    ) -> None:
        # Resolve and hash-check manifests before any model allocation occurs.
        self.asr_path = load_verified_asr_path(asr_manifest)
        self.embedding_path = load_verified_embedding_path(embedding_manifest)
        manifest_bytes = Path(text_manifest).read_bytes()
        self.text_path = load_verified_text_path(text_manifest)
        if Path(text_manifest).read_bytes() != manifest_bytes:
            raise ValueError("text manifest changed during verification")
        self._text_model_identity = TextModelIdentity(
            **QWEN25_TEXT_SPEC, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest()
        )
        self.yamnet_path = require_resolved_local_directory(yamnet_path)
        self._transcriber = _LazyModel(lambda: load_asr(self.asr_path))
        self._embedder = _LazyModel(lambda: load_embedder(self.embedding_path))
        self._screener = _LazyModel(lambda: load_screener(self.yamnet_path))
        self._text_generator = _LazyModel(lambda: load_text(self.text_path))

    @property
    def transcriber(self) -> MlxAsrAdapter:
        return self._transcriber.get()

    @property
    def screener(self) -> MlxYamnetAdapter:
        return self._screener.get()

    @property
    def embedder(self) -> MlxEmbeddingAdapter:
        return self._embedder.get()

    @property
    def text_generator(self) -> MlxConstrainedTextAdapter:
        return self._text_generator.get()

    @property
    def text_model_identity(self) -> TextModelIdentity:
        return self._text_model_identity

    def release(self) -> None:
        # Keep stable adapter identities: TaxonomyIndex retains the embedder.
        # The text adapter is terminal after release, so its lazy holder must
        # discard it and construct a fresh adapter for the next job.
        self._text_generator.release()
        self._embedder.release(clear=False)
        self._screener.release()
        self._transcriber.release()
