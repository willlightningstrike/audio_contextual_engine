"""MLX-free deterministic retrieval over a prebuilt taxonomy vector index."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engine.cancel import CancellationToken
from engine.signals.protocol import Embedder
from engine.standards import TaxonomyRow


QUERY_PREFIX = (
    "Instruct: Given an audio transcript, retrieve the most relevant IAB content "
    "taxonomy categories describing its subject matter.\nQuery:\n"
)
CANONICALIZATION_VERSION = "iab-name-hierarchy-v1"
_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class TaxonomyCandidate:
    taxonomy_id: str
    name: str
    path: tuple[str, ...]
    similarity: float


def canonical_taxonomy_text(row: TaxonomyRow) -> str:
    """Return the exact document text used by the reviewed retrieval trial."""
    return "\n".join(
        (f"Category: {row.name}", f"Hierarchy: {' > '.join(row.path)}")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_vectors(
    vectors: np.ndarray, *, row_count: int, description: str
) -> np.ndarray:
    array = np.asarray(vectors)
    if (
        array.ndim != 2
        or array.shape[0] != row_count
        or array.shape[1] == 0
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"invalid {description} matrix")
    norms = np.linalg.norm(array.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"{description} vectors must be finite and non-zero")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _validate_rows(taxonomy: Sequence[TaxonomyRow]) -> tuple[TaxonomyRow, ...]:
    rows = tuple(taxonomy)
    ids = [row.id for row in rows]
    if not rows or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("taxonomy rows must have non-empty string IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("taxonomy row IDs must be unique")
    return rows


class TaxonomyIndex:
    def __init__(
        self,
        taxonomy: Sequence[TaxonomyRow],
        vectors: np.ndarray,
        embedder: Embedder,
    ) -> None:
        self._taxonomy = _validate_rows(taxonomy)
        self._vectors = _normalize_vectors(
            vectors, row_count=len(self._taxonomy), description="taxonomy vector"
        )
        self._embedder = embedder

    def shortlist(
        self,
        text: str,
        *,
        top_k: int = 15,
        token: CancellationToken,
    ) -> tuple[TaxonomyCandidate, ...]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= len(self._taxonomy)
        ):
            raise ValueError("top_k must be an integer within index bounds")
        token.raise_if_cancelled()
        encoded = self._embedder.encode([QUERY_PREFIX + text])
        token.raise_if_cancelled()
        query = np.asarray(encoded)
        if (
            query.ndim != 2
            or query.shape != (1, self._vectors.shape[1])
            or not np.issubdtype(query.dtype, np.number)
            or np.issubdtype(query.dtype, np.complexfloating)
            or not np.isfinite(query).all()
        ):
            raise ValueError("invalid query embedding matrix")
        norm = float(np.linalg.norm(query.astype(np.float64)))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("query embedding vector must be finite and non-zero")
        normalized_query = np.ascontiguousarray(query[0] / norm, dtype=np.float32)
        token.raise_if_cancelled()
        scores = self._vectors @ normalized_query
        token.raise_if_cancelled()
        if not np.isfinite(scores).all():
            raise ValueError("matrix search produced non-finite similarities")
        ids = [row.id for row in self._taxonomy]
        order = sorted(
            range(len(ids)), key=lambda index: (-float(scores[index]), ids[index])
        )
        return tuple(
            TaxonomyCandidate(
                taxonomy_id=self._taxonomy[index].id,
                name=self._taxonomy[index].name,
                path=self._taxonomy[index].path,
                similarity=float(scores[index]),
            )
            for index in order[:top_k]
        )


def _row_payload(rows: Sequence[TaxonomyRow]) -> list[dict[str, Any]]:
    return [
        {"taxonomy_id": row.id, "name": row.name, "path": list(row.path)}
        for row in rows
    ]


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_bytes_atomically(temporary: Path, destination: Path, payload: bytes) -> None:
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def build_taxonomy_index(
    output_dir: Path,
    *,
    taxonomy_path: Path,
    taxonomy: Sequence[TaxonomyRow],
    embedder: Embedder,
    embedding_revision: str,
    token: CancellationToken | None = None,
) -> dict[str, Any]:
    """Build verified content-addressed artifacts and publish the manifest last."""
    rows = _validate_rows(taxonomy)
    if not isinstance(embedding_revision, str) or not embedding_revision:
        raise ValueError("embedding revision must be non-empty")
    taxonomy_source = Path(taxonomy_path).resolve(strict=True)
    if not taxonomy_source.is_file():
        raise ValueError("taxonomy path must be a regular file")
    requested_root = Path(output_dir)
    if requested_root.is_symlink():
        raise ValueError("index output must not be a symlink")
    if requested_root.exists() and not requested_root.is_dir():
        raise ValueError("index output must be a directory")
    requested_root.mkdir(parents=True, exist_ok=True)
    if requested_root.is_symlink():
        raise ValueError("index output must not be a symlink")
    root = requested_root.resolve(strict=True)
    active_token = token or CancellationToken()
    active_token.raise_if_cancelled()
    encoded = embedder.encode([canonical_taxonomy_text(row) for row in rows])
    active_token.raise_if_cancelled()
    vectors = _normalize_vectors(
        encoded, row_count=len(rows), description="taxonomy vector"
    )

    vectors_temporary = root / "taxonomy-vectors.tmp"
    with vectors_temporary.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    vector_digest = _sha256(vectors_temporary)
    vectors_path = root / f"taxonomy-vectors-{vector_digest}.npy"
    os.replace(vectors_temporary, vectors_path)

    rows_bytes = (
        json.dumps(_row_payload(rows), separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()
    rows_digest = hashlib.sha256(rows_bytes).hexdigest()
    rows_path = root / f"taxonomy-rows-{rows_digest}.json"
    _write_bytes_atomically(root / "taxonomy-rows.tmp", rows_path, rows_bytes)

    vectors_record = _artifact_record(vectors_path, relative_to=root)
    vectors_record.update(dtype="float32", shape=list(vectors.shape))
    rows_record = _artifact_record(rows_path, relative_to=root)
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "embedding_revision": embedding_revision,
        "taxonomy_sha256": _sha256(taxonomy_source),
        "ordered_ids": [row.id for row in rows],
        "rows": rows_record,
        "vectors": vectors_record,
    }
    # Verify the temporary build using the same strict reader before publishing it.
    _load_artifacts(root, manifest, taxonomy_source, rows, embedding_revision)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_bytes_atomically(root / "manifest.json.tmp", root / _MANIFEST_NAME, manifest_bytes)
    return manifest


def _checked_artifact(root: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"invalid {label} artifact record")
    relative_value = record.get("path")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if (
        not isinstance(relative_value, str)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
        or not isinstance(expected_sha, str)
    ):
        raise ValueError(f"invalid {label} artifact record")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {label} artifact path")
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ValueError(f"invalid {label} artifact path")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
        raise ValueError(f"{label} artifact size or SHA mismatch")
    return path


def _load_artifacts(
    root: Path,
    manifest: Any,
    taxonomy_path: Path,
    taxonomy: Sequence[TaxonomyRow],
    expected_embedding_revision: str,
) -> np.ndarray:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("invalid taxonomy index manifest")
    if manifest.get("canonicalization_version") != CANONICALIZATION_VERSION:
        raise ValueError("taxonomy canonicalization version mismatch")
    if manifest.get("embedding_revision") != expected_embedding_revision:
        raise ValueError("embedding revision mismatch")
    if manifest.get("taxonomy_sha256") != _sha256(taxonomy_path):
        raise ValueError("taxonomy SHA mismatch")
    expected_ids = [row.id for row in taxonomy]
    if manifest.get("ordered_ids") != expected_ids:
        raise ValueError("taxonomy ordered IDs mismatch")
    rows_path = _checked_artifact(root, manifest.get("rows"), label="row")
    try:
        persisted_rows = json.loads(rows_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid row artifact") from exc
    if persisted_rows != _row_payload(taxonomy):
        raise ValueError("persisted taxonomy rows mismatch")
    vectors_record = manifest.get("vectors")
    vectors_path = _checked_artifact(root, vectors_record, label="vector")
    if vectors_record.get("dtype") != "float32":
        raise ValueError("vector artifact dtype mismatch")
    shape = vectors_record.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in shape)
        or shape[0] != len(taxonomy)
    ):
        raise ValueError("vector artifact shape mismatch")
    try:
        vectors = np.load(vectors_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("invalid vector artifact") from exc
    if vectors.dtype != np.float32 or list(vectors.shape) != shape:
        raise ValueError("vector artifact shape or dtype mismatch")
    return _normalize_vectors(
        vectors, row_count=len(taxonomy), description="taxonomy vector"
    )


def load_taxonomy_index(
    index_dir: Path,
    *,
    taxonomy_path: Path,
    taxonomy: Sequence[TaxonomyRow],
    embedder: Embedder,
    expected_embedding_revision: str,
) -> TaxonomyIndex:
    """Load a complete index only after manifest and artifact verification."""
    rows = _validate_rows(taxonomy)
    requested_root = Path(index_dir)
    if requested_root.is_symlink():
        raise ValueError("taxonomy index must not be a symlink")
    root = requested_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("taxonomy index must be a directory")
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("taxonomy index manifest is missing or unsafe")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid taxonomy index manifest") from exc
    taxonomy_source = Path(taxonomy_path).resolve(strict=True)
    vectors = _load_artifacts(
        root, manifest, taxonomy_source, rows, expected_embedding_revision
    )
    return TaxonomyIndex(rows, vectors, embedder)
