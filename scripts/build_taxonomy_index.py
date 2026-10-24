"""Build and diagnose the pinned MLX taxonomy retrieval index offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import resource
import subprocess
from typing import Any, Sequence

from engine.cancel import CancellationToken
from engine.models import mlx_memory_statistics
from engine.models.embeddings import MlxEmbeddingAdapter
from engine.models.registry import MODEL_SPECS, load_verified_embedding_path
from engine.signals.taxonomy import build_taxonomy_index, load_taxonomy_index
from engine.standards import load_taxonomy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_MANIFEST = ROOT / "data" / "models" / "model-manifest.json"
DEFAULT_TAXONOMY = ROOT / "data" / "standards" / "content-taxonomy-3.1.tsv"
DEFAULT_INDEX_DIR = ROOT / "data" / "models" / "taxonomy-index"
DEFAULT_DIAGNOSTIC_SOURCES = (
    (
        "N",
        ROOT
        / "benchmarks"
        / "runs.local"
        / "prediction-review-v1"
        / "run-v1"
        / "classification-inputs.json",
    ),
    (
        "B04",
        ROOT
        / "benchmarks"
        / "runs.local"
        / "secondary-policy-v2"
        / "b04-regression-v1"
        / "inputs.json",
    ),
)
RSS_CEILING_BYTES = 64 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _swap_bytes() -> int:
    """Read the system-wide swap endpoint; never label it as process memory."""
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("system swap measurement unavailable") from exc
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", raw)
    if match is None:
        raise RuntimeError("system swap measurement format is unsupported")
    scales = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(match.group(1)) * scales[match.group(2)])


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return {
        "taxonomy_id": candidate.taxonomy_id,
        "name": candidate.name,
        "path": list(candidate.path),
        "similarity": candidate.similarity,
    }


def run_diagnostics(
    index: Any, cases: Sequence[dict[str, str]], *, top_k: int = 15
) -> list[dict[str, Any]]:
    """Replay transcript-only inputs twice and retain every candidate list."""
    output: list[dict[str, Any]] = []
    for case in cases:
        clip_id = case.get("clip_id")
        transcript = case.get("transcript")
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("diagnostic clip ID must be non-empty")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError(f"diagnostic transcript is empty for {clip_id}")
        first = index.shortlist(
            transcript, top_k=top_k, token=CancellationToken()
        )
        repeated = index.shortlist(
            transcript, top_k=top_k, token=CancellationToken()
        )
        first_payload = [_candidate_payload(candidate) for candidate in first]
        repeated_payload = [_candidate_payload(candidate) for candidate in repeated]
        output.append(
            {
                "clip_id": clip_id,
                "candidate_count": len(first_payload),
                "stable_repeat_ordering": first_payload == repeated_payload,
                "candidates": first_payload,
            }
        )
    return output


def _load_diagnostic_source(prefix: str, path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text())
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"diagnostic source has no cases: {path}")
    output: list[dict[str, str]] = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or not isinstance(case.get("transcript"), str):
            raise ValueError(f"diagnostic source has malformed transcript: {path}")
        clip_id = prefix if len(cases) == 1 else f"{prefix}{index:02d}"
        output.append({"clip_id": clip_id, "transcript": case["transcript"]})
    return output


def _model_inventory_evidence(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text())
    item = payload["models"]["embeddings"]
    files = item["files"]
    inventory_bytes = json.dumps(
        files, separators=(",", ":"), sort_keys=True
    ).encode()
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "repository": item["repository"],
        "revision": item["revision"],
        "local_path": item["local_path"],
        "file_count": len(files),
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "files": files,
    }


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_canonical_taxonomy_path(taxonomy_path: Path) -> Path:
    requested = Path(taxonomy_path)
    if requested.is_symlink():
        raise ValueError("taxonomy path must be the canonical vendored taxonomy, not a symlink")
    resolved = requested.resolve(strict=True)
    canonical = DEFAULT_TAXONOMY.resolve(strict=True)
    if resolved != canonical:
        raise ValueError("taxonomy path must be the canonical vendored taxonomy")
    return canonical


def build_and_diagnose(
    *,
    model_manifest: Path,
    taxonomy_path: Path,
    index_dir: Path,
    evidence_path: Path,
    diagnostic_sources: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    taxonomy_source = _require_canonical_taxonomy_path(taxonomy_path)
    before_swap = _swap_bytes()
    embedding_path = load_verified_embedding_path(model_manifest)
    embedder = MlxEmbeddingAdapter(embedding_path)
    taxonomy = load_taxonomy()
    revision = MODEL_SPECS["embeddings"]["revision"]
    manifest = build_taxonomy_index(
        index_dir,
        taxonomy_path=taxonomy_source,
        taxonomy=taxonomy,
        embedder=embedder,
        embedding_revision=revision,
        token=CancellationToken(),
    )
    index = load_taxonomy_index(
        index_dir,
        taxonomy_path=taxonomy_source,
        taxonomy=taxonomy,
        embedder=embedder,
        expected_embedding_revision=revision,
    )
    cases: list[dict[str, str]] = []
    source_evidence: list[dict[str, Any]] = []
    for prefix, source in diagnostic_sources:
        resolved = source.resolve(strict=True)
        source_cases = _load_diagnostic_source(prefix, resolved)
        cases.extend(source_cases)
        source_evidence.append(
            {
                "path": str(resolved),
                "sha256": _sha256(resolved),
                "clip_ids": [case["clip_id"] for case in source_cases],
                "input_policy": "transcript_only_no_labels",
            }
        )
    diagnostics = run_diagnostics(index, cases) if cases else []
    after_swap = _swap_bytes()
    peak_rss = _peak_rss_bytes()
    evidence = {
        "schema_version": 1,
        "scope": "offline taxonomy build and transcript-only diagnostic replay",
        "index": {
            "directory": str(index_dir.resolve()),
            "manifest_sha256": _sha256(index_dir.resolve() / "manifest.json"),
            "manifest": manifest,
        },
        "embedding_model": _model_inventory_evidence(model_manifest.resolve()),
        "runtime_provenance": embedder.runtime_evidence(),
        "memory": {
            "peak_rss_bytes": peak_rss,
            "rss_ceiling_bytes": RSS_CEILING_BYTES,
            "process_rss_ceiling_margin_bytes": RSS_CEILING_BYTES - peak_rss,
            "mlx_allocator": mlx_memory_statistics(),
            "system_swap_before_bytes": before_swap,
            "system_swap_after_bytes": after_swap,
            "system_swap_delta_bytes": after_swap - before_swap,
            "measurement_scope": {
                "rss": "current process peak",
                "mlx_allocator": "current process MLX allocator endpoints",
                "swap": "system-wide endpoint; not process attribution",
            },
        },
        "diagnostic_sources": source_evidence,
        "diagnostic_candidate_lists": diagnostics,
        "all_repeat_orderings_stable": all(
            item["stable_repeat_ordering"] for item in diagnostics
        ),
        "claims": {
            "classification_accuracy_measured": False,
            "gold_labels_read_or_rewritten": False,
            "host_free_memory_claim": False,
        },
    }
    _write_json_atomically(evidence_path, evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MODEL_MANIFEST)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_INDEX_DIR / "build-evidence.json",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="build the index without replaying preserved local transcript inputs",
    )
    args = parser.parse_args()
    sources = () if args.no_diagnostics else DEFAULT_DIAGNOSTIC_SOURCES
    evidence = build_and_diagnose(
        model_manifest=args.model_manifest,
        taxonomy_path=args.taxonomy,
        index_dir=args.index_dir,
        evidence_path=args.evidence,
        diagnostic_sources=sources,
    )
    print(
        json.dumps(
            {
                "index_manifest": str(args.index_dir.resolve() / "manifest.json"),
                "evidence": str(args.evidence.resolve()),
                "diagnostic_clips": len(evidence["diagnostic_candidate_lists"]),
                "stable_repeats": evidence["all_repeat_orderings_stable"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
