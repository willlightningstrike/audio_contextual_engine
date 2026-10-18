"""Verify the production MLX YAMNet port against frozen TensorFlow fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from engine.models.vocals import EXPECTED_ASSET_HASHES, MlxYamnetAdapter


ABSOLUTE_TOLERANCES = {"scores": 2e-4, "embeddings": 2e-3, "logmel": 2e-4}
RELATIVE_TOLERANCE = 2e-4
EXPECTED_FIXTURES = 23
RSS_CEILING_BYTES = 64 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _swap_bytes() -> int:
    """Return current system swap usage; evidence is invalid if unavailable."""
    try:
        text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        used = text.split("used =", 1)[1].split("M", 1)[0].strip()
        return int(float(used) * 1024 * 1024)
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("system swap measurement unavailable") from exc


def verify(model_dir: Path, fixtures_dir: Path) -> dict[str, Any]:
    import mlx.core as mx

    before_swap = _swap_bytes()
    adapter = MlxYamnetAdapter(model_dir)
    rows: list[dict[str, Any]] = []
    fixture_paths = sorted(fixtures_dir.glob("*.npz"))
    if len(fixture_paths) != EXPECTED_FIXTURES:
        raise ValueError(f"expected {EXPECTED_FIXTURES} parity fixtures, found {len(fixture_paths)}")

    for path in fixture_paths:
        with np.load(path, allow_pickle=False) as data:
            start = time.perf_counter()
            actual = adapter.model(mx.array(data["waveform"]))
            mx.eval(*actual)
            elapsed = time.perf_counter() - start
            errors: dict[str, float] = {}
            for name, value in zip(("scores", "embeddings", "logmel"), actual, strict=True):
                observed = np.asarray(value)
                expected = data[name]
                np.testing.assert_allclose(
                    observed,
                    expected,
                    atol=ABSOLUTE_TOLERANCES[name],
                    rtol=RELATIVE_TOLERANCE,
                    err_msg=f"{path.stem}: {name}",
                )
                errors[name] = float(np.max(np.abs(observed - expected)))
        rows.append(
            {
                "id": path.stem,
                "seconds": elapsed,
                "errors": errors,
                "fixture_sha256": _sha256(path),
            }
        )
        print(path.stem, errors, flush=True)

    forbidden_loaded = sorted(name for name in ("torch", "tensorflow") if name in sys.modules)
    if forbidden_loaded:
        raise RuntimeError(f"forbidden runtimes imported during parity: {forbidden_loaded}")
    after_swap = _swap_bytes()
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    source_path = Path(__file__).parents[1] / "engine" / "models" / "vocals.py"
    return {
        "passed": True,
        "fixtures": len(rows),
        "rows": rows,
        "atol": ABSOLUTE_TOLERANCES,
        "rtol": RELATIVE_TOLERANCE,
        "asset_sha256": dict(EXPECTED_ASSET_HASHES),
        "adapter_sha256": _sha256(source_path),
        "peak_mlx_bytes": int(mx.get_peak_memory()),
        # This verifier is Darwin-only in the approved Phase 1 environment;
        # ru_maxrss is bytes on macOS.
        "peak_rss_bytes": peak_rss,
        "rss_ceiling_bytes": RSS_CEILING_BYTES,
        "process_rss_ceiling_margin_bytes": RSS_CEILING_BYTES - peak_rss,
        "system_swap_before_bytes": before_swap,
        "system_swap_after_bytes": after_swap,
        "system_swap_delta_bytes": after_swap - before_swap,
        "no_tensorflow_or_torch_imported": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = verify(args.model_dir.resolve(strict=True), args.fixtures_dir.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"PASS {result['fixtures']} fixtures; wrote fresh evidence to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
