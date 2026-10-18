"""Production word-timestamp/tail-filter replay of the fixed Task 9 ASR windows."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.transcribe.service import _rebase_segments, _top_language


RUNTIME_PACKAGES = ("mlx", "mlx-audio", "numpy", "pydantic")
MODE = "production_word_timestamps_tail_filter"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_identity(
    root: Path,
    manifest_path: Path,
    *,
    code_path: Path = Path(__file__),
    adapter_path: Path | None = None,
    version_fn: Callable[[str], str] = importlib.metadata.version,
    run_fn: Callable[..., object] = subprocess.run,
) -> dict[str, object]:
    """Collect reproducibility metadata without importing model runtimes."""
    manifest = json.loads(manifest_path.read_text())
    models = manifest.get("models", {})
    revisions = {
        name: {
            "repository": item.get("repository"),
            "revision": item.get("revision"),
        }
        for name, item in models.items()
        if isinstance(name, str) and isinstance(item, dict)
    }
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = version_fn(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    head = run_fn(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        run_fn(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    adapter_source = adapter_path or root / "engine/models/asr.py"
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "packages": packages,
        "git": {"head": head, "dirty": dirty},
        "code_sha256": _sha256(code_path),
        "adapter_sha256": _sha256(adapter_source),
        "manifest_sha256": _sha256(manifest_path),
        "model_revisions": revisions,
    }


def initialize_protocol(output: Path, identity: dict[str, object]) -> dict[str, object]:
    """Reserve a run and persist its identity before inference can begin."""
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": 1,
        "mode": MODE,
        "status": "initialized_before_inference",
        "created_utc": datetime.now(UTC).isoformat(),
        "runtime_identity": identity,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def build_summary(
    rows: list[dict[str, object]],
    memory: dict[str, object],
    source_protocol_path: Path,
    run_protocol_path: Path,
    identity: dict[str, object],
    cancellation: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "mode": MODE,
        "cases": rows,
        "cancellation": cancellation,
        "memory": memory,
        "protocol_sha256": _sha256(source_protocol_path),
        "run_protocol_sha256": _sha256(run_protocol_path),
        "runtime_identity": identity,
    }


def cancellation_probe(
    asr,
    audio: Path,
    duration: float,
    *,
    timer_factory=threading.Timer,
    monotonic=time.monotonic,
) -> dict[str, object]:
    token = CancellationToken()
    request_time: list[float] = []

    def request_cancel() -> None:
        request_time.append(monotonic())
        token.cancel()

    timer = timer_factory(1.0, request_cancel)
    timer.start()
    stopped = None
    outcome = "generation_completed_before_cancel"
    try:
        asr.transcribe_window(
            audio,
            TimeRange(start_seconds=0, end_seconds=duration),
            "en",
            token,
        )
    except JobCancelled:
        stopped = monotonic()
        outcome = "cancelled"
    finally:
        timer.cancel()
        timer.join()
    latency = None if not request_time or stopped is None else stopped - request_time[0]
    return {
        "outcome": outcome,
        "request_to_stop_seconds": latency,
        "target_seconds": 35.0,
        "passed": outcome == "cancelled" and latency is not None and latency <= 35.0,
        "attempt_count": 1,
    }


def probe_window(audio, duration, asr):
    token = CancellationToken()
    window = TimeRange(start_seconds=0, end_seconds=duration)
    result = {}
    stage = "language_detection"
    started = time.monotonic()
    try:
        probabilities = asr.detect_language_probabilities(audio, window, token)
        result["language_probabilities"] = probabilities
        language = _top_language(probabilities)
        result["language"] = language
        if language != "en":
            return {**result, "status": "ineligible", "seconds": time.monotonic() - started}
        stage = "transcription"
        segments = asr.transcribe_window(audio, window, language, token)
        result["segments"] = [segment.model_dump(mode="json") for segment in segments]
        stage = "timestamp_validation"
        _rebase_segments(segments, window)
        result["status"] = "passed"
    except JobCancelled:
        raise
    except Exception as exc:
        result.update(status="failed", failure_stage=stage,
                      exception_type=type(exc).__name__, reason=str(exc))
    result["seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = ROOT / "data/models/model-manifest.json"
    identity = runtime_identity(ROOT, manifest_path)
    # Reserve and identify a new output before model imports or inference.
    run_protocol = initialize_protocol(args.output, identity)
    from engine.models.asr import MlxAsrAdapter
    from engine.models.registry import load_verified_asr_path
    from engine.models.vocals import MlxYamnetAdapter
    from scripts.verify_text_classification import _memory, _swap_bytes

    source = ROOT / "benchmarks/runs.local/task9-supplemental-english-v1-2026-09-14"
    prior = json.loads((source / "fresh-transcription.json").read_text())
    source_protocol = json.loads((source / "protocol.json").read_text())
    asr = MlxAsrAdapter(load_verified_asr_path(manifest_path))
    screener = MlxYamnetAdapter(ROOT / "data/models/yamnet")
    before_swap = _swap_bytes()
    rows = []
    cancellation_audio = None
    cancellation_duration = None
    for case, interval in zip(
        prior["cases"], source_protocol["fresh_intervals"], strict=True
    ):
        if case["clip_id"] != interval["clip_id"]:
            raise ValueError("case identity mismatch")
        for index, old in enumerate(case["window_checkpoints"]):
            start = old["time_range"]["start_seconds"]
            end = old["time_range"]["end_seconds"]
            audio = source / f'{case["clip_id"]}-task7/asr-windows/asr-{int(start*1000):012d}-{int(end*1000):012d}.wav'
            with wave.open(str(audio)) as wav:
                duration = wav.getnframes() / wav.getframerate()
            screen = screener.screen(audio, CancellationToken())
            row = dict(clip_id=case["clip_id"], index=index, audio=str(audio),
                       audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                       duration=duration, prior_error=old["error_code"],
                       source_range=[interval["start_seconds"] + start, interval["start_seconds"] + end],
                       screen=screen.model_dump(mode="json"))
            row.update(probe_window(audio, duration, asr) if screen.admitted else {"status": "ineligible"})
            (args.output / f'{case["clip_id"]}-{index}.json').write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            if case["clip_id"] == "F04" and index == 1:
                cancellation_audio = audio
                cancellation_duration = duration
            print(case["clip_id"], index, row["status"], row.get("reason", ""), flush=True)
    if cancellation_audio is None or cancellation_duration is None:
        raise ValueError("ASR cancellation evidence window is missing")
    cancellation = cancellation_probe(
        asr,
        cancellation_audio,
        cancellation_duration,
    )
    summary = build_summary(
        rows,
        _memory(before_swap),
        source / "protocol.json",
        args.output / "protocol.json",
        run_protocol["runtime_identity"],
        cancellation,
    )
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 1 if any(row["status"] != "passed" for row in rows) or not cancellation["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
