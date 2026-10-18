"""Model-free helpers for durable, privacy-safe runtime measurements."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import Event, Thread
import time
from typing import Any, Callable

from engine.cancel import CancellationToken, JobCancelled


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    descriptor_owned = True
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor_owned = False
        with stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if descriptor_owned:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


class RuntimeCheckpoint:
    """Persist every completed runtime observation with atomic replacement."""

    def __init__(self, path: Path, *, candidate: str) -> None:
        self.path = path
        self.payload: dict[str, Any] = {
            "status": "running",
            "candidate": candidate,
            "completed_measurements": 0,
            "measurements": {},
        }

    def record(self, name: str, result: dict[str, Any]) -> None:
        measurements = self.payload["measurements"]
        if not isinstance(measurements, dict):
            raise ValueError("runtime measurement checkpoint is malformed")
        updated_measurements = dict(measurements)
        updated_measurements[name] = result
        updated = {
            **self.payload,
            "completed_measurements": len(updated_measurements),
            "measurements": updated_measurements,
        }
        atomic_write_json(self.path, updated)
        self.payload = updated

    def finish(self, result: dict[str, Any]) -> dict[str, Any]:
        updated = {**self.payload, **result}
        atomic_write_json(self.path, updated)
        self.payload = updated
        return updated


def print_safe_runtime_result(result: dict[str, Any]) -> None:
    safe = {
        name: result[name]
        for name in ("candidate", "completed_measurements", "status")
        if name in result
    }
    if "error_code" in result:
        safe["error_code"] = result["error_code"]
    print(json.dumps(safe, sort_keys=True), flush=True)


def measure_pre_cancelled(
    action: Callable[[CancellationToken, Callable[[], None]], Any],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    token = CancellationToken()
    token.cancel()
    call_started = False

    def started() -> None:
        nonlocal call_started
        call_started = True

    began = clock()
    try:
        action(token, started)
    except JobCancelled:
        acknowledged = clock()
        return {
            "status": "passed" if not call_started else "failed",
            "call_started": call_started,
            "acknowledgement_latency_seconds": acknowledged - began,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "call_started": call_started,
            "error_code": type(exc).__name__,
            "reason": str(exc),
        }
    return {
        "status": "failed",
        "call_started": call_started,
        "error_code": "CancellationIgnored",
    }


def measure_active_cancellation(
    action: Callable[[CancellationToken, Callable[[], None]], Any],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    token = CancellationToken()
    active = Event()
    activity_times: list[float] = []
    request_times: list[float] = []

    def started() -> None:
        if not activity_times:
            activity_times.append(clock())
        active.set()

    def request_cancellation() -> None:
        active.wait()
        if not activity_times:
            return
        request_times.append(clock())
        token.cancel()

    began = clock()
    canceller = Thread(target=request_cancellation, daemon=True)
    canceller.start()
    try:
        action(token, started)
    except JobCancelled:
        acknowledged = clock()
        canceller.join()
        if not request_times:
            return {
                "status": "failed",
                "call_started": bool(activity_times),
                "error_code": "CancellationNotRequested",
            }
        requested = request_times[0]
        return {
            "status": "passed" if activity_times else "failed",
            "call_started": bool(activity_times),
            "first_activity_latency_seconds": activity_times[0] - began,
            "acknowledgement_latency_seconds": acknowledged - requested,
        }
    except Exception as exc:
        active.set()
        canceller.join()
        return {
            "status": "failed",
            "call_started": bool(activity_times),
            "error_code": type(exc).__name__,
            "reason": str(exc),
        }
    active.set()
    canceller.join()
    return {
        "status": "failed",
        "call_started": bool(activity_times),
        "error_code": "CancellationIgnored",
    }
