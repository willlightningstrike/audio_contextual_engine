"""Development diagnostic: bounded full-track YAMNet scores, never genre admission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Callable
from urllib.request import urlopen
import wave

import numpy as np

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.models.vocals import (
    EXPECTED_ASSET_HASHES, _default_load, _default_score, _sha256, _verify_snapshot,
)
from engine.signals.acoustic_probe import schedule_audio_windows
from scripts.prepare_yamnet import YAMNET_REVISION, YAMNET_SOURCE_PROVENANCE


ROOT = Path(__file__).resolve().parents[1]
YAMNET_DIR = ROOT / "data/models/yamnet"
CLASS_MAP_DIR = ROOT / "benchmarks/runs.local/yamnet-class-map"
CLASS_MAP_URL = (
    f"https://raw.githubusercontent.com/tensorflow/models/{YAMNET_REVISION}"
    "/research/audioset/yamnet/yamnet_class_map.csv"
)
SAMPLE_RATE = 16_000


def _snapshot_path() -> Path:
    return YAMNET_DIR.resolve(strict=True)


def class_map_metadata() -> dict:
    """Fetch only pinned upstream metadata; unavailable names stay unverified."""
    metadata = {"url": CLASS_MAP_URL, "revision": YAMNET_REVISION,
                "license": "Apache-2.0", "verified": False, "names": None,
                "sha256": None}
    try:
        with urlopen(CLASS_MAP_URL, timeout=15) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("class map exceeds metadata size limit")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
        if len(rows) != 521 or any(
            row.get("index") != str(index) or not row.get("mid") or not row.get("display_name")
            for index, row in enumerate(rows)
        ):
            raise ValueError("class map must contain all 521 ordered native classes")
        digest = hashlib.sha256(raw).hexdigest()
        CLASS_MAP_DIR.mkdir(parents=True, exist_ok=True)
        target = CLASS_MAP_DIR / f"{YAMNET_REVISION}-{digest}.csv"
        try:
            with target.open("xb") as stream:
                stream.write(raw)
        except FileExistsError:
            if _sha256(target) != digest:
                raise ValueError("cached class map SHA-256 mismatch")
        metadata.update(verified=True, sha256=digest, names=[row["display_name"] for row in rows],
                        local_path=str(target))
    except Exception as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def split_window(window: TimeRange, seconds: float = 5.0) -> tuple[TimeRange, ...]:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("unit duration must be finite and positive")
    return tuple(TimeRange(start_seconds=window.start_seconds + i * seconds,
                           end_seconds=min(window.end_seconds, window.start_seconds + (i + 1) * seconds))
                 for i in range(math.ceil((window.end_seconds - window.start_seconds) / seconds)))


def read_pcm_samples(stream: wave.Wave_read, unit: TimeRange, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    start, end = round(unit.start_seconds * sample_rate), round(unit.end_seconds * sample_rate)
    stream.setpos(start)
    raw = stream.readframes(end - start)
    if len(raw) != (end - start) * 2:
        raise ValueError("short PCM read")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def checked_scores(value: object) -> np.ndarray:
    scores = np.asarray(value)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] != 521:
        raise ValueError("YAMNet must return 521 class scores")
    if scores.dtype.kind not in "fiu" or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("YAMNet returned invalid scores")
    return scores


def _json_safe(value: object) -> object:
    """Retain malformed numeric diagnostics without emitting nonstandard JSON."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def probe_yamnet(audio: Path, *, out: Path, token: CancellationToken,
                 load_fn: Callable = _default_load, score_fn: Callable = _default_score) -> dict:
    """Reserve a report before inference and preserve diagnostics on failure."""
    path = Path(audio).resolve(strict=True)
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels() != 1 or stream.getsampwidth() != 2
                or stream.getframerate() != SAMPLE_RATE or stream.getcomptype() != "NONE"):
            raise ValueError("probe requires mono 16 kHz PCM16 WAV")
        frames = stream.getnframes()
        if not frames:
            raise ValueError("probe requires nonempty WAV")
        units = [unit for window in schedule_audio_windows(frames / SAMPLE_RATE)
                 for unit in split_window(window)]
        report = {"schema_version": 1, "kind": "yamnet_full_scores_diagnostic",
                  "status": "running", "audio": str(path), "sample_rate": SAMPLE_RATE,
                  "sample_count": frames, "duration_seconds": frames / SAMPLE_RATE,
                  "source_provenance": dict(YAMNET_SOURCE_PROVENANCE),
                  "asset_hashes": {}, "class_map": None,
                  "class_indices": list(range(521)), "records": [], "errors": [],
                  "progress": {"completed_units": 0, "expected_units": len(units)}}
        # Keep this handle: no later reopen can follow a substituted output path.
        with Path(out).open("x", encoding="utf-8") as output:
            def persist() -> None:
                encoded = json.dumps(report, allow_nan=False, indent=2) + "\n"
                output.seek(0)
                output.write(encoded)
                output.truncate()
                output.flush()

            current = None
            raw_scores = None
            persist()
            try:
                token.raise_if_cancelled()
                root = _verify_snapshot(_snapshot_path(), EXPECTED_ASSET_HASHES)
                report["asset_hashes"] = {name: _sha256(root / name) for name in EXPECTED_ASSET_HASHES}
                report["license"] = {"name": "Apache-2.0", "sha256": _sha256(root / "LICENSE")}
                report["class_map"] = class_map_metadata()
                token.raise_if_cancelled()
                model = load_fn(root)
                for unit in units:
                    current = {"start_seconds": unit.start_seconds, "end_seconds": unit.end_seconds,
                               "start_sample": round(unit.start_seconds * SAMPLE_RATE),
                               "end_sample": round(unit.end_seconds * SAMPLE_RATE)}
                    raw_scores = None
                    token.raise_if_cancelled()
                    waveform = read_pcm_samples(stream, unit)
                    token.raise_if_cancelled()
                    raw_scores = score_fn(model, waveform)
                    scores = checked_scores(raw_scores)
                    report["records"].append({**current, "scores": scores.tolist()})
                    report["progress"]["completed_units"] += 1
                    # Returned diagnostic evidence is retained even if inference cancelled.
                    persist()
                    token.raise_if_cancelled()
                if len(report["records"]) != len(units):
                    raise ValueError("incomplete YAMNet diagnostic coverage")
                token.raise_if_cancelled()
                report["status"] = "complete"
                token.run_if_not_cancelled(persist)
                return report
            except Exception as exc:
                report["status"] = "cancelled" if isinstance(exc, JobCancelled) else "technical_error"
                report["errors"].append({"type": type(exc).__name__, "message": str(exc),
                                         "interval": current, "raw_scores": _json_safe(raw_scores)})
                persist()
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    probe_yamnet(args.audio, out=args.out, token=CancellationToken())


if __name__ == "__main__":
    main()
