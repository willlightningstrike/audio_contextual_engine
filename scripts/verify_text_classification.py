"""Serialized offline Task 9 verification with durable prediction-first evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import resource
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import (
    ContainerKind,
    ContentContext,
    PreparedMedia,
    TimeRange,
    TranscriptSegment,
)
from engine.models import mlx_memory_statistics
from engine.models.asr import MlxAsrAdapter
from engine.models.embeddings import MlxEmbeddingAdapter
from engine.models.registry import (
    MODEL_SPECS,
    TextModelRegistry,
    load_verified_asr_path,
    load_verified_embedding_path,
    load_verified_text_path,
)
from engine.models.vocals import MlxYamnetAdapter
from engine.signals.classify import analysis_windows_from_run, classify_window
from engine.signals.taxonomy import (
    TaxonomyCandidate,
    load_taxonomy_index,
)
from engine.standards import load_taxonomy
from engine.transcribe.service import _rebase_segments, transcribe_media
from engine.transcribe.types import AnalysisWindow
from engine.ingest.probe import choose_audio_stream, probe_source


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT.parent.parent.parent
    / "contextual_engine"
    / "Stuck in an RV at 118°F! Parked on a Rooftop!.mp4"
)
DEFAULT_EVIDENCE_DIR = ROOT / "benchmarks/runs.local/task9-production-v1-2026-09-14"
DEFAULT_SUPPLEMENTAL_DIR = (
    ROOT / "benchmarks/runs.local/task9-supplemental-english-v1-2026-09-14"
)
MODEL_MANIFEST = ROOT / "data/models/model-manifest.json"
TEXT_MANIFEST = ROOT / "data/models/qwen25-manifest.json"
TAXONOMY_PATH = ROOT / "data/standards/content-taxonomy-3.0.tsv"
INDEX_DIR = ROOT / "data/models/taxonomy-index"
YAMNET_DIR = ROOT / "data/models/yamnet"
SAVED_SOURCES = (
    (
        "N",
        ROOT
        / "benchmarks/runs.local/prediction-review-v1/run-v1/classification-inputs.json",
    ),
    (
        "B04",
        ROOT
        / "benchmarks/runs.local/secondary-policy-v2/b04-regression-v1/inputs.json",
    ),
)
FRESH_INTERVALS = ((0.0, 90.0), (300.0, 390.0), (600.0, 690.0))
SUPPLEMENTAL_SELECTIONS = (
    (
        "F04",
        ROOT.parent.parent.parent / "contextual_engine/The Almost Perfect Travel Camera.mp4",
        540.0,
        630.0,
    ),
    (
        "F05",
        ROOT.parent.parent.parent
        / "contextual_engine/Optical Switches Are So Smooth Epomaker SK61 Review.mp4",
        600.0,
        690.0,
    ),
    (
        "F06",
        ROOT.parent.parent.parent
        / "contextual_engine/5 Things to Know Before Buying a Cartier Santos.mp4",
        420.0,
        510.0,
    ),
)
RSS_CEILING_BYTES = 64 * 1024**3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _swap_bytes() -> int:
    raw = subprocess.check_output(
        ["sysctl", "-n", "vm.swapusage"], text=True, stderr=subprocess.STDOUT
    )
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", raw)
    if match is None:
        raise RuntimeError("system swap measurement format is unsupported")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
    return int(float(match.group(1)) * scale[match.group(2)])


def _memory(before_swap: int) -> dict[str, Any]:
    after_swap = _swap_bytes()
    peak_rss = _peak_rss_bytes()
    return {
        "peak_process_rss_bytes": peak_rss,
        "process_rss_ceiling_bytes": RSS_CEILING_BYTES,
        "process_rss_ceiling_margin_bytes": RSS_CEILING_BYTES - peak_rss,
        "mlx_allocator": mlx_memory_statistics(),
        "system_swap_before_bytes": before_swap,
        "system_swap_after_bytes": after_swap,
        "system_swap_delta_bytes": after_swap - before_swap,
        "measurement_scope": {
            "rss": "current process peak",
            "mlx_allocator": "current process allocator endpoints",
            "swap": "system-wide endpoints; not process attribution",
        },
    }


def initialize_evidence(
    evidence_dir: Path,
    *,
    source: Path,
    intervals: Sequence[tuple[float, float]] = FRESH_INTERVALS,
) -> dict[str, Any]:
    _require_new(evidence_dir)
    source_path = Path(source).resolve(strict=True)
    probe = probe_source(source_path)
    normalized = tuple((float(start), float(end)) for start, end in intervals)
    if (
        len(normalized) != 3
        or any(start < 0 or end <= start or end > probe.duration_seconds for start, end in normalized)
        or any(first[1] > second[0] for first, second in zip(normalized, normalized[1:]))
    ):
        raise ValueError("fresh evidence requires three ordered disjoint in-range intervals")
    evidence_dir.mkdir(parents=True)
    protocol = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "selection_timing": "fixed before prediction",
        "source": {
            "path": str(source_path),
            "title_for_reporting_only": source_path.name,
            "duration_seconds": probe.duration_seconds,
            "sha256": _sha256(source_path),
        },
        "fresh_intervals": [
            {
                "clip_id": f"F{index:02d}",
                "start_seconds": start,
                "end_seconds": end,
            }
            for index, (start, end) in enumerate(normalized, start=1)
        ],
        "fresh_source_count": 1,
        "fresh_excerpt_count": 3,
        "saved_case_count": 5,
        "planned_classification_count": 8,
        "labels_collected_before_prediction": False,
        "prompt_input_policy": "transcript plus candidate ID/name/path only",
    }
    _write_json(evidence_dir / "protocol.json", protocol)
    return protocol


def initialize_supplemental_evidence(
    evidence_dir: Path,
    *,
    selections: Sequence[tuple[str, Path, float, float]] = SUPPLEMENTAL_SELECTIONS,
    probe_fn: Any = probe_source,
) -> dict[str, Any]:
    _require_new(evidence_dir)
    if len(selections) != 3:
        raise ValueError("supplemental evidence requires exactly three selections")
    intervals = []
    for clip_id, source, start, end in selections:
        source_path = Path(source).resolve(strict=True)
        probe = probe_fn(source_path)
        if start < 0 or end <= start or end > probe.duration_seconds:
            raise ValueError(f"supplemental interval is out of range: {clip_id}")
        intervals.append(
            {
                "clip_id": clip_id,
                "source_path": str(source_path),
                "source_title_for_reporting_only": source_path.name,
                "source_duration_seconds": probe.duration_seconds,
                "source_sha256": _sha256(source_path),
                "start_seconds": float(start),
                "end_seconds": float(end),
            }
        )
    if len({item["clip_id"] for item in intervals}) != 3:
        raise ValueError("supplemental clip IDs must be unique")
    evidence_dir.mkdir(parents=True)
    protocol = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "selection_timing": "fixed before supplemental prediction",
        "selection_context": (
            "previously used source videos, unused intervals selected after the "
            "single-source F01-F03 set failed English routing"
        ),
        "fresh_intervals": intervals,
        "fresh_source_count": 3,
        "fresh_excerpt_count": 3,
        "saved_case_count": 0,
        "planned_classification_count": 3,
        "include_saved_cases": False,
        "labels_collected_before_prediction": False,
        "prompt_input_policy": "transcript plus candidate ID/name/path only",
    }
    _write_json(evidence_dir / "protocol.json", protocol)
    return protocol


def _extract_interval(
    source: Path,
    destination: Path,
    *,
    stream_index: int,
    start: float,
    end: float,
) -> None:
    temporary = destination.with_suffix(".partial.wav")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-t",
        str(end - start),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(destination)


def prepare_fresh(evidence_dir: Path) -> dict[str, Any]:
    destination = evidence_dir / "fresh-transcription.json"
    _require_new(destination)
    protocol = json.loads((evidence_dir / "protocol.json").read_text())
    # Task 7 uses the existing verified ASR snapshot; no legacy audio model is loaded.
    asr_path = load_verified_asr_path(MODEL_MANIFEST)
    transcriber = MlxAsrAdapter(asr_path)
    screener = MlxYamnetAdapter(YAMNET_DIR.resolve(strict=True))
    before_swap = _swap_bytes()
    output: dict[str, Any] = {
        "status": "running",
        "protocol_sha256": _sha256(evidence_dir / "protocol.json"),
        "cases": [],
    }
    _write_json(destination, output)
    for interval in protocol["fresh_intervals"]:
        clip_id = interval["clip_id"]
        start = float(interval["start_seconds"])
        end = float(interval["end_seconds"])
        source_value = interval.get("source_path")
        if source_value is None:
            source_value = protocol["source"]["path"]
        source = Path(source_value).resolve(strict=True)
        probe = probe_source(source)
        selected = choose_audio_stream(probe, None)
        audio = evidence_dir / f"{clip_id}.wav"
        _require_new(audio)
        _extract_interval(
            source,
            audio,
            stream_index=selected.index,
            start=start,
            end=end,
        )
        media = PreparedMedia(
            source_path=source,
            audio_path=audio.resolve(strict=True),
            container_kind=ContainerKind.VIDEO,
            content_context_hint=ContentContext.VIDEO,
            selected_audio_stream=selected.index,
            available_audio_streams=probe.audio_streams,
            duration_seconds=end - start,
            source_metadata=probe.source_metadata,
        )
        checkpoints = []
        progress = []
        run = transcribe_media(
            media,
            transcriber=transcriber,
            screener=screener,
            work_dir=evidence_dir / f"{clip_id}-task7",
            token=CancellationToken(),
            checkpoint=lambda result: checkpoints.append(result.model_dump(mode="json")),
            on_progress=lambda completed, total: progress.append([completed, total]),
        )
        analysis = analysis_windows_from_run(run)
        output["cases"].append(
            {
                "clip_id": clip_id,
                "source_path_for_reporting_only": str(source),
                "source_range": {"start_seconds": start, "end_seconds": end},
                "audio_sha256": _sha256(audio),
                "transcription_status": run.status,
                "window_checkpoints": checkpoints,
                "progress": progress,
                "analysis_windows": [
                    {
                        "source_range": {
                            "start_seconds": start + window.time_range.start_seconds,
                            "end_seconds": start + window.time_range.end_seconds,
                        },
                        "transcript": " ".join(segment.text for segment in window.segments),
                        "segments": [segment.model_dump(mode="json") for segment in window.segments],
                    }
                    for window in analysis
                ],
            }
        )
        _write_json(destination, output)
    output["status"] = "complete"
    output["memory"] = _memory(before_swap)
    _write_json(destination, output)
    return output


def _load_saved_transcripts() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for prefix, source in SAVED_SOURCES:
        resolved = source.resolve(strict=True)
        payload = json.loads(resolved.read_text())
        source_cases = payload["cases"]
        sources.append(
            {
                "path": str(resolved),
                "sha256": _sha256(resolved),
                "labels_read": False,
            }
        )
        for index, case in enumerate(source_cases, start=1):
            clip_id = prefix if len(source_cases) == 1 else f"{prefix}{index:02d}"
            cases.append(
                {
                    "clip_id": clip_id,
                    "source_range": {"start_seconds": 0.0, "end_seconds": 90.0},
                    "transcript": case["transcript"],
                    "origin": "saved_transcript",
                }
            )
    return cases, sources


def _candidate_payload(candidate: TaxonomyCandidate) -> dict[str, Any]:
    return {
        "taxonomy_id": candidate.taxonomy_id,
        "name": candidate.name,
        "path": list(candidate.path),
        "similarity": candidate.similarity,
    }


def retrieve_cases(evidence_dir: Path) -> dict[str, Any]:
    destination = evidence_dir / "classification-inputs.json"
    _require_new(destination)
    fresh = json.loads((evidence_dir / "fresh-transcription.json").read_text())
    protocol = json.loads((evidence_dir / "protocol.json").read_text())
    if protocol.get("include_saved_cases", True):
        cases, saved_sources = _load_saved_transcripts()
    else:
        cases, saved_sources = [], []
    for fresh_case in fresh["cases"]:
        for index, window in enumerate(fresh_case["analysis_windows"], start=1):
            clip_id = fresh_case["clip_id"]
            if len(fresh_case["analysis_windows"]) > 1:
                clip_id = f"{clip_id}-{index}"
            cases.append(
                {
                    "clip_id": clip_id,
                    "source_range": window["source_range"],
                    "transcript": window["transcript"],
                    "origin": "fresh_task7",
                    "parent_excerpt": fresh_case["clip_id"],
                }
            )
    before_swap = _swap_bytes()
    embedding_path = load_verified_embedding_path(MODEL_MANIFEST)
    embedder = MlxEmbeddingAdapter(embedding_path)
    taxonomy = load_taxonomy()
    index = load_taxonomy_index(
        INDEX_DIR,
        taxonomy_path=TAXONOMY_PATH,
        taxonomy=taxonomy,
        embedder=embedder,
        expected_embedding_revision=MODEL_SPECS["embeddings"]["revision"],
    )
    output: dict[str, Any] = {
        "status": "running",
        "saved_sources": saved_sources,
        "fresh_transcription_sha256": _sha256(evidence_dir / "fresh-transcription.json"),
        "cases": [],
    }
    _write_json(destination, output)
    for case in cases:
        candidates = index.shortlist(
            case["transcript"], top_k=15, token=CancellationToken()
        )
        output["cases"].append(
            {**case, "candidates": [_candidate_payload(candidate) for candidate in candidates]}
        )
        # Every complete candidate list is durable before the text model process starts.
        _write_json(destination, output)
    output["status"] = "complete"
    output["memory"] = _memory(before_swap)
    _write_json(destination, output)
    return output


def _taxonomy_candidate(value: dict[str, Any]) -> TaxonomyCandidate:
    return TaxonomyCandidate(
        taxonomy_id=value["taxonomy_id"],
        name=value["name"],
        path=tuple(value["path"]),
        similarity=float(value["similarity"]),
    )


def classify_cases(
    cases: Sequence[dict[str, Any]],
    *,
    generator: Any,
    destination: Path,
    token: CancellationToken,
) -> dict[str, Any]:
    output: dict[str, Any] = {"status": "running", "cases": []}
    _write_json(destination, output)
    for case in cases:
        candidates = tuple(_taxonomy_candidate(value) for value in case["candidates"])
        source_range = TimeRange.model_validate(case["source_range"])
        window = AnalysisWindow(
            time_range=source_range,
            segments=(
                TranscriptSegment(
                    time_range=source_range,
                    text=case["transcript"],
                    confidence=1.0,
                    language="en",
                ),
            ),
        )
        record = {
            "clip_id": case["clip_id"],
            "origin": case.get("origin"),
            "source_range": case["source_range"],
            "transcript": case["transcript"],
            "candidates": case["candidates"],
            "attempts": [],
            "payload": None,
            "error_code": None,
        }
        output["cases"].append(record)
        # Reassert candidate durability in this process before generator invocation.
        _write_json(destination, output)

        def checkpoint(attempt: Any) -> None:
            record["attempts"].append(attempt.model_dump(mode="json"))
            _write_json(destination, output)

        result = classify_window(
            window,
            candidates,
            generator=generator,
            token=token,
            checkpoint=checkpoint,
        )
        record["payload"] = (
            None if result.payload is None else result.payload.model_dump(mode="json")
        )
        record["error_code"] = result.error_code
        record["category_citations"] = [
            item.model_dump(mode="json") for item in result.category_citations
        ]
        record["policy_version"] = result.policy_version
        record["outcome"] = (
            "semantic_abstention"
            if result.error_code == "INSUFFICIENT_TOPIC_EVIDENCE"
            else "technical_failure" if result.error_code is not None else "classified"
        )
        _write_json(destination, output)
        if record["outcome"] == "technical_failure":
            output["status"] = "failed"
            _write_json(destination, output)
            return output
    output["status"] = "passed"
    _write_json(destination, output)
    return output


def _cancellation_probe(generator: Any, case: dict[str, Any]) -> dict[str, Any]:
    candidates = tuple(_taxonomy_candidate(value) for value in case["candidates"])
    source_range = TimeRange.model_validate(case["source_range"])
    window = AnalysisWindow(
        time_range=source_range,
        segments=(TranscriptSegment(time_range=source_range, text=case["transcript"], confidence=1.0, language="en"),),
    )
    token = CancellationToken()
    request_time: list[float] = []
    attempts: list[dict[str, Any]] = []

    def request_cancel() -> None:
        request_time.append(time.monotonic())
        token.cancel()

    timer = threading.Timer(1.0, request_cancel)
    timer.start()
    stopped = None
    outcome = "generation_completed_before_cancel"
    try:
        classify_window(
            window,
            candidates,
            generator=generator,
            token=token,
            checkpoint=lambda attempt: attempts.append(
                attempt.model_dump(mode="json")
            ),
        )
    except JobCancelled:
        stopped = time.monotonic()
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
        "attempts": attempts,
        "attempt_stages": [attempt["stage"] for attempt in attempts],
        "attempt_error_codes": [attempt["error_code"] for attempt in attempts],
    }


def required_gate_verdict(
    result: dict[str, Any], *, require_cancellation: bool
) -> dict[str, Any]:
    """Combine required classification, cancellation, memory, and release evidence."""
    failures: list[str] = []
    if result.get("status") != "passed":
        failures.append("CLASSIFICATION_FAILED")
    cancellation = result.get("cancellation")
    if require_cancellation and (
        not isinstance(cancellation, dict) or cancellation.get("passed") is not True
    ):
        failures.append("CANCELLATION_TARGET_FAILED")
    memory = result.get("memory")
    peak_rss = memory.get("peak_process_rss_bytes") if isinstance(memory, dict) else None
    rss_ceiling = (
        memory.get("process_rss_ceiling_bytes") if isinstance(memory, dict) else None
    )
    if (
        not isinstance(peak_rss, int)
        or isinstance(peak_rss, bool)
        or not isinstance(rss_ceiling, int)
        or isinstance(rss_ceiling, bool)
        or peak_rss < 0
        or rss_ceiling <= 0
    ):
        failures.append("PROCESS_RSS_EVIDENCE_INVALID")
    elif peak_rss > rss_ceiling:
        failures.append("PROCESS_RSS_CEILING_EXCEEDED")
    if result.get("registry_release") is not True:
        failures.append("MODEL_RELEASE_FAILED")
    return {"passed": not failures, "failures": failures}


def run_classification(
    evidence_dir: Path, *, run_cancellation_probe: bool = True
) -> dict[str, Any]:
    destination = evidence_dir / "classification-results.json"
    _require_new(destination)
    inputs_path = evidence_dir / "classification-inputs.json"
    inputs = json.loads(inputs_path.read_text())
    before_swap = _swap_bytes()
    text_path = load_verified_text_path(TEXT_MANIFEST)
    registry = TextModelRegistry(text_path)
    output: dict[str, Any] = {"status": "running", "cases": []}
    try:
        output = classify_cases(
            inputs["cases"],
            generator=registry.generator,
            destination=destination,
            token=CancellationToken(),
        )
        if (
            run_cancellation_probe
            and output["status"] == "passed"
            and inputs["cases"]
        ):
            output["cancellation"] = _cancellation_probe(
                registry.generator, inputs["cases"][0]
            )
        output["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "outlines", "outlines-core", "transformers")
        }
        output["input_sha256"] = _sha256(inputs_path)
        output["text_manifest_sha256"] = _sha256(TEXT_MANIFEST)
    finally:
        output.setdefault("registry_release", registry.release())
    output["memory"] = _memory(before_swap)
    output["classification_status"] = output.get("status")
    output["required_gate"] = required_gate_verdict(
        output, require_cancellation=run_cancellation_probe
    )
    output["status"] = "passed" if output["required_gate"]["passed"] else "failed"
    _write_json(destination, output)
    return output


def diagnose_failed_window(
    evidence_dir: Path, *, clip_id: str = "F04", local_start_seconds: float = 0.0
) -> dict[str, Any]:
    """Capture one collapsed Task 7 MODEL_FAILURE without changing its policy."""
    suffix = f"{int(local_start_seconds):03d}"
    destination = evidence_dir / f"diagnosis-{clip_id}-{suffix}.json"
    _require_new(destination)
    audio = (
        evidence_dir
        / f"{clip_id}-task7/asr-windows"
        / f"asr-{int(local_start_seconds * 1000):012d}-{int((local_start_seconds + 30) * 1000):012d}.wav"
    ).resolve(strict=True)
    before_swap = _swap_bytes()
    transcriber = MlxAsrAdapter(load_verified_asr_path(MODEL_MANIFEST))
    screener = MlxYamnetAdapter(YAMNET_DIR.resolve(strict=True))
    window = TimeRange(start_seconds=0.0, end_seconds=30.0)
    output: dict[str, Any] = {
        "status": "running",
        "clip_id": clip_id,
        "local_window": window.model_dump(mode="json"),
        "audio_path": str(audio),
        "audio_sha256": _sha256(audio),
        "stages": {},
    }
    _write_json(destination, output)
    try:
        screen = screener.screen(audio, CancellationToken())
        output["stages"]["screen"] = {
            "status": "passed",
            "result": screen.model_dump(mode="json"),
        }
        _write_json(destination, output)
        probabilities = transcriber.detect_language_probabilities(
            audio, window, CancellationToken()
        )
        language = max(probabilities, key=probabilities.get)
        output["stages"]["language"] = {
            "status": "passed",
            "selected": language,
            "probabilities": probabilities,
        }
        _write_json(destination, output)
        raw_segments = transcriber.transcribe_window(
            audio, window, language, CancellationToken()
        )
        output["stages"]["adapter_transcribe"] = {
            "status": "passed",
            "segments": [segment.model_dump(mode="json") for segment in raw_segments],
        }
        _write_json(destination, output)
        rebased = _rebase_segments(raw_segments, window)
        output["stages"]["service_rebase"] = {
            "status": "passed",
            "segments": [segment.model_dump(mode="json") for segment in rebased],
        }
        output["status"] = "unexpected_success"
    except Exception as exc:
        stage = "screen"
        for candidate in ("service_rebase", "adapter_transcribe", "language"):
            if candidate in output["stages"]:
                stage = {
                    "service_rebase": "complete",
                    "adapter_transcribe": "service_rebase",
                    "language": "adapter_transcribe",
                }[candidate]
                break
        output["failure"] = {
            "stage": stage,
            "error_code": type(exc).__name__,
            "reason": str(exc),
        }
        output["status"] = "reproduced_failure"
    output["memory"] = _memory(before_swap)
    _write_json(destination, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "init",
            "supplemental-init",
            "prepare",
            "retrieve",
            "classify",
            "diagnose",
        ),
    )
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--skip-cancellation-probe", action="store_true")
    args = parser.parse_args()
    if args.mode == "init":
        result = initialize_evidence(args.evidence_dir, source=args.source)
    elif args.mode == "supplemental-init":
        evidence_dir = (
            DEFAULT_SUPPLEMENTAL_DIR
            if args.evidence_dir == DEFAULT_EVIDENCE_DIR
            else args.evidence_dir
        )
        result = initialize_supplemental_evidence(evidence_dir)
    elif args.mode == "prepare":
        result = prepare_fresh(args.evidence_dir)
    elif args.mode == "retrieve":
        result = retrieve_cases(args.evidence_dir)
    elif args.mode == "classify":
        result = run_classification(
            args.evidence_dir,
            run_cancellation_probe=not args.skip_cancellation_probe,
        )
    else:
        result = diagnose_failed_window(args.evidence_dir)
    print(json.dumps({"mode": args.mode, "status": result.get("status", "initialized")}))
    required_gate = result.get("required_gate")
    if result.get("status") == "failed":
        return 1
    if isinstance(required_gate, dict) and required_gate.get("passed") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
