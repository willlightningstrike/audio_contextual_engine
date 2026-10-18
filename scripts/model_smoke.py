"""Run the real-model offline, residency, bounded-work, and memory smoke gate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from engine.cancel import CancellationToken
from engine.domain import TimeRange
from engine.models import mlx_memory_statistics
from engine.models.registry import (
    CANDIDATES,
    ModelRegistry,
    candidate_manifest_path,
    load_verified_model_paths,
    model_specs,
)
from engine.models.response import canonical_json_container

try:
    from scripts.benchmark_contracts import (
        ACOUSTIC_JSON_PROMPT,
        acoustic_description,
        parse_acoustic,
        parse_text_rerank,
        text_rerank_prompt,
    )
    from scripts.benchmark_runtime import (
        RuntimeCheckpoint,
        measure_active_cancellation,
        measure_pre_cancelled,
        print_safe_runtime_result,
    )
    from scripts.accuracy_harness import MlxObservationRunner
except ModuleNotFoundError:  # Direct ``python scripts/model_smoke.py`` execution.
    from benchmark_contracts import (  # type: ignore[no-redef]
        ACOUSTIC_JSON_PROMPT,
        acoustic_description,
        parse_acoustic,
        parse_text_rerank,
        text_rerank_prompt,
    )
    from benchmark_runtime import (  # type: ignore[no-redef]
        RuntimeCheckpoint,
        measure_active_cancellation,
        measure_pre_cancelled,
        print_safe_runtime_result,
    )
    from accuracy_harness import MlxObservationRunner  # type: ignore[no-redef]


def error_result(error: BaseException) -> dict[str, str]:
    """Only unavailable local artifacts block the gate; execution defects fail it."""
    return {
        "status": "blocked" if isinstance(error, FileNotFoundError) else "failed",
        "error_code": type(error).__name__,
        "reason": str(error),
    }


def _physical_memory_bytes() -> int:
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())


def _swap_bytes() -> int | None:
    try:
        text = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        used = text.split("used =", 1)[1].split("M", 1)[0].strip()
        return int(float(used) * 1024 * 1024)
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return None


def _calls_per_second(latencies: list[float]) -> list[float]:
    if any(value <= 0 for value in latencies):
        raise ValueError("latencies must be positive")
    return [1.0 / value for value in latencies]


def _output_evidence(text: str, elapsed: float) -> dict[str, Any]:
    encoded = text.encode()
    return {
        "status": "passed",
        "latency_seconds": elapsed,
        "output_utf8_bytes": len(encoded),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _timed_model_call(
    name: str,
    action: Callable[[], Any],
    *,
    on_measurement: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Any, float]:
    started = time.monotonic()
    try:
        value = action()
    except Exception as exc:
        if on_measurement is not None:
            on_measurement(
                name,
                {
                    "status": "failed",
                    "error_code": type(exc).__name__,
                    "reason": str(exc),
                },
            )
        raise
    elapsed = time.monotonic() - started
    if on_measurement is not None:
        on_measurement(name, {"status": "passed", "latency_seconds": elapsed})
    return value, elapsed


def exercise_realistic_metadata(
    text_audio: Any,
    audio: Path,
    window: TimeRange,
    *,
    transcript: str,
    retrieve: Callable[
        [list[tuple[str, str]], CancellationToken], list[dict[str, Any]]
    ],
    on_measurement: Callable[[str, dict[str, Any]], None] | None = None,
    on_text_prompt: Callable[[str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Exercise the same bounded schema workload used by the accuracy harness."""
    def record(name: str, result: dict[str, Any]) -> dict[str, Any]:
        if on_measurement is not None:
            on_measurement(name, result)
        return result

    def failure(
        error: Exception, *, kind: str, evidence: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            **({} if evidence is None else evidence),
            "status": "failed",
            "error_code": type(error).__name__,
            "error_kind": kind,
            "reason": str(error),
        }

    def unattempted(dependency: str) -> dict[str, Any]:
        return {
            "status": "unattempted",
            "reason_code": "prior_execution_failure",
            "dependency": dependency,
        }

    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    try:
        audio_output = text_audio.generate_audio(
            audio,
            window,
            ACOUSTIC_JSON_PROMPT,
            max_tokens=160,
            token=CancellationToken(),
        )
        audio_elapsed = time.monotonic() - started
    except Exception as exc:
        results["audio"] = record(
            "realistic_audio", failure(exc, kind="execution")
        )
        results["retrieval"] = record(
            "realistic_retrieval", unattempted("realistic_audio")
        )
        results["text"] = record(
            "realistic_text", unattempted("realistic_retrieval")
        )
        return results

    audio_evidence = _output_evidence(audio_output, audio_elapsed)
    acoustic: dict[str, Any] | None = None
    try:
        acoustic = parse_acoustic(audio_output)
    except ValueError as exc:
        results["audio"] = record(
            "realistic_audio",
            failure(exc, kind="schema", evidence=audio_evidence),
        )
    else:
        results["audio"] = record("realistic_audio", audio_evidence)

    audio_texts: list[tuple[str, str]] = []
    normalized_transcript = transcript.strip()
    if normalized_transcript:
        audio_texts.append(("transcript", normalized_transcript))
    if acoustic is not None and acoustic["is_music"]:
        audio_texts.append(("acoustic", acoustic_description(acoustic)))
    try:
        shortlist = retrieve(audio_texts, CancellationToken())
        if len(shortlist) != 15:
            raise ValueError("realistic retrieval must return exactly 15 candidates")
    except Exception as exc:
        results["retrieval"] = record(
            "realistic_retrieval", failure(exc, kind="execution")
        )
        results["text"] = record(
            "realistic_text", unattempted("realistic_retrieval")
        )
        return results
    results["retrieval"] = record(
        "realistic_retrieval",
        {
            "status": "passed",
            "candidate_count": len(shortlist),
            "input_sources": [source for source, _text in audio_texts],
        },
    )

    prompt = text_rerank_prompt([text for _source, text in audio_texts], shortlist)
    if on_text_prompt is not None:
        on_text_prompt(prompt)
    started = time.monotonic()
    try:
        text_output = text_audio.generate_text(
            prompt, max_tokens=320, token=CancellationToken()
        )
        text_elapsed = time.monotonic() - started
    except Exception as exc:
        results["text"] = record(
            "realistic_text", failure(exc, kind="execution")
        )
        return results
    text_evidence = {
        **_output_evidence(text_output, text_elapsed),
        "candidate_count": len(shortlist),
        "transcript_utf8_bytes": len(normalized_transcript.encode()),
        "transcript_sha256": hashlib.sha256(normalized_transcript.encode()).hexdigest(),
    }
    try:
        parse_text_rerank(text_output, shortlist)
    except ValueError as exc:
        results["text"] = record(
            "realistic_text",
            failure(exc, kind="schema", evidence=text_evidence),
        )
    else:
        results["text"] = record("realistic_text", text_evidence)
    return results


@contextmanager
def _instrument_call(
    owner: Any,
    attribute: str,
    started: Callable[[], None],
    *,
    streaming: bool = False,
) -> Iterator[None]:
    original = getattr(owner, attribute)
    if streaming:
        def wrapped(*args: Any, **kwargs: Any) -> Iterator[Any]:
            stream = original(*args, **kwargs)
            try:
                for chunk in stream:
                    started()
                    yield chunk
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
    else:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started()
            return original(*args, **kwargs)
    setattr(owner, attribute, wrapped)
    try:
        yield
    finally:
        setattr(owner, attribute, original)


def _asr_cancellation_action(
    asr: Any, audio: Path, window: TimeRange, language: str
) -> Callable[[CancellationToken, Callable[[], None]], Any]:
    def action(token: CancellationToken, started: Callable[[], None]) -> Any:
        with _instrument_call(asr.model, "generate", started):
            return asr.transcribe_window(audio, window, language, token)

    return action


def _text_audio_cancellation_action(
    text_audio: Any,
    candidate: str,
    mode: str,
    audio: Path,
    window: TimeRange,
    text_prompt: str,
) -> Callable[[CancellationToken, Callable[[], None]], Any]:
    if mode not in {"text", "audio"}:
        raise ValueError("cancellation mode must be text or audio")

    def action(token: CancellationToken, started: Callable[[], None]) -> Any:
        if candidate == "qwen3_omni":
            owner, attribute, streaming = text_audio, "_stream_generate_fn", True
        elif mode == "text":
            owner, attribute, streaming = text_audio, "_text_generate_fn", False
        else:
            owner, attribute, streaming = text_audio.model, "generate", False
        with _instrument_call(owner, attribute, started, streaming=streaming):
            if mode == "text":
                return text_audio.generate_text(
                    text_prompt,
                    max_tokens=320,
                    token=token,
                )
            return text_audio.generate_audio(
                audio,
                window,
                ACOUSTIC_JSON_PROMPT,
                max_tokens=160,
                token=token,
            )

    return action


def measure_cancellation_suite(
    actions: dict[
        str, Callable[[CancellationToken, Callable[[], None]], Any]
    ],
    *,
    unavailable_modes: dict[str, str] | None = None,
    pre_measure: Callable[[Callable[..., Any]], dict[str, Any]] = measure_pre_cancelled,
    active_measure: Callable[[Callable[..., Any]], dict[str, Any]] = measure_active_cancellation,
    on_measurement: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Measure safe modes and explicitly retain why dependent modes were skipped."""
    unavailable = {} if unavailable_modes is None else unavailable_modes
    results: dict[str, dict[str, dict[str, Any]]] = {
        "pre_cancelled": {},
        "active": {},
    }
    for phase, measure in (
        ("pre_cancelled", pre_measure),
        ("active", active_measure),
    ):
        for mode, action in actions.items():
            if mode in unavailable:
                result = {
                    "status": "unattempted",
                    "reason_code": unavailable[mode],
                }
            else:
                result = measure(action)
            results[phase][mode] = result
            if on_measurement is not None:
                prefix = "pre_cancelled" if phase == "pre_cancelled" else "active_cancellation"
                on_measurement(f"{prefix}_{mode}", result)
    return results


def sample_speech_audio(manifest_path: Path) -> tuple[Path, str]:
    labels = json.loads(manifest_path.read_text())
    samples = labels.get("samples") if isinstance(labels, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("labelled manifest must include at least one sample")
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        language = sample.get("language")
        if language not in {"en", "zh", "ja"} or "speech" not in sample.get("content_types", []):
            continue
        if not isinstance(sample.get("audio_path"), str):
            raise ValueError("speech-labelled sample must include audio_path")
        return Path(sample["audio_path"]).expanduser().resolve(strict=True), language
    raise ValueError("labelled manifest must include an English, Chinese, or Japanese speech sample")


def _block_network() -> None:
    original_connect = socket.socket.connect

    def blocked_connect(self: socket.socket, address: object) -> object:
        raise RuntimeError(f"network access is forbidden during model smoke: {address!r}")

    socket.socket.connect = blocked_connect  # type: ignore[assignment]
    # Retain a reference so static analyzers cannot treat the patch as unused.
    assert original_connect is not socket.socket.connect


def _expect_isolated_json(text: str, request: str) -> dict[str, str]:
    try:
        decoded = json.loads(canonical_json_container(text))
    except ValueError as exc:
        raise RuntimeError("text/audio model isolation response was not JSON") from exc
    if decoded != {"request": request}:
        raise RuntimeError("text/audio model retained prompt state or returned an unexpected isolation response")
    return decoded


def exercise_prompt_isolation(
    text_audio: Any,
    audio: Path,
    window: TimeRange,
    *,
    timings: dict[str, list[float]] | None = None,
    on_measurement: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Alternate text/audio calls so shared-model state cannot hide between modes."""
    results: dict[str, list[dict[str, str]]] = {"text": [], "audio": []}
    for mode, request in (("text", "text-first"), ("audio", "audio-first"), ("text", "text-second"), ("audio", "audio-second")):
        prompt = f'Reply with exactly {{"request":"{request}"}}'
        started = time.monotonic()
        name = f"isolation_{mode}_{len(results[mode]) + 1}"
        try:
            if mode == "text":
                output = text_audio.generate_text(prompt, max_tokens=24, token=CancellationToken())
            else:
                output = text_audio.generate_audio(audio, window, prompt, max_tokens=24, token=CancellationToken())
            elapsed = time.monotonic() - started
            decoded = _expect_isolated_json(output, request)
        except Exception as exc:
            if on_measurement is not None:
                on_measurement(
                    name,
                    {
                        "status": "failed",
                        "error_code": type(exc).__name__,
                        "reason": str(exc),
                    },
                )
            raise
        if timings is not None:
            timings[mode].append(elapsed)
        if on_measurement is not None:
            on_measurement(name, {"status": "passed", "latency_seconds": elapsed})
        results[mode].append(decoded)
    return results


def run(
    models_dir: Path,
    labels: Path,
    candidate: str = "qwen2_audio",
    acceptance: Path = Path("benchmarks/acceptance.json"),
    checkpoint: RuntimeCheckpoint | None = None,
) -> dict[str, Any]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("real model smoke requires Darwin arm64")
    memory = _physical_memory_bytes()
    if memory < 16 * 1024**3:
        raise RuntimeError("real model smoke requires at least 16 GB physical memory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _block_network()
    manifest_path = candidate_manifest_path(models_dir.resolve(), candidate)
    paths = (
        load_verified_model_paths(manifest_path)
        if candidate == "qwen2_audio"
        else load_verified_model_paths(manifest_path, candidate=candidate)
    )
    audio, language = sample_speech_audio(labels.resolve())
    before_swap = _swap_bytes()
    start = time.monotonic()
    registry = (
        ModelRegistry(paths)
        if candidate == "qwen2_audio"
        else ModelRegistry(paths, candidate=candidate)
    )
    asr = registry.asr
    embedder = registry.embedder
    text_audio = registry.text_generator
    object_ids = {"asr": id(asr.model), "embedder": id(embedder.model), "text_audio": id(text_audio.model)}
    window = TimeRange(start_seconds=0, end_seconds=30)
    language_latencies: list[float] = []
    language_probabilities: list[dict[str, float]] = []
    for _ in range(2):
        result, latency = _timed_model_call(
            f"language_{len(language_latencies) + 1}",
            lambda: asr.detect_language_probabilities(
                audio, window, CancellationToken()
            ),
            on_measurement=None if checkpoint is None else checkpoint.record,
        )
        language_probabilities.append(result)
        language_latencies.append(latency)
    asr_latencies: list[float] = []
    segments: list[Any] = []
    for _ in range(2):
        segments, latency = _timed_model_call(
            f"asr_{len(asr_latencies) + 1}",
            lambda: asr.transcribe_window(
                audio, window, language, CancellationToken()
            ),
            on_measurement=None if checkpoint is None else checkpoint.record,
        )
        asr_latencies.append(latency)
    transcript = " ".join(segment.text for segment in segments).strip()
    embedding_latencies: list[float] = []
    for _ in range(2):
        _, latency = _timed_model_call(
            f"embeddings_{len(embedding_latencies) + 1}",
            lambda: embedder.encode(
                ["privacy-preserving advertising", "multilingual audio context"]
            ),
            on_measurement=None if checkpoint is None else checkpoint.record,
        )
        embedding_latencies.append(latency)
    text_audio_latencies: dict[str, list[float]] = {"text": [], "audio": []}
    prompt_outputs = exercise_prompt_isolation(
        text_audio,
        audio,
        window,
        timings=text_audio_latencies,
        on_measurement=None if checkpoint is None else checkpoint.record,
    )
    realistic_text_prompts: list[str] = []
    retriever = MlxObservationRunner(registry)
    realistic = exercise_realistic_metadata(
        text_audio,
        audio,
        window,
        transcript=transcript,
        retrieve=retriever.retrieve,
        on_measurement=None if checkpoint is None else checkpoint.record,
        on_text_prompt=realistic_text_prompts.append,
    )
    provenance = {"asr": asr.runtime_evidence(), "embeddings": embedder.runtime_evidence(), "text_audio": text_audio.runtime_evidence()}
    type_checks = {
        "asr_parameters": provenance["asr"]["parameter_leaves_are_mlx"],
        "asr_language_tensor": provenance["asr"]["tensors_are_mlx"],
        "embedding_parameters": provenance["embeddings"]["parameter_leaves_are_mlx"],
        "embedding_tensor": provenance["embeddings"]["tensors_are_mlx"],
        "text_audio_parameters": provenance["text_audio"]["parameter_leaves_are_mlx"],
        "text_audio_tensors": provenance["text_audio"]["tensors_are_mlx"],
    }
    if not all(type_checks.values()):
        raise RuntimeError("real-model MLX provenance assertion failed")
    cancellation_actions = {
        "asr": _asr_cancellation_action(asr, audio, window, language),
        "text": _text_audio_cancellation_action(
            text_audio,
            candidate,
            "text",
            audio,
            window,
            realistic_text_prompts[0] if realistic_text_prompts else "",
        ),
        "audio": _text_audio_cancellation_action(
            text_audio,
            candidate,
            "audio",
            audio,
            window,
            realistic_text_prompts[0] if realistic_text_prompts else "",
        ),
    }
    unavailable_modes: dict[str, str] = {}
    if any(
        realistic[mode].get("error_kind") == "execution"
        for mode in ("audio", "text")
    ):
        unavailable_modes.update(
            {
                "text": "prior_execution_failure",
                "audio": "prior_execution_failure",
            }
        )
    elif not realistic_text_prompts:
        unavailable_modes["text"] = "realistic_retrieval_unavailable"
    cancellation = measure_cancellation_suite(
        cancellation_actions,
        unavailable_modes=unavailable_modes,
        on_measurement=None if checkpoint is None else checkpoint.record,
    )
    after_swap = _swap_bytes()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mlx_memory = mlx_memory_statistics()
    thresholds = json.loads(acceptance.read_text())["thresholds"]
    peak_limit = int(thresholds["maximum_peak_rss_bytes_on_current_64gb_host"])
    cancellation_limit = float(thresholds["maximum_cancellation_latency_seconds"])
    # macOS reports ru_maxrss in bytes; Linux uses KiB but Linux is rejected above.
    memory_checks = {
        "peak_rss_bytes": peak_rss,
        "swap_delta_bytes": None if before_swap is None or after_swap is None else after_swap - before_swap,
        "resident_model_count_stable": object_ids == {"asr": id(registry.asr.model), "embedder": id(registry.embedder.model), "text_audio": id(registry.audio_understanding.model)},
        "mlx": mlx_memory,
    }
    runtime_thresholds = {
        "peak_rss_at_most_bytes": peak_limit,
        "cancellation_at_most_seconds": cancellation_limit,
    }
    pre_cancelled_passed = all(
        result["status"] == "passed" and result["call_started"] is False
        for result in cancellation["pre_cancelled"].values()
    )
    active_cancellation_passed = all(
        result["status"] == "passed"
        and result["call_started"] is True
        and result["acknowledgement_latency_seconds"] <= cancellation_limit
        for result in cancellation["active"].values()
    )
    checks = {
        "mlx_types": all(type_checks.values()),
        "resident_model_count_stable": memory_checks["resident_model_count_stable"],
        "peak_rss": peak_rss <= peak_limit,
        "swap_measurement_available": before_swap is not None and after_swap is not None,
        "mlx_memory_statistics_available": all(
            isinstance(value, int) and value >= 0 for value in mlx_memory.values()
        ),
        "pre_cancelled_rejected_before_call": pre_cancelled_passed,
        "active_cancellation_acknowledged": active_cancellation_passed,
        "realistic_metadata_controls": all(
            result["status"] == "passed" for result in realistic.values()
        ),
    }
    latencies = {
        "language": language_latencies,
        "asr": asr_latencies,
        "embeddings": embedding_latencies,
        **text_audio_latencies,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "candidate": candidate,
        "network_blocked": True,
        "elapsed_seconds": time.monotonic() - start,
        "physical_memory_bytes": memory,
        "hardware_feasibility": {
            "current_host_has_at_least_64_gib": memory >= 64 * 1024**3,
            "within_unchanged_peak_rss_gate": peak_rss <= peak_limit,
        },
        "model_identities": {
            **model_specs(candidate),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "model_object_ids": object_ids,
        "type_checks": type_checks,
        "mlx_provenance": provenance,
        "language_probabilities": language_probabilities,
        "transcript_evidence": {
            "utf8_bytes": len(transcript.encode()),
            "sha256": hashlib.sha256(transcript.encode()).hexdigest(),
        },
        "prompt_isolation_outputs": prompt_outputs,
        "realistic_metadata": realistic,
        "latencies_seconds": latencies,
        "throughput_calls_per_second": {
            name: _calls_per_second(values) for name, values in latencies.items()
        },
        "memory": memory_checks,
        "cancellation": cancellation,
        "thresholds": runtime_thresholds,
        "checks": checks,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES, default="qwen2_audio")
    parser.add_argument("--acceptance", type=Path, default=Path("benchmarks/acceptance.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest-model-smoke.json"))
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    checkpoint = RuntimeCheckpoint(args.output, candidate=args.candidate)
    try:
        if not args.manifest.is_file():
            raise FileNotFoundError(f"private labelled manifest is missing: {args.manifest}")
        result = run(
            args.models_dir,
            args.manifest,
            args.candidate,
            args.acceptance,
            checkpoint,
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        result = error_result(exc)
    persisted = checkpoint.finish(result)
    print_safe_runtime_result(persisted)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
