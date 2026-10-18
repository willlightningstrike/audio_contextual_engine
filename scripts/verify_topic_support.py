"""Non-blind, isolated verifier replay; never modifies saved v1 jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.signals.classify import CandidateRecord, analysis_windows_from_run
from engine.signals.protocol import ConstrainedTextGenerator
from engine.signals.taxonomy import TaxonomyCandidate
from engine.signals.text_types import ClassificationResult
from engine.signals.topic_verify import VERIFIER_POLICY, parse_verifier_judgment
from engine.signals.verified_topics import verify_window_topics
from engine.transcribe.types import TranscriptionRun, WindowResult

ROOT = Path(__file__).resolve().parents[1]
TEXT_MANIFEST = ROOT / "data/models/qwen25-manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl_snapshot(path: Path) -> tuple[tuple[dict, ...], str]:
    content = path.read_bytes()
    records = tuple(json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip())
    return records, hashlib.sha256(content).hexdigest()


def _range_key(interval: TimeRange) -> tuple[float, float]:
    return interval.start_seconds, interval.end_seconds


def load_saved_case(job_dir: Path) -> tuple[TranscriptionRun, tuple[ClassificationResult, ...], tuple[CandidateRecord, ...]]:
    case, _ = _load_saved_case_snapshot(job_dir)
    return case


def _load_saved_case_snapshot(job_dir: Path) -> tuple[
    tuple[TranscriptionRun, tuple[ClassificationResult, ...], tuple[CandidateRecord, ...]], dict[str, str],
]:
    transcription, transcription_sha = _read_jsonl_snapshot(job_dir / "transcription-windows.jsonl")
    diagnostics, diagnostics_sha = _read_jsonl_snapshot(job_dir / "analysis-diagnostics.jsonl")
    windows = tuple(WindowResult.model_validate(record["unit"])
                    for record in transcription)
    successes = sum(window.error_code is None for window in windows)
    run = TranscriptionRun(windows=windows, status=("ready" if windows and successes == len(windows)
                           else "partial" if successes else "failed"))
    results = []
    candidates = []
    for record in diagnostics:
        payload = record["payload"]
        if record["kind"] == "result":
            results.append(ClassificationResult.model_validate_json(json.dumps(payload)))
        elif record["kind"] == "candidates":
            candidates.append(CandidateRecord(
                time_range=TimeRange.model_validate(payload["time_range"]),
                candidates=tuple(TaxonomyCandidate(taxonomy_id=item["taxonomy_id"], name=item["name"],
                    path=tuple(item["path"]), similarity=item["similarity"]) for item in payload["candidates"])))
    expected = [_range_key(window.time_range) for window in analysis_windows_from_run(run)]
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate analysis window range")
    aligned = []
    for records in (results, candidates):
        by_range = {_range_key(item.time_range): item for item in records}
        if len(by_range) != len(records) or set(by_range) != set(expected):
            raise ValueError("missing, extra, or duplicate saved analysis record")
        aligned.append(tuple(by_range[key] for key in expected))
    return (run, aligned[0], aligned[1]), {
        "transcription-windows.jsonl": transcription_sha,
        "analysis-diagnostics.jsonl": diagnostics_sha,
    }


def run_probe(*, cases: Sequence[tuple[str, Path]], out: Path,
              generator_factory: Callable[[], ConstrainedTextGenerator]) -> dict:
    out = Path(out)
    sidecar = out.with_suffix(".attempts.jsonl")
    for path in (out, sidecar):
        if path.exists():
            raise FileExistsError(path)
    if len({name for name, _ in cases}) != len(cases):
        raise ValueError("case names must be unique")
    saved = [(name, Path(path), _load_saved_case_snapshot(Path(path))) for name, path in cases]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve both paths before inference. Open final report once, exclusively;
    # no replacement or reopening can overwrite concurrent/historical evidence.
    with out.open("x", encoding="utf-8") as destination, sidecar.open("x", encoding="utf-8") as journal:
        started = time.monotonic()
        metadata_error = None
        model = None
        try:
            manifest_bytes = TEXT_MANIFEST.read_bytes()
            manifest = json.loads(manifest_bytes)
            model = {"manifest_path": str(TEXT_MANIFEST), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                     "repository": manifest["model"]["repository"], "revision": manifest["model"]["revision"]}
            generator = generator_factory()
        except Exception as exc:
            generator = None
            metadata_error = f"{type(exc).__name__}: {exc}"
        rows = []
        inputs = []
        cancelled = False
        for name, path, ((run, results, candidates), source_sha256) in saved:
            inputs.append({"case": name, "job_dir": str(path.resolve()), "transcription_status": run.status,
                "inherited_asr_errors": [window.model_dump(mode="json") for window in run.windows if window.error_code],
                "source_sha256": source_sha256})
            windows = analysis_windows_from_run(run)
            if not windows:
                rows.append({"case": name, "window": None, "proposed_ids": [], "verified_ids": [],
                             "attempts": [], "verdicts": [],
                             "error": "NOT_ATTEMPTED_AFTER_CANCELLATION" if cancelled else "NO_ANALYSIS_WINDOWS",
                             "status": "not_attempted" if cancelled else "failed", "non_blind": True})
            for window, proposed, ranked in zip(windows, results, candidates, strict=True):
                row = {"case": name, "window": window.time_range.model_dump(mode="json"),
                       "proposed_ids": ([proposed.payload.primary_category, *proposed.payload.secondary_categories]
                                        if proposed.payload else []),
                       "verified_ids": [], "attempts": [], "verdicts": [], "error": None,
                       "status": "completed", "non_blind": True}
                def checkpoint(attempt):
                    row["attempts"].append(attempt.model_dump(mode="json"))
                    journal.write(json.dumps({"case": name, "window": row["window"],
                                              "attempt": attempt.model_dump(mode="json")}) + "\n")
                    journal.flush()
                if cancelled:
                    row["error"] = "NOT_ATTEMPTED_AFTER_CANCELLATION"
                    row["status"] = "not_attempted"
                elif generator is None:
                    row["error"] = "MODEL_PREFLIGHT_ERROR"
                    row["status"] = "failed"
                else:
                    try:
                        verified = verify_window_topics(window, proposed, ranked.candidates,
                            generator=generator, token=CancellationToken(), checkpoint=checkpoint)
                    except JobCancelled:
                        cancelled = True
                        row["error"] = "JOB_CANCELLED"
                        row["status"] = "cancelled"
                    else:
                        row["verified_ids"] = ([verified.primary_category, *verified.secondary_categories]
                                               if verified.primary_category else [])
                        row["attempts"] = [attempt.model_dump(mode="json") for attempt in verified.attempts]
                        row["error"] = verified.error_code
                        if verified.error_code:
                            row["status"] = "failed"
                        for attempt in verified.attempts:
                            if attempt.error_code is None and attempt.raw is not None:
                                row["verdicts"].append(parse_verifier_judgment(attempt.raw, attempt.category_id, window).model_dump(mode="json"))
                rows.append(row)
        raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory = None
        if generator is not None and type(generator).__module__.startswith("engine.models"):
            from engine.models import mlx_memory_statistics
            memory = mlx_memory_statistics()
        report = {"policy_version": VERIFIER_POLICY, "non_blind": True, "model": model,
                  "preflight_error": metadata_error, "inputs": inputs, "cases": rows,
                  "attempt_journal": str(sidecar.resolve()), "attempt_journal_sha256": _sha256(sidecar),
                  "wall_seconds": time.monotonic() - started,
                  "peak_process_rss_bytes": int(raw_rss if platform.system() == "Darwin" else raw_rss * 1024),
                  "mlx_allocator": memory}
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
        return report


def _live_generator() -> ConstrainedTextGenerator:
    from engine.models.registry import TextModelRegistry, load_verified_text_path
    return TextModelRegistry(load_verified_text_path(TEXT_MANIFEST)).generator


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], metavar="NAME=JOB_DIR")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = []
    for value in args.case:
        name, separator, directory = value.partition("=")
        if not separator or not name or not directory:
            parser.error("--case must be NAME=JOB_DIR")
        cases.append((name, Path(directory)))
    report = run_probe(cases=cases, out=args.out, generator_factory=_live_generator)
    return int(any(row["error"] for row in report["cases"]) or report["preflight_error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
