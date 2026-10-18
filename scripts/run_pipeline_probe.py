"""Run one local, real-model end-to-end pipeline probe from Terminal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import threading
import time
from typing import Sequence
from uuid import UUID

from engine.cancel import CancellationToken
from engine.domain import ContentContext, JobRecord, JobRequest
from engine.lifecycle import JobStage
from engine.models.registry import EnglishPipelineModelRegistry, MODEL_SPECS
from engine.models import mlx_memory_statistics
from engine.pipeline import Pipeline
from engine.signals.taxonomy import load_taxonomy_index
from engine.standards import TaxonomyRow, load_taxonomy
from engine.store import JobStore


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"
TAXONOMY_PATH = ROOT / "data" / "standards" / "content-taxonomy-3.0.tsv"
TAXONOMY_INDEX_DIR = MODELS_DIR / "taxonomy-index"
DEFAULT_JOBS_DIR = ROOT / "benchmarks" / "runs.local" / "task13-terminal-probe"


def persist_summary(jobs_dir: Path, summary: dict[str, object]) -> Path:
    """Write one non-overwriting probe report beside its durable job record."""
    job_id = summary.get("job_id")
    if not isinstance(job_id, str) or str(UUID(job_id)) != job_id:
        raise ValueError("summary must contain a canonical UUID job_id")
    requested_root = Path(jobs_dir)
    if requested_root.is_symlink():
        raise ValueError("jobs directory must not be a symlink")
    root = requested_root.resolve(strict=True)
    job_dir = root / job_id
    if job_dir.is_symlink() or not job_dir.is_dir() or job_dir.resolve().parent != root:
        raise ValueError("job directory is missing or unsafe")
    report_path = job_dir / "probe-report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return report_path


def build_measurements(
    *,
    started_at: float,
    stopped_at: float,
    max_rss: int,
    platform_name: str,
    mlx_memory: dict[str, int] | None,
    cancellation_requested_at: float | None,
) -> dict[str, object]:
    """Normalize process and MLX memory readings and cancellation latency."""
    peak_rss_bytes = int(max_rss) if platform_name == "darwin" else int(max_rss) * 1024
    return {
        "wall_seconds": max(0.0, stopped_at - started_at),
        "peak_process_rss_bytes": peak_rss_bytes,
        "mlx_memory": mlx_memory,
        "cancellation_request_to_stop_seconds": (
            None
            if cancellation_requested_at is None
            else max(0.0, stopped_at - cancellation_requested_at)
        ),
    }


def summarize_record(
    record: JobRecord, taxonomy: Sequence[TaxonomyRow]
) -> dict[str, object]:
    """Return a compact report with human-readable predicted category labels."""
    content = record.ortb_content or {}
    raw_categories = content.get("cat", ())
    category_ids = raw_categories if isinstance(raw_categories, (list, tuple)) else ()
    names = {row.id: row.name for row in taxonomy}
    categories = [
        {"id": category_id, "name": names.get(category_id, "Unknown taxonomy ID")}
        for category_id in category_ids
        if isinstance(category_id, str)
    ]
    context = record.resolved_content_context
    return {
        "job_id": record.id,
        "status": record.status.value,
        "source": None if record.request is None else str(record.request.source),
        "progress": record.progress.percent,
        "message": record.progress.message,
        "language": record.language,
        "context": None if context is None else context.value,
        "categories": categories,
        "error": record.error,
    }


def _set_initialization_failure(store: JobStore, record: JobRecord, exc: Exception) -> JobRecord:
    failed = record.model_copy(
        update={
            "status": JobStage.FAILED,
            "error": type(exc).__name__,
            "progress": record.progress.advance(
                stage=JobStage.FAILED,
                percent=100,
                message="pipeline initialization failed",
            ),
        }
    )
    store.save(failed)
    return failed


def run_probe(
    source: Path,
    jobs_dir: Path = DEFAULT_JOBS_DIR,
    *,
    context: ContentContext | None = None,
    audio_stream_index: int | None = None,
) -> dict[str, object]:
    """Run the production pipeline with its pinned local MLX models."""
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"source is not a file: {source}")

    started_at = time.monotonic()
    cancellation_requested_at: float | None = None
    store = JobStore(jobs_dir)
    record = store.create(
        JobRequest(
            source=source,
            requested_context=context,
            audio_stream_index=audio_stream_index,
        )
    )
    models: EnglishPipelineModelRegistry | None = None
    pipeline_started = False
    try:
        print(json.dumps({"event": "initializing_models"}), flush=True)
        models = EnglishPipelineModelRegistry(
            asr_manifest=MODELS_DIR / "model-manifest.json",
            embedding_manifest=MODELS_DIR / "model-manifest.json",
            text_manifest=MODELS_DIR / "qwen25-manifest.json",
            yamnet_path=MODELS_DIR / "yamnet",
        )
        taxonomy = load_taxonomy()
        taxonomy_index = load_taxonomy_index(
            TAXONOMY_INDEX_DIR,
            taxonomy_path=TAXONOMY_PATH,
            taxonomy=taxonomy,
            embedder=models.embedder,
            expected_embedding_revision=MODEL_SPECS["embeddings"]["revision"],
        )
        token = CancellationToken()
        pipeline = Pipeline(store, models, taxonomy_index)
        worker = threading.Thread(
            target=pipeline.run,
            args=(record.id, token),
            name=f"pipeline-probe-{record.id}",
            daemon=True,
        )
        worker.start()
        pipeline_started = True
        last_revision = -1
        try:
            while worker.is_alive():
                current = store.load(record.id)
                if current.progress.revision != last_revision:
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "stage": current.status.value,
                                "percent": current.progress.percent,
                                "message": current.progress.message,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    last_revision = current.progress.revision
                worker.join(timeout=0.5)
        except KeyboardInterrupt:
            cancellation_requested_at = time.monotonic()
            token.cancel()
            print(json.dumps({"event": "cancellation_requested"}), flush=True)
            while worker.is_alive():
                worker.join(timeout=0.5)
        result = store.load(record.id)
        summary = summarize_record(result, taxonomy)
    except Exception as exc:
        current = store.load(record.id)
        if current.status is JobStage.QUEUED:
            current = _set_initialization_failure(store, current, exc)
        summary = summarize_record(current, load_taxonomy())
    finally:
        if models is not None and not pipeline_started:
            try:
                models.release()
            except Exception:
                pass
    stopped_at = time.monotonic()
    try:
        mlx_memory = mlx_memory_statistics()
    except Exception:
        mlx_memory = None
    summary["measurements"] = build_measurements(
        started_at=started_at,
        stopped_at=stopped_at,
        max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform_name=sys.platform,
        mlx_memory=mlx_memory,
        cancellation_requested_at=cancellation_requested_at,
    )
    try:
        report_path = persist_summary(store.jobs_root, summary)
        summary["evidence_file"] = str(report_path)
    except (OSError, ValueError) as exc:
        summary["evidence_write_error"] = type(exc).__name__
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real-model English audio pipeline probe. The source media is "
            "read-only; job evidence is written beneath --jobs-dir."
        )
    )
    parser.add_argument("source", type=Path, help="English speech media file to analyze")
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument(
        "--context",
        choices=[str(value.value) for value in ContentContext],
        help="optional explicit OpenRTB content context override",
    )
    parser.add_argument("--audio-stream-index", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = None if args.context is None else ContentContext(int(args.context))
    try:
        result = run_probe(
            args.source,
            args.jobs_dir,
            context=context,
            audio_stream_index=args.audio_stream_index,
        )
    except Exception as exc:
        result = {
            "status": "failed_before_job_creation",
            "error": type(exc).__name__,
        }
    print(json.dumps({"event": "result", **result}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == JobStage.COMPLETE.value else 2


if __name__ == "__main__":
    raise SystemExit(main())
