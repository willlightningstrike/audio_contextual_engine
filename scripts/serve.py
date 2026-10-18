"""Start the local UI/API only after all offline setup checks pass."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from engine.config import Settings
from scripts.check_setup import OFFLINE_ENVIRONMENT, check_setup, enable_offline_mode


def build_production_app(settings: Settings) -> Any:
    """Construct the app and worker from already verified local artifacts."""
    from engine.models.registry import (
        MODEL_SPECS,
        EnglishPipelineModelRegistry,
        load_verified_yamnet_path,
    )
    from engine.pipeline import Pipeline
    from engine.runner import JobRunner
    from engine.signals.taxonomy import load_taxonomy_index
    from engine.standards import load_taxonomy
    from engine.store import JobStore
    from engine.web.app import create_app

    model_manifest = settings.models_dir / "english-phase1-manifest.json"
    text_manifest = settings.models_dir / "qwen25-manifest.json"
    yamnet_manifest = settings.models_dir / "yamnet-manifest.json"
    models = EnglishPipelineModelRegistry(
        asr_manifest=model_manifest,
        embedding_manifest=model_manifest,
        text_manifest=text_manifest,
        yamnet_path=load_verified_yamnet_path(yamnet_manifest),
    )
    taxonomy_index = load_taxonomy_index(
        settings.taxonomy_index_dir,
        taxonomy_path=(
            Path(__file__).resolve().parents[1]
            / "data"
            / "standards"
            / "content-taxonomy-3.0.tsv"
        ),
        taxonomy=load_taxonomy(),
        embedder=models.embedder,
        expected_embedding_revision=MODEL_SPECS["embeddings"]["revision"],
    )
    store = JobStore(settings.jobs_dir)
    runner = JobRunner(Pipeline(store, models, taxonomy_index), store)
    app = create_app(store=store, runner=runner, settings=settings)
    app.router.add_event_handler("shutdown", runner.shutdown)
    return app


def main() -> int:
    enable_offline_mode()
    settings = Settings.from_environment()
    failures = check_setup(settings)
    if failures:
        print("Contextual Engine will not start:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    app = build_production_app(settings)
    import uvicorn

    uvicorn.run(app, host=settings.bind_host, port=settings.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
