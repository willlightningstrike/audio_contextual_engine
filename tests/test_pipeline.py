from __future__ import annotations

import json
import shutil
import wave
from types import SimpleNamespace
from pathlib import Path

import pytest

from engine.cancel import CancellationToken
from engine.domain import (
    AudioStream,
    ContainerKind,
    ContentContext,
    JobRequest,
    PreparedMedia,
    SourceMetadata,
    TimeRange,
    TranscriptSegment,
)
from engine.lifecycle import JobStage
from engine.pipeline import Pipeline, resolve_content_context
from engine.signals.taxonomy import TaxonomyCandidate
from engine.store import JobStore


def test_context_defaults_are_container_based_and_never_inferred_from_audio():
    assert resolve_content_context(None, ContainerKind.AUDIO) is ContentContext.UNKNOWN
    assert resolve_content_context(None, ContainerKind.OTHER) is ContentContext.UNKNOWN
    assert resolve_content_context(None, ContainerKind.VIDEO) is ContentContext.VIDEO
    assert resolve_content_context(ContentContext.OTHER, ContainerKind.AUDIO) is ContentContext.OTHER
    assert resolve_content_context(ContentContext.MUSIC, ContainerKind.VIDEO) is ContentContext.MUSIC


class _Screener:
    def screen(self, audio, token):
        from engine.transcribe.types import ScreenResult

        return ScreenResult(speech_max=0.9, singing_max=0.0, admitted=True)


class _Transcriber:
    def detect_language_probabilities(self, audio, window, token):
        return {"en": 0.99}

    def transcribe_window(self, audio, window, language, token):
        return [
            TranscriptSegment(
                time_range=TimeRange(start_seconds=0, end_seconds=3),
                text="A camera review",
                confidence=0.9,
                language="en",
            )
        ]


class _Generator:
    def __init__(self, outputs=None):
        self.outputs = iter(
            outputs if outputs is not None else [
                {"primary_category": "633"},
                {
                    "primary_category": "633",
                    "secondary_categories": ["255"],
                    "keywords": [{"term": "camera", "confidence": 0.9}],
                    "entities": [],
                    "sentiment": {"label": "neutral", "score": 0.0},
                    "category_evidence": [
                        {"category_id": "633", "segment_indices": [0]},
                        {"category_id": "255", "segment_indices": [0]},
                    ],
                },
            ]
        )

    def generate_json(self, prompt, schema, *, max_tokens, token):
        from engine.signals.text_types import GenerationResult

        token.raise_if_cancelled()
        if "oneOf" in schema:
            output = next(self.verifier_outputs) if hasattr(self, "verifier_outputs") else {
                "status": "supported", "category_id": schema["oneOf"][0]["properties"]["category_id"]["const"],
                "segment_indices": [0],
            }
        else:
            output = next(self.outputs)
        return GenerationResult(raw=json.dumps(output), finish_reason="stop", seconds=0.01, token_count=10)


class _CancellingGenerator(_Generator):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def generate_json(self, prompt, schema, *, max_tokens, token):
        self.calls += 1
        generated = super().generate_json(prompt, schema, max_tokens=max_tokens, token=token)
        if self.calls == 2:
            token.cancel()
        return generated


class _Models:
    def __init__(self, outputs=None):
        self.transcriber = _Transcriber()
        self.screener = _Screener()
        self.embedder = object()
        self.text_generator = _Generator(outputs)
        self.text_model_identity = SimpleNamespace(
            repository="mlx-community/Qwen3.6-35B-A3B-4bit",
            revision="38740b847e4cb78f352aba30aa41c76e08e6eb46", manifest_sha256="a" * 64,
        )

    def release(self):
        self.released = True


class _CancellingModels(_Models):
    def __init__(self):
        super().__init__()
        self.text_generator = _CancellingGenerator()


class _TwoClassificationsModels(_Models):
    def __init__(self):
        super().__init__()
        first = next(self.text_generator.outputs)
        second = next(self.text_generator.outputs)
        self.text_generator.outputs = iter([first, second, first, second])


class _UncertainScreener(_Screener):
    def screen(self, audio, token):
        from engine.transcribe.types import ScreenResult

        return ScreenResult(speech_max=0.1, singing_max=0.001, admitted=False)


class _OutOfMemoryScreener(_Screener):
    def screen(self, audio, token):
        raise MemoryError("test-only model allocation failure")


class _FrenchTranscriber(_Transcriber):
    def detect_language_probabilities(self, audio, window, token):
        return {"fr": 0.99}


class _NoClassificationModels:
    def __init__(self, *, transcriber=None, screener=None):
        self.transcriber = transcriber or _Transcriber()
        self.screener = screener or _Screener()
        self.embedder = object()
        self.released = False

    @property
    def text_generator(self):
        raise AssertionError("ineligible input must not load the text model")

    def release(self):
        self.released = True


class _MalformedModels(_Models):
    def __init__(self):
        super().__init__()
        self.text_generator = _Generator()
        self.text_generator.outputs = iter([{"not_primary_category": "633"}])


class _Index:
    def shortlist(self, text, *, top_k, token):
        return (
            TaxonomyCandidate("633", "Cameras", ("Technology", "Cameras"), 0.9),
            TaxonomyCandidate("255", "Photography", ("Hobbies", "Photography"), 0.8),
        )


def _category_output(indices=(0,)):
    return {
        "primary_category": "633", "secondary_categories": [],
        "keywords": [], "entities": [],
        "sentiment": {"label": "neutral", "score": 0},
        "category_evidence": [{"category_id": "633", "segment_indices": list(indices)}],
    }


@pytest.fixture
def pipeline_fixture(tmp_path, monkeypatch):
    import engine.pipeline as pipeline_module
    from engine.transcribe.types import TranscriptionRun, WindowResult

    def make(*, audio_seconds=30, transcript_segments=((0, 3, "camera review"),),
             model_outputs=None, windows=None):
        source = tmp_path / f"source-{len(list(tmp_path.glob('source-*')))}.wav"
        with wave.open(str(source), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"")
        media = PreparedMedia(
            source_path=source, audio_path=source,
            container_kind=ContainerKind.AUDIO,
            content_context_hint=ContentContext.UNKNOWN,
            selected_audio_stream=0,
            available_audio_streams=(AudioStream(index=0, codec_name="pcm_s16le"),),
            duration_seconds=audio_seconds,
            source_metadata=SourceMetadata(duration_seconds=audio_seconds),
        )
        segments = tuple(
            TranscriptSegment(time_range=TimeRange(start_seconds=start, end_seconds=end),
                              text=text, confidence=0.9, language="en")
            for start, end, text in transcript_segments
        )
        source_windows = windows or (WindowResult(
            time_range=TimeRange(start_seconds=0, end_seconds=audio_seconds),
            segments=segments, language="en",
        ),)

        def transcribe(_media, *, checkpoint, on_progress, **_kwargs):
            for window in source_windows:
                checkpoint(window)
            on_progress(len(source_windows), len(source_windows))
            return TranscriptionRun(windows=source_windows, status="ready")

        monkeypatch.setattr(pipeline_module, "prepare_audio", lambda *_args, **_kwargs: media)
        monkeypatch.setattr(pipeline_module, "transcribe_media", transcribe)
        store = JobStore(tmp_path / "jobs")
        job = store.create(JobRequest(source=source))
        outputs = model_outputs if model_outputs is not None else (
            {"primary_category": "633"}, _category_output(),
        )
        return Pipeline(store, _Models(outputs), _Index()), store, job.id

    return make


def test_verified_job_journals_identity_citations_and_safe_snapshot(pipeline_fixture):
    from engine.web.snapshots import job_snapshot
    pipeline, store, job_id = pipeline_fixture(audio_seconds=3)
    pipeline.run(job_id, CancellationToken())
    saved = store.load(job_id)
    assert saved.status is JobStage.COMPLETE
    diagnostics = store.load_analysis_diagnostics(job_id)
    attempt = next(item["payload"] for item in diagnostics if item["kind"] == "verifier_attempt")
    assert attempt["policy_version"] == "english-transcript-verifier-v2"
    assert attempt["time_range"] == {"start_seconds": 0, "end_seconds": 3}
    assert attempt["model"] == vars(pipeline.models.text_model_identity)
    assert attempt["attempt"]["raw"]
    assert len(attempt["attempt"]["prompt_sha256"]) == 64
    result = next(item["payload"] for item in diagnostics if item["kind"] == "result")
    assert result["verification_policy_version"] == "english-transcript-verifier-v2"
    assert result["category_citations"] == [{"category_id": "633", "segment_indices": [0]}]
    snapshot = job_snapshot(store, saved)
    assert snapshot["evidence_mode"] == "transcript_only"
    assert snapshot["verification_windows"][0]["status"] == "supported"
    assert snapshot["verification_windows"][0]["citations"][0]["time_ranges"] == [
        {"start_seconds": 0, "end_seconds": 3}]
    assert "prompt_sha256" not in json.dumps(snapshot)
    assert "raw" not in snapshot["verification_windows"][0]


@pytest.mark.parametrize("outcome", ["unsupported", "uncertain", "invalid", "missing_model"])
def test_verifier_rejection_and_technical_failure_are_distinct(pipeline_fixture, outcome):
    from engine.web.snapshots import job_snapshot
    pipeline, store, job_id = pipeline_fixture(audio_seconds=3)
    if outcome == "missing_model":
        class MissingGenerator(_Generator):
            def generate_json(self, prompt, schema, **kwargs):
                if "oneOf" in schema:
                    raise FileNotFoundError("local model missing")
                return super().generate_json(prompt, schema, **kwargs)
        pipeline.models.text_generator = MissingGenerator(({"primary_category": "633"}, _category_output()))
    else:
        pipeline.models.text_generator.verifier_outputs = iter([
            {"status": outcome, "category_id": category, "segment_indices": []}
            for category in ("633", "255")])
    pipeline.run(job_id, CancellationToken())
    saved = store.load(job_id)
    assert saved.status is JobStage.FAILED
    assert store.read_export(job_id) is None
    assert saved.text_timeline == ()
    semantic = outcome in {"unsupported", "uncertain"}
    assert (saved.error == "INSUFFICIENT_TOPIC_EVIDENCE") is semantic
    snapshot = job_snapshot(store, saved)
    assert snapshot["verification_windows"][0]["status"] == ("abstained" if semantic else "technical_failure")
    assert snapshot["text_timeline"] == []
    assert any(item["kind"] == "verifier_attempt" for item in store.load_analysis_diagnostics(job_id))
    result = next(item["payload"] for item in store.load_analysis_diagnostics(job_id) if item["kind"] == "result")
    assert result["error_code"] == ("INSUFFICIENT_TOPIC_EVIDENCE" if semantic else
                                    "GENERATION_ERROR" if outcome == "missing_model" else "INVALID_VERIFIER_OUTPUT")


@pytest.mark.parametrize("boundary", ["before_checkpoint", "after_checkpoint"])
def test_verifier_checkpoint_cancellation_preserves_raw_attempt_and_stops(pipeline_fixture, monkeypatch, boundary):
    pipeline, store, job_id = pipeline_fixture(audio_seconds=3)
    token = CancellationToken()
    original = store.append_analysis_diagnostic
    def checkpoint(job_id, kind, payload):
        if kind == "verifier_attempt" and boundary == "before_checkpoint":
            token.cancel()
        result = original(job_id, kind, payload)
        if kind == "verifier_attempt" and boundary == "after_checkpoint":
            token.cancel()
        return result
    monkeypatch.setattr(store, "append_analysis_diagnostic", checkpoint)
    pipeline.run(job_id, token)
    assert store.load(job_id).status is JobStage.CANCELLED
    diagnostics = store.load_analysis_diagnostics(job_id)
    assert sum(item["kind"] == "verifier_attempt" for item in diagnostics) == 1
    assert not any(item["kind"] == "result" for item in diagnostics)
    assert store.read_export(job_id) is None


@pytest.mark.parametrize("active_window", [1, 2])
def test_blocking_verifier_reports_active_window_before_durable_completed_unit(pipeline_fixture, monkeypatch, active_window):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    if active_window == 1:
        pipeline, store, job_id = pipeline_fixture(audio_seconds=3)
    else:
        from engine.transcribe.types import WindowResult
        windows = tuple(WindowResult(
            time_range=TimeRange(start_seconds=start, end_seconds=end),
            segments=(() if start == 3 else (TranscriptSegment(
                time_range=TimeRange(start_seconds=start, end_seconds=end),
                text="camera review", confidence=0.9, language="en"),)),
            language="en",
        ) for start, end in ((0, 3), (3, 4), (4, 7)))
        pipeline, store, job_id = pipeline_fixture(
            audio_seconds=7, windows=windows,
            model_outputs=({"primary_category": "633"}, _category_output(),
                           {"primary_category": "633"}, _category_output()))
    entered, resume = Event(), Event()
    generator = pipeline.models.text_generator
    generate = generator.generate_json
    observed = []
    save = store.save
    def capture(record):
        if "calls completed; completed" in record.progress.message:
            assert any(item["kind"] == "verifier_attempt" for item in store.load_analysis_diagnostics(job_id))
        result = save(record)
        observed.append(record.progress)
        return result
    monkeypatch.setattr(store, "save", capture)
    def block(prompt, schema, **kwargs):
        if "oneOf" in schema and len([
            item for item in store.load_analysis_diagnostics(job_id) if item["kind"] == "result"
        ]) == active_window - 1:
            entered.set()
            assert resume.wait(5)
        return generate(prompt, schema, **kwargs)
    monkeypatch.setattr(generator, "generate_json", block)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(pipeline.run, job_id, CancellationToken())
        try:
            assert entered.wait(5)
            active = store.load(job_id).progress
            assert active.stage is JobStage.ANALYSING
            assert "transcript verification" in active.message
            assert f"window {active_window} of {active_window}" in active.message
            assert "0/5" in active.message
            assert ("0.000–3.000" if active_window == 1 else "4.000–7.000") in active.message
            assert active.completed_units == (3 if active_window == 1 else 8)
            assert active.total_units == (11 if active_window == 1 else 20)
            assert sum(item["kind"] == "verifier_attempt" for item in store.load_analysis_diagnostics(job_id)) == active_window - 1
        finally:
            resume.set()
        future.result(timeout=5)
    completed = next(item for item in observed if "1/5" in item.message
                     and f"window {active_window} of {active_window}" in item.message)
    assert completed.completed_units == (4 if active_window == 1 else 9)
    assert [item.percent for item in observed] == sorted(item.percent for item in observed)
    assert [item.revision for item in observed] == sorted(item.revision for item in observed)


def test_cancellation_between_transcription_writes_stops_before_transcript_append(pipeline_fixture, monkeypatch):
    pipeline, store, job_id = pipeline_fixture(audio_seconds=3)
    token = CancellationToken()
    append = store.append_transcription_window
    def cancel_after_window(*args):
        result = append(*args)
        token.cancel()
        return result
    monkeypatch.setattr(store, "append_transcription_window", cancel_after_window)
    pipeline.run(job_id, token)
    assert store.load(job_id).status is JobStage.CANCELLED
    assert store.load_transcription_windows(job_id)
    assert store.load(job_id).transcript == ()
    assert store.read_export(job_id) is None


def test_sparse_topical_audio_keeps_cited_timeline_but_cannot_export(pipeline_fixture):
    pipeline, store, job_id = pipeline_fixture()
    pipeline.run(job_id, CancellationToken())

    saved = store.load(job_id)
    diagnostics = store.load_analysis_diagnostics(job_id)
    assert saved.status is JobStage.FAILED
    assert saved.error == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert saved.transcript and saved.text_timeline
    assert saved.text_timeline[0]["evidence_ranges"] == (
        {"start_seconds": 0.0, "end_seconds": 3.0},
    )
    assert any(item["kind"] == "attempt" for item in diagnostics)
    assert any(item["kind"] == "result" and item["payload"]["category_citations"] for item in diagnostics)
    assert diagnostics[-1]["kind"] == "topic_policy"
    assert diagnostics[-1]["payload"]["outcome"] == "insufficient"
    assert saved.ortb_content is None and store.read_export(job_id) is None


def test_fully_supported_audio_exports_v2_cattax_9(pipeline_fixture):
    pipeline, store, job_id = pipeline_fixture(
        audio_seconds=10, transcript_segments=((0, 8, "camera review"),)
    )
    pipeline.run(job_id, CancellationToken())

    saved = store.load(job_id)
    assert saved.status is JobStage.COMPLETE
    assert saved.ortb_content["cattax"] == 9
    assert store.read_export(job_id) is not None
    assert store.load_analysis_diagnostics(job_id)[-1]["payload"]["outcome"] == "eligible"


def test_all_abstain_fails_with_semantic_policy(pipeline_fixture):
    pipeline, store, job_id = pipeline_fixture(
        model_outputs=({"primary_category": "__INSUFFICIENT_TOPIC_EVIDENCE__"},)
    )
    pipeline.models.text_generator.verifier_outputs = iter([
        {"status": "unsupported", "category_id": category, "segment_indices": []}
        for category in ("633", "255")])
    pipeline.run(job_id, CancellationToken())

    saved = store.load(job_id)
    assert saved.status is JobStage.FAILED
    assert saved.error == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert saved.text_timeline == ()
    assert store.load_analysis_diagnostics(job_id)[-1]["payload"]["outcome"] == "insufficient"


def test_hour_long_sparse_topical_audio_never_promotes(pipeline_fixture):
    pipeline, store, job_id = pipeline_fixture(audio_seconds=3600)
    pipeline.run(job_id, CancellationToken())
    assert store.load(job_id).error == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert store.read_export(job_id) is None


def test_mixed_abstention_and_cited_category_can_promote(pipeline_fixture):
    from engine.transcribe.types import WindowResult
    from engine.web.snapshots import job_snapshot

    windows = (
        WindowResult(time_range=TimeRange(start_seconds=0, end_seconds=2),
                     segments=(TranscriptSegment(time_range=TimeRange(start_seconds=0, end_seconds=2),
                                                 text="incidental words", confidence=0.9, language="en"),), language="en"),
        WindowResult(time_range=TimeRange(start_seconds=2, end_seconds=2)),
        WindowResult(time_range=TimeRange(start_seconds=2, end_seconds=20),
                     segments=(TranscriptSegment(time_range=TimeRange(start_seconds=2, end_seconds=14),
                                                 text="camera review", confidence=0.9, language="en"),), language="en"),
    )
    pipeline, store, job_id = pipeline_fixture(
        audio_seconds=20, windows=windows,
        model_outputs=({"primary_category": "__INSUFFICIENT_TOPIC_EVIDENCE__"},
                       {"primary_category": "633"}, _category_output()),
    )
    pipeline.models.text_generator.verifier_outputs = iter([
        {"status": "unsupported", "category_id": "633", "segment_indices": []},
        {"status": "uncertain", "category_id": "255", "segment_indices": []},
        {"status": "supported", "category_id": "633", "segment_indices": [0]},
    ])
    pipeline.run(job_id, CancellationToken())

    saved = store.load(job_id)
    snapshot = job_snapshot(store, saved)
    assert saved.status is JobStage.COMPLETE
    assert len(saved.text_timeline) == 1
    assert snapshot["topic_abstentions"] == [
        {"time_range": {"start_seconds": 0.0, "end_seconds": 2.0},
         "error_code": "INSUFFICIENT_TOPIC_EVIDENCE"}
    ]
    assert snapshot["coverage"]["failed_ranges"] == []
    assert store.read_export(job_id) is not None


def test_cancellation_before_policy_journal_keeps_partial_timeline(pipeline_fixture, monkeypatch):
    import engine.pipeline as pipeline_module

    pipeline, store, job_id = pipeline_fixture()
    token = CancellationToken()
    original = pipeline_module.evaluate_topic_policy_v3

    def cancel_after_decision(*args, **kwargs):
        decision = original(*args, **kwargs)
        token.cancel()
        return decision

    monkeypatch.setattr(pipeline_module, "evaluate_topic_policy_v3", cancel_after_decision)
    pipeline.run(job_id, token)

    saved = store.load(job_id)
    assert saved.status is JobStage.CANCELLED
    assert saved.text_timeline
    assert not any(item["kind"] == "topic_policy" for item in store.load_analysis_diagnostics(job_id))
    assert store.read_export(job_id) is None


def test_cancellation_during_policy_journal_remains_cancelled(pipeline_fixture, monkeypatch):
    pipeline, store, job_id = pipeline_fixture()
    token = CancellationToken()
    original = store.append_analysis_diagnostic

    def append_then_cancel(job_id, kind, payload):
        record = original(job_id, kind, payload)
        if kind == "topic_policy":
            token.cancel()
        return record

    monkeypatch.setattr(store, "append_analysis_diagnostic", append_then_cancel)
    pipeline.run(job_id, token)

    assert store.load(job_id).status is JobStage.CANCELLED
    assert store.load_analysis_diagnostics(job_id)[-1]["kind"] == "topic_policy"
    assert store.read_export(job_id) is None


def test_contiguous_cited_segments_form_one_timeline_evidence_range(pipeline_fixture):
    pipeline, store, job_id = pipeline_fixture(
        audio_seconds=20,
        transcript_segments=((0, 5, "camera sensor"), (5, 9, "camera lens")),
        model_outputs=({"primary_category": "633"}, _category_output((0, 1))),
    )
    pipeline.models.text_generator.verifier_outputs = iter([
        {"status": "supported", "category_id": "633", "segment_indices": [0, 1]}])
    pipeline.run(job_id, CancellationToken())

    saved = store.load(job_id)
    assert saved.status is JobStage.COMPLETE
    assert saved.text_timeline[0]["evidence_ranges"] == (
        {"start_seconds": 0.0, "end_seconds": 9.0},
    )


@pytest.mark.asyncio
async def test_batch_continues_after_insufficient_topic_child(tmp_path, monkeypatch):
    import engine.pipeline as pipeline_module
    from engine.runner import JobRunner
    from engine.transcribe.types import TranscriptionRun, WindowResult

    sources = tuple(tmp_path / name for name in ("sparse.wav", "supported.wav"))
    for source in sources:
        with wave.open(str(source), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16_000)
            audio.writeframes(b"")
    durations = {sources[0]: 30, sources[1]: 10}

    def prepare(source, *_args, **_kwargs):
        duration = durations[source]
        return PreparedMedia(
            source_path=source, audio_path=source,
            container_kind=ContainerKind.AUDIO,
            content_context_hint=ContentContext.UNKNOWN,
            selected_audio_stream=0,
            available_audio_streams=(AudioStream(index=0, codec_name="pcm_s16le"),),
            duration_seconds=duration,
            source_metadata=SourceMetadata(duration_seconds=duration),
        )

    def transcribe(media, *, checkpoint, on_progress, **_kwargs):
        duration = media.duration_seconds
        speech_end = 3 if duration == 30 else 8
        window = WindowResult(
            time_range=TimeRange(start_seconds=0, end_seconds=duration),
            segments=(TranscriptSegment(
                time_range=TimeRange(start_seconds=0, end_seconds=speech_end),
                text="camera review", confidence=0.9, language="en",
            ),), language="en",
        )
        checkpoint(window)
        on_progress(1, 1)
        return TranscriptionRun(windows=(window,), status="ready")

    monkeypatch.setattr(pipeline_module, "prepare_audio", prepare)
    monkeypatch.setattr(pipeline_module, "transcribe_media", transcribe)
    store = JobStore(tmp_path / "jobs")
    children = tuple(store.create(JobRequest(source=source)) for source in sources)
    batch = store.create_batch(tmp_path, tuple(job.id for job in children))
    models = _Models(({"primary_category": "633"}, _category_output(),
                      {"primary_category": "633"}, _category_output()))
    runner = JobRunner(Pipeline(store, models, _Index()), store)
    try:
        await runner.enqueue(batch.id)
        result = await runner.wait(batch.id)
    finally:
        await runner.shutdown()

    assert result.status is JobStage.COMPLETE
    assert result.failure_count == result.success_count == 1
    assert store.load(children[0].id).error == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert store.load(children[1].id).status is JobStage.COMPLETE


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_pipeline_e2e_uses_real_store_audio_ingest_and_semantic_export(tmp_path, monkeypatch):
    import engine.pipeline as pipeline_module

    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))
    original_prepare = pipeline_module.prepare_audio
    prepared_paths = []

    def capture_prepared_path(*args, **kwargs):
        media = original_prepare(*args, **kwargs)
        prepared_paths.append(media.audio_path)
        return media

    monkeypatch.setattr(pipeline_module, "prepare_audio", capture_prepared_path)

    models = _Models()
    Pipeline(store, models, _Index()).run(job.id, CancellationToken())

    completed = store.load(job.id)
    assert completed.status is JobStage.COMPLETE
    assert completed.prepared_media is None
    assert completed.resolved_content_context is ContentContext.UNKNOWN
    assert len(completed.transcript) == 1
    assert len(completed.text_timeline) == 1
    assert completed.ortb_content["cat"] == ("633", "255")
    assert completed.ortb_content["context"] == 7
    assert store.read_export(job.id) is not None
    assert prepared_paths and not prepared_paths[0].exists()
    assert models.released is True
    assert {item["kind"] for item in store.load_analysis_diagnostics(job.id)} == {
        "candidates",
        "attempt",
        "verifier_attempt",
        "result",
        "coverage_policy",
        "topic_policy",
    }


def test_cancelled_pipeline_keeps_audio_diagnostics_but_never_promotes(tmp_path):
    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))

    Pipeline(store, _CancellingModels(), _Index()).run(job.id, CancellationToken())

    cancelled = store.load(job.id)
    assert cancelled.status is JobStage.CANCELLED
    assert cancelled.transcript
    assert store.load_transcription_windows(job.id)
    assert any(
        item["kind"] == "attempt"
        for item in store.load_analysis_diagnostics(job.id)
    )
    assert cancelled.ortb_content is None
    assert store.read_export(job.id) is None


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_malformed_classifier_output_is_diagnostic_failure_never_generic_export(tmp_path):
    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))

    Pipeline(store, _MalformedModels(), _Index()).run(job.id, CancellationToken())

    failed = store.load(job.id)
    assert failed.status is JobStage.FAILED
    assert failed.ortb_content is None
    assert store.read_export(job.id) is None
    assert any(item["kind"] == "result" for item in store.load_analysis_diagnostics(job.id))


def test_missing_audio_source_fails_without_export_or_model_call(tmp_path):
    models = _Models()
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=tmp_path / "missing.wav"))

    Pipeline(store, models, _Index()).run(job.id, CancellationToken())

    failed = store.load(job.id)
    assert failed.status is JobStage.FAILED
    assert failed.ortb_content is None
    assert store.read_export(job.id) is None
    assert models.released is True


@pytest.mark.parametrize(
    ("models", "expected_language"),
    [
        (_NoClassificationModels(screener=_UncertainScreener()), None),
        (_NoClassificationModels(transcriber=_FrenchTranscriber()), "fr"),
    ],
)
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_ineligible_or_non_english_audio_fails_without_loading_text_model(
    tmp_path, models, expected_language
):
    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))

    Pipeline(store, models, _Index()).run(job.id, CancellationToken())

    failed = store.load(job.id)
    assert failed.status is JobStage.FAILED
    assert failed.language == expected_language
    assert failed.ortb_content is None
    assert store.read_export(job.id) is None
    assert models.released is True


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_cancellation_immediately_before_promotion_leaves_no_result(tmp_path, monkeypatch):
    import engine.pipeline as pipeline_module

    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))
    token = CancellationToken()
    build = pipeline_module.build_content

    def cancel_after_build(*args, **kwargs):
        result = build(*args, **kwargs)
        token.cancel()
        return result

    monkeypatch.setattr(pipeline_module, "build_content", cancel_after_build)
    Pipeline(store, _Models(), _Index()).run(job.id, token)

    cancelled = store.load(job.id)
    assert cancelled.status is JobStage.CANCELLED
    assert cancelled.ortb_content is None
    assert store.read_export(job.id) is None


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_model_memory_error_fails_closed_and_preserves_window_diagnostic(tmp_path):
    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))
    models = _NoClassificationModels(screener=_OutOfMemoryScreener())

    Pipeline(store, models, _Index()).run(job.id, CancellationToken())

    failed = store.load(job.id)
    assert failed.status is JobStage.FAILED
    assert failed.ortb_content is None
    assert store.read_export(job.id) is None
    windows = store.load_transcription_windows(job.id)
    assert windows and windows[0].failure_reason == "model_call_failed"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_single_final_uncertain_vocal_window_can_export_with_excluded_range(tmp_path, monkeypatch):
    import engine.pipeline as pipeline_module
    from engine.transcribe.types import TranscriptionRun, WindowResult

    source = tmp_path / "vocal-song.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * (35 * 16_000))
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))
    original_aggregate = pipeline_module.aggregate_topic_sufficient
    aggregation_inputs = []

    def capture_aggregation_inputs(entries, *, decision, coverage, taxonomy):
        aggregation_inputs.append((tuple(entries), coverage))
        return original_aggregate(entries, decision=decision, coverage=coverage, taxonomy=taxonomy)

    monkeypatch.setattr(pipeline_module, "aggregate_topic_sufficient", capture_aggregation_inputs)

    first = WindowResult(
        time_range=TimeRange(start_seconds=0, end_seconds=30),
        segments=(
            TranscriptSegment(
                time_range=TimeRange(start_seconds=1, end_seconds=12),
                text="English lyrics",
                confidence=0.9,
                language="en",
            ),
        ),
        language="en",
    )
    final_uncertain = WindowResult(
        time_range=TimeRange(start_seconds=30, end_seconds=35),
        error_code="VOCAL_PRESENCE_UNCERTAIN",
        failure_stage="vocal_screening",
        failure_reason="vocal_presence_uncertain",
    )

    def partial_vocal_run(media, *, checkpoint, on_progress, **kwargs):
        windows = (first, final_uncertain)
        for item in windows:
            checkpoint(item)
        on_progress(len(windows), len(windows))
        return TranscriptionRun(windows=windows, status="partial")

    monkeypatch.setattr(pipeline_module, "transcribe_media", partial_vocal_run)
    Pipeline(store, _Models(), _Index()).run(job.id, CancellationToken())

    completed = store.load(job.id)
    assert completed.status is JobStage.COMPLETE
    assert completed.ortb_content is not None
    assert store.read_export(job.id) is not None
    assert completed.ortb_content["cat"] == ("633", "255")
    saved_windows = store.load_transcription_windows(job.id)
    assert saved_windows == (first, final_uncertain)
    assert len(aggregation_inputs) == 1
    entries, coverage = aggregation_inputs[0]
    assert coverage.excluded_ranges == (TimeRange(start_seconds=30, end_seconds=35),)
    assert coverage.failed_ranges == ()
    assert all(
        evidence.end_seconds <= 30
        for entry in entries
        for evidence in entry.evidence_ranges
    )


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_partial_asr_gap_splits_classification_windows_and_prevents_promotion(tmp_path, monkeypatch):
    import engine.pipeline as pipeline_module
    from engine.transcribe.types import TranscriptionRun, WindowResult

    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * (65 * 16_000))
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))

    def partial_run(media, *, checkpoint, on_progress, **kwargs):
        first = WindowResult(
            time_range=TimeRange(start_seconds=0, end_seconds=30),
            segments=(TranscriptSegment(time_range=TimeRange(start_seconds=1, end_seconds=3), text="camera review", confidence=0.9, language="en"),),
            language="en",
        )
        gap = WindowResult(
            time_range=TimeRange(start_seconds=30, end_seconds=60),
            language="en",
            error_code="MODEL_FAILURE",
            failure_stage="transcription",
            failure_reason="model_call_failed",
        )
        last = WindowResult(
            time_range=TimeRange(start_seconds=60, end_seconds=65),
            segments=(TranscriptSegment(time_range=TimeRange(start_seconds=61, end_seconds=63), text="camera lens", confidence=0.9, language="en"),),
            language="en",
        )
        windows = (first, gap, last)
        for item in windows:
            checkpoint(item)
        on_progress(3, 3)
        return TranscriptionRun(windows=windows, status="partial")

    monkeypatch.setattr(pipeline_module, "transcribe_media", partial_run)
    Pipeline(store, _TwoClassificationsModels(), _Index()).run(job.id, CancellationToken())

    failed = store.load(job.id)
    ranges = tuple(item["time_range"] for item in failed.text_timeline)
    assert failed.status is JobStage.FAILED
    assert ranges == (
        {"start_seconds": 1.0, "end_seconds": 3.0},
        {"start_seconds": 61.0, "end_seconds": 63.0},
    )
    assert failed.ortb_content is None
    assert store.read_export(job.id) is None


@pytest.mark.parametrize("boundary", ["extraction", "transcription", "classification"])
@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools required")
def test_cancellation_after_each_pipeline_stage_preserves_diagnostics_and_never_exports(
    tmp_path, monkeypatch, boundary
):
    import engine.pipeline as pipeline_module

    source = tmp_path / "spoken.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x01\x00" * 80_000)
    store = JobStore(tmp_path / "jobs")
    job = store.create(JobRequest(source=source))
    token = CancellationToken()
    target_name = {
        "extraction": "prepare_audio",
        "transcription": "transcribe_media",
        "classification": "classify_and_verify_transcription_run",
    }[boundary]
    original = getattr(pipeline_module, target_name)

    def cancel_after(*args, **kwargs):
        result = original(*args, **kwargs)
        token.cancel()
        return result

    monkeypatch.setattr(pipeline_module, target_name, cancel_after)
    Pipeline(store, _Models(), _Index()).run(job.id, token)

    cancelled = store.load(job.id)
    assert cancelled.status is JobStage.CANCELLED
    assert cancelled.ortb_content is None
    assert store.read_export(job.id) is None
    if boundary != "extraction":
        assert store.load_transcription_windows(job.id)


def test_pipeline_model_registry_releases_weights_without_replacing_embedder(tmp_path, monkeypatch):
    import engine.models.registry as registry_module

    model_path = tmp_path.resolve()
    monkeypatch.setattr(registry_module, "load_verified_asr_path", lambda _path: model_path)
    monkeypatch.setattr(registry_module, "load_verified_embedding_path", lambda _path: model_path)
    monkeypatch.setattr(registry_module, "load_verified_text_path", lambda _path: model_path)
    (model_path / "text.json").write_bytes(b'{"verified": true}\n')

    class Adapter:
        def __init__(self):
            self.release_count = 0

        def release(self):
            self.release_count += 1

    created = []

    def load(_path):
        value = Adapter()
        created.append(value)
        return value

    models = registry_module.EnglishPipelineModelRegistry(
        asr_manifest=model_path / "asr.json",
        embedding_manifest=model_path / "embedding.json",
        text_manifest=model_path / "text.json",
        yamnet_path=model_path,
        load_asr=load,
        load_embedder=load,
        load_screener=load,
        load_text=load,
    )
    embedder = models.embedder
    text_generator = models.text_generator

    models.release()

    assert embedder.release_count == 1
    assert models.embedder is embedder
    assert text_generator.release_count == 1
    assert models.text_generator is not text_generator
    import hashlib
    assert models.text_model_identity.repository == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert models.text_model_identity.revision == "38740b847e4cb78f352aba30aa41c76e08e6eb46"
    assert models.text_model_identity.manifest_sha256 == hashlib.sha256(b'{"verified": true}\n').hexdigest()


def _v3_windows(kinds):
    """One 30 s source window per code: e English, m music, f French."""
    from engine.transcribe.types import WindowResult

    windows = []
    for index, kind in enumerate(kinds):
        start = index * 30.0
        time_range = TimeRange(start_seconds=start, end_seconds=start + 30)
        if kind == "e":
            windows.append(WindowResult(time_range=time_range, language="en", segments=(
                TranscriptSegment(time_range=TimeRange(start_seconds=start, end_seconds=start + 29),
                                  text="A camera review", confidence=0.9, language="en"),
            )))
        elif kind == "m":
            windows.append(WindowResult(
                time_range=time_range, error_code="VOCAL_PRESENCE_UNCERTAIN",
                failure_stage="vocal_screening", failure_reason="vocal_presence_uncertain",
            ))
        else:
            windows.append(WindowResult(
                time_range=time_range, language="fr",
                error_code="NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE",
                failure_stage="language_detection",
                failure_reason="non_english_or_unconfirmed_language",
            ))
    return tuple(windows)


def _segments(content):
    return {item["id"]: item["value"] for item in content["data"][0]["segment"]}


def test_music_break_and_short_non_english_stretch_no_longer_block_export(pipeline_fixture):
    windows = _v3_windows("eeemfeee")
    pipeline, store, job_id = pipeline_fixture(
        audio_seconds=240, windows=windows,
        model_outputs=[{"primary_category": "633"}, _category_output((0, 1, 2))] * 2,
    )
    pipeline.models.text_generator.verifier_outputs = iter(
        [{"status": "supported", "category_id": "633", "segment_indices": [0, 1, 2]}] * 2
    )

    pipeline.run(job_id, CancellationToken())

    saved = store.load(job_id)
    assert saved.status is JobStage.COMPLETE
    assert saved.ortb_content["cat"] == ("633",)
    segments = _segments(saved.ortb_content)
    assert segments["topic_structure"] == "single"
    assert segments["coverage"] == "complete"
    coverage = next(item["payload"] for item in store.load_analysis_diagnostics(job_id)
                    if item["kind"] == "coverage_policy")
    assert coverage["non_english_share"] == pytest.approx(30 / 210)
    assert coverage["excluded_ranges"] == [
        {"start_seconds": 90.0, "end_seconds": 120.0},
        {"start_seconds": 120.0, "end_seconds": 150.0},
    ]
    from engine.web.snapshots import job_snapshot

    snapshot = job_snapshot(store, saved)
    assert snapshot["coverage"]["failed_ranges"] == []
    assert [item["error_code"] for item in snapshot["coverage"]["excluded_ranges"]] == [
        "VOCAL_PRESENCE_UNCERTAIN", "NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE",
    ]


@pytest.mark.parametrize(("failures", "expected"), [(1, "complete"), (2, "failed")])
def test_failed_windows_are_excluded_up_to_the_technical_cap(pipeline_fixture, failures, expected):
    # Music breaks split the speech into five separately classified windows.
    windows = _v3_windows("emememememe"[:9])
    outputs = []
    for index in range(5):
        outputs.append({"primary_category": "633"})
        outputs.append({"malformed": True} if index < failures else _category_output())
    pipeline, store, job_id = pipeline_fixture(
        audio_seconds=270, windows=windows, model_outputs=outputs,
    )

    pipeline.run(job_id, CancellationToken())

    saved = store.load(job_id)
    if expected == "complete":
        assert saved.status is JobStage.COMPLETE
        assert _segments(saved.ortb_content)["coverage"] == "partial_technical_failures"
        assert store.read_export(job_id) is not None
    else:
        assert saved.status is JobStage.FAILED
        assert saved.error == "TECHNICAL_FAILURE_SHARE_EXCEEDED"
        assert store.read_export(job_id) is None
