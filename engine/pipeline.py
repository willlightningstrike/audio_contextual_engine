"""Orchestrate the bounded English audio-to-contextual-result pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Protocol

from engine.cancel import CancellationToken, JobCancelled
from engine.completion import MAX_NON_ENGLISH_SHARE, CoverageAssessment, assess_coverage
from engine.domain import (
    ContainerKind,
    ContentContext,
    JobRecord,
    PreparedMedia,
    TimeRange,
)
from engine.events import Progress
from engine.ingest.audio import prepare_audio
from engine.lifecycle import JobStage
from engine.ortb.build import build_content
from engine.signals.aggregate import aggregate_topic_sufficient
from engine.signals.classify import analysis_windows_from_run
from engine.signals.verified_classification import classify_and_verify_transcription_run
from engine.signals.topic_verify import VERIFIER_POLICY
from engine.signals.topic_sufficiency import evaluate_topic_policy_v3
from engine.signals.protocol import ConstrainedTextGenerator, Embedder, VocalScreener
from engine.signals.result_types import (
    AudioTranscriptProvenance,
    CoverageSummary,
    TextTimelineEntry,
)
from engine.signals.taxonomy import TaxonomyIndex
from engine.standards import load_taxonomy
from engine.store import JobStore
from engine.transcribe.protocol import Transcriber
from engine.transcribe.service import transcribe_media
from engine.transcribe.types import WindowResult

if TYPE_CHECKING:
    from engine.models.registry import TextModelIdentity


class PipelineModels(Protocol):
    @property
    def transcriber(self) -> Transcriber: ...

    @property
    def screener(self) -> VocalScreener: ...

    @property
    def embedder(self) -> Embedder: ...

    @property
    def text_generator(self) -> ConstrainedTextGenerator: ...

    @property
    def text_model_identity(self) -> TextModelIdentity: ...

    def release(self) -> None: ...


class InsufficientTopicEvidence(Exception):
    """Handled whole-file outcome when cited English topic support is too sparse."""


class CoverageFailure(Exception):
    """Handled whole-file outcome when a v3 coverage cap is exceeded."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _union_ranges(ranges: tuple[TimeRange, ...]) -> tuple[TimeRange, ...]:
    ordered = sorted(ranges, key=lambda item: (item.start_seconds, item.end_seconds))
    if not ordered:
        return ()
    merged = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if item.start_seconds <= previous.end_seconds:
            merged[-1] = TimeRange(
                start_seconds=previous.start_seconds,
                end_seconds=max(previous.end_seconds, item.end_seconds),
            )
        else:
            merged.append(item)
    return tuple(merged)


def resolve_content_context(
    requested: ContentContext | None, container: ContainerKind
) -> ContentContext:
    """Resolve descriptive context without inferring it from audio semantics."""
    if requested is not None:
        return ContentContext(requested)
    if container is ContainerKind.VIDEO:
        return ContentContext.VIDEO
    return ContentContext.UNKNOWN


class Pipeline:
    def __init__(
        self,
        store: JobStore,
        models: PipelineModels,
        taxonomy_index: TaxonomyIndex,
    ) -> None:
        self.store = store
        self.models = models
        self.taxonomy_index = taxonomy_index

    def run(self, job_id: str, token: CancellationToken) -> None:
        record = self.store.load(job_id)
        workspace: TemporaryDirectory[str] | None = None
        try:
            token.raise_if_cancelled()
            if record.request is None:
                raise ValueError("job request is missing")
            record = self._advance(record, JobStage.EXTRACTING_AUDIO, 2, "extracting selected audio")
            workspace = TemporaryDirectory(prefix="contextual-engine-")
            job_dir = Path(workspace.name)

            def emit(event: Progress) -> None:
                current = self.store.load(job_id)
                self._report(
                    current,
                    percent=max(2, min(15, event.percent)),
                    message=event.message,
                )

            media = prepare_audio(
                record.request.source,
                job_dir,
                record.request.audio_stream_index,
                record.request.requested_context,
                token,
                emit,
            )
            token.raise_if_cancelled()
            context = resolve_content_context(
                record.request.requested_context, media.container_kind
            )
            record = self.store.load(job_id)
            record = self._advance(
                record,
                JobStage.DETECTING_LANGUAGE,
                15,
                "screening vocals and checking English audio",
                prepared_media=media,
                resolved_content_context=context,
            )
            record = self._advance(
                record,
                JobStage.TRANSCRIBING,
                20,
                "transcribing eligible English audio",
            )

            def checkpoint(window: WindowResult) -> None:
                token.raise_if_cancelled()
                self.store.append_transcription_window(job_id, window)
                token.raise_if_cancelled()
                for segment in window.segments:
                    token.raise_if_cancelled()
                    self.store.append_transcript(job_id, segment)
                    token.raise_if_cancelled()

            def transcribe_progress(completed: int, total: int) -> None:
                current = self.store.load(job_id)
                ratio = completed / total if total else 1.0
                self._report(
                    current,
                    percent=20 + ratio * 48,
                    message=f"completed audio unit {completed} of {total}",
                    completed_units=completed,
                    total_units=total,
                    current_chunk=completed,
                    total_chunks=total,
                )

            run = transcribe_media(
                media,
                transcriber=self.models.transcriber,
                screener=self.models.screener,
                work_dir=job_dir,
                token=token,
                checkpoint=checkpoint,
                on_progress=transcribe_progress,
                # Past this, the non-English cap is exceeded whatever follows.
                non_english_stop_seconds=MAX_NON_ENGLISH_SHARE * media.duration_seconds,
            )
            token.raise_if_cancelled()
            # ASR and screening weights are not needed during retrieval or generation.
            self.models.release()
            detected_languages = {
                window.language
                for window in run.windows
                if window.language is not None
            }
            record = self._advance(
                self.store.load(job_id),
                JobStage.ANALYSING,
                70,
                "classifying transcript evidence",
                language=next(iter(detected_languages)) if len(detected_languages) == 1 else None,
            )
            analysis_windows = analysis_windows_from_run(run)
            diagnostics_count = 0
            diagnostic_budget = len(analysis_windows) * 9 + 2
            active_verifier_range: TimeRange | None = None
            self._report(
                self.store.load(job_id), percent=70,
                message="classifying transcript evidence", completed_units=0,
                total_units=diagnostic_budget,
            )

            def diagnostic(kind: str, payload: Mapping[str, object]) -> None:
                nonlocal diagnostics_count
                # A completed model generation remains diagnostic evidence even
                # when cancellation arrived during that call.
                if kind != "verifier_attempt":
                    token.raise_if_cancelled()
                self.store.append_analysis_diagnostic(job_id, kind, payload)
                diagnostics_count += 1
                current = self.store.load(job_id)
                self._report(
                    current,
                    percent=70 + min(20, 20 * diagnostics_count / diagnostic_budget),
                    message="retrieving and validating transcript categories",
                    completed_units=diagnostics_count,
                    total_units=diagnostic_budget,
                )
                if kind != "verifier_attempt":
                    token.raise_if_cancelled()

            def verifier_progress(time_range, completed_calls, call_limit, call_state):
                nonlocal active_verifier_range
                active_verifier_range = time_range
                index = next(index for index, window in enumerate(analysis_windows, 1)
                             if window.time_range == time_range)
                self._report(
                    self.store.load(job_id),
                    percent=70 + min(20, 20 * diagnostics_count / diagnostic_budget),
                    message=(f"transcript verification: window {index} of {len(analysis_windows)} "
                             f"({time_range.start_seconds:.3f}–{time_range.end_seconds:.3f}s), "
                             f"{completed_calls}/{call_limit} calls completed; {call_state}"),
                    completed_units=diagnostics_count, total_units=diagnostic_budget,
                    current_chunk=index, total_chunks=len(analysis_windows),
                )

            def verifier_checkpoint(attempt):
                assert active_verifier_range is not None
                identity = self.models.text_model_identity
                diagnostic("verifier_attempt", {
                    "time_range": active_verifier_range.model_dump(mode="json"),
                    "policy_version": VERIFIER_POLICY,
                    "model": {"repository": identity.repository, "revision": identity.revision,
                              "manifest_sha256": identity.manifest_sha256},
                    "attempt": attempt.model_dump(mode="json"),
                })

            classifications = (
                classify_and_verify_transcription_run(
                    run,
                    taxonomy_index=self.taxonomy_index,
                    generator=self.models.text_generator,
                    token=token,
                    candidate_checkpoint=lambda item: diagnostic(
                        "candidates",
                        {
                            "time_range": item.time_range.model_dump(mode="json"),
                            "candidates": [
                                {
                                    "taxonomy_id": candidate.taxonomy_id,
                                    "name": candidate.name,
                                    "path": candidate.path,
                                    "similarity": candidate.similarity,
                                }
                                for candidate in item.candidates
                            ],
                        },
                    ),
                    classification_attempt_checkpoint=lambda item: diagnostic(
                        "attempt", item.model_dump(mode="json")
                    ),
                    verifier_attempt_checkpoint=verifier_checkpoint,
                    on_verifier_progress=verifier_progress,
                    # This durable marker records the configured verifier policy,
                    # including failed verification. error_code remains decisive;
                    # the in-memory ClassificationResult is not changed here.
                    result_checkpoint=lambda item: diagnostic(
                        "result", {**item.model_dump(mode="json"),
                                   "verification_policy_version": VERIFIER_POLICY}
                    ),
                )
                if analysis_windows
                else ()
            )
            token.raise_if_cancelled()
            self._report(
                self.store.load(job_id),
                percent=93,
                message="transcript classification complete",
            )
            self.models.release()
            entries = self._timeline_entries(run, classifications, media)
            for entry in entries:
                token.raise_if_cancelled()
                self.store.append_text_timeline(job_id, entry)
                token.raise_if_cancelled()
            assessment = assess_coverage(
                run, classifications, audio_seconds=media.duration_seconds
            )
            diagnostic("coverage_policy", assessment.model_dump(mode="json"))
            if assessment.error_code is not None:
                raise CoverageFailure(assessment.error_code)
            token.raise_if_cancelled()
            taxonomy = load_taxonomy()
            decision = evaluate_topic_policy_v3(
                run, classifications, audio_seconds=media.duration_seconds, taxonomy=taxonomy
            )
            token.raise_if_cancelled()
            diagnostic("topic_policy", decision.model_dump(mode="json"))
            token.raise_if_cancelled()
            if decision.outcome == "insufficient":
                raise InsufficientTopicEvidence()
            token.raise_if_cancelled()
            coverage = self._coverage(assessment, decision)
            result = aggregate_topic_sufficient(
                entries, decision=decision, coverage=coverage, taxonomy=taxonomy
            )
            token.raise_if_cancelled()
            record = self._advance(
                self.store.load(job_id),
                JobStage.EXPORTING,
                95,
                "validating contextual result",
                prepared_media=None,
            )
            content = build_content(
                result,
                metadata=media.source_metadata,
                context=context,
                duration_seconds=media.duration_seconds,
            )
            token.raise_if_cancelled()
            token.run_if_not_cancelled(lambda: self.store.promote_result(job_id, content))
        except JobCancelled:
            self._terminalize(job_id, JobStage.CANCELLED, "cancelled")
        except InsufficientTopicEvidence:
            self._terminalize(job_id, JobStage.FAILED, "INSUFFICIENT_TOPIC_EVIDENCE")
        except CoverageFailure as exc:
            self._terminalize(job_id, JobStage.FAILED, exc.code)
        except Exception as exc:
            self._terminalize(job_id, JobStage.FAILED, type(exc).__name__)
        finally:
            try:
                self.models.release()
            except Exception:
                pass
            if workspace is not None:
                workspace.cleanup()

    def _timeline_entries(self, run, classifications, media: PreparedMedia):
        entries: list[TextTimelineEntry] = []
        windows = analysis_windows_from_run(run)
        for window, result in zip(windows, classifications, strict=True):
            if result.payload is None:
                continue
            cited_indices = {
                index
                for citation in result.category_citations
                for index in citation.segment_indices
            }
            evidence = _union_ranges(tuple(
                segment.time_range for index, segment in enumerate(window.segments)
                if index in cited_indices
            ))
            if not evidence:
                continue
            entries.append(
                TextTimelineEntry(
                    time_range=result.time_range,
                    payload=result.payload,
                    evidence_ranges=evidence,
                    provenance=AudioTranscriptProvenance(
                        audio_stream_index=media.selected_audio_stream,
                        transcriber="local-mlx",
                    ),
                )
            )
        return tuple(entries)

    @staticmethod
    def _coverage(assessment: CoverageAssessment, decision) -> CoverageSummary:
        return CoverageSummary(
            source_seconds=assessment.audio_seconds,
            analysed_seconds=min(assessment.audio_seconds, decision.transcript_seconds),
            failed_ranges=assessment.failed_ranges,
            excluded_ranges=assessment.excluded_ranges,
        )

    def _report(self, record: JobRecord, *, percent: float, message: str, **units: object) -> JobRecord:
        progress = record.progress.advance(
            stage=record.status,
            percent=percent,
            message=message,
            **units,
        )
        updated = record.model_copy(update={"progress": progress})
        self.store.save(updated)
        return updated

    def _advance(self, record: JobRecord, stage: JobStage, percent: float, message: str, **values) -> JobRecord:
        progress = record.progress.advance(stage=stage, percent=percent, message=message)
        updated = record.model_copy(update={"status": stage, "progress": progress, **values})
        self.store.save(updated)
        return updated

    def _terminalize(self, job_id: str, target: JobStage, message: str) -> None:
        try:
            record = self.store.load(job_id)
            if record.status in {JobStage.COMPLETE, JobStage.CANCELLED, JobStage.FAILED}:
                return
            if record.status is not JobStage.CANCELLING:
                record = self._advance(
                    record,
                    JobStage.CANCELLING,
                    record.progress.percent,
                    "cancellation requested" if target is JobStage.CANCELLED else "job failed",
                )
            progress = record.progress.advance(stage=target, percent=record.progress.percent, message=message)
            self.store.save(
                record.model_copy(
                    update={
                        "status": target,
                        "progress": progress,
                        "error": message,
                        "prepared_media": None,
                    }
                )
            )
        except Exception:
            pass
