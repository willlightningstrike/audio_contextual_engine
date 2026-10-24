"""MLX-free values exchanged across Contextual Engine application boundaries."""

from __future__ import annotations

import re
import math
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from engine.events import Progress
from engine.lifecycle import JobStage


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)


def _deep_freeze(value: Any) -> Any:
    """Recursively make JSON-shaped structures safe inside frozen records."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise ValueError("JSON floats must be finite")
    raise ValueError("JSON values must be mappings, sequences, or JSON primitives")


def _deep_thaw(value: Any) -> Any:
    """Produce ordinary JSON-compatible values for deterministic persistence."""
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_deep_thaw(item) for item in value)
    return value


def _default_progress(status: JobStage) -> Progress:
    return Progress(
        stage=status,
        percent=100.0 if status is JobStage.COMPLETE else 0.0,
        message=status.value,
    )


class ContainerKind(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


class ContentContext(int, Enum):
    """OpenRTB content.context values, separate from a media container type."""

    VIDEO = 1
    MUSIC = 3
    OTHER = 6
    UNKNOWN = 7


class TimeRange(_FrozenModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def _is_ordered(self) -> TimeRange:
        if self.end_seconds < self.start_seconds:
            raise ValueError("end_seconds must be greater than or equal to start_seconds")
        return self


class AudioStream(_FrozenModel):
    index: int = Field(ge=0)
    codec_name: str = Field(min_length=1)
    language: str | None = None
    channels: int | None = Field(default=None, ge=1)
    sample_rate: int | None = Field(default=None, ge=1)


class SourceMetadata(_FrozenModel):
    title: str | None = None
    url: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0)


class PreparedMedia(_FrozenModel):
    source_path: Path
    audio_path: Path
    container_kind: ContainerKind
    content_context_hint: ContentContext
    selected_audio_stream: int = Field(ge=0)
    available_audio_streams: tuple[AudioStream, ...]
    duration_seconds: float = Field(ge=0)
    source_metadata: SourceMetadata = Field(default_factory=SourceMetadata)

    @model_validator(mode="after")
    def _has_unique_streams_and_selection(self) -> PreparedMedia:
        indexes = tuple(stream.index for stream in self.available_audio_streams)
        if len(indexes) != len(set(indexes)):
            raise ValueError("available_audio_streams must have unique indexes")
        if self.selected_audio_stream not in indexes:
            raise ValueError("selected_audio_stream must identify an available audio stream")
        return self


class TranscriptSegment(_FrozenModel):
    time_range: TimeRange
    text: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    language: str | None = None


# Content Taxonomy 3.1 has legitimate alphanumeric IDs (for example, 80DV8O).
# This intentionally corrects the phase plan's stale digits-only requirement.
_TAXONOMY_ID = re.compile(r"^[A-Za-z0-9]+$")


class CategorySignal(_FrozenModel):
    taxonomy_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _has_safe_taxonomy_id(self) -> CategorySignal:
        if not _TAXONOMY_ID.fullmatch(self.taxonomy_id):
            raise ValueError("taxonomy_id must be a non-blank alphanumeric Content Taxonomy ID")
        return self


class KeywordSignal(_FrozenModel):
    keyword: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class EntitySignal(_FrozenModel):
    text: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class SentimentSignal(_FrozenModel):
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class AcousticSignals(_FrozenModel):
    genres: tuple[str, ...] = ()
    moods: tuple[str, ...] = ()
    vocal_presence: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class SignalSet(_FrozenModel):
    categories: tuple[CategorySignal, ...] = ()
    keywords: tuple[KeywordSignal, ...] = ()
    entities: tuple[EntitySignal, ...] = ()
    sentiment: SentimentSignal | None = None
    acoustic: AcousticSignals | None = None


class TimelineEntry(_FrozenModel):
    time_range: TimeRange
    signals: SignalSet


class JobRequest(_FrozenModel):
    source: Path
    requested_language: str | None = None
    requested_context: ContentContext | None = None
    audio_stream_index: int | None = Field(default=None, ge=0)

    @field_validator("audio_stream_index", mode="before")
    @classmethod
    def _requires_a_real_stream_index(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("audio_stream_index must be a non-negative integer")
        return value


class ModelProvenance(_FrozenModel):
    app_version: str = Field(min_length=1)
    package_versions: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}))
    model_revisions: Mapping[str, str] = Field(default_factory=lambda: MappingProxyType({}))
    standards_revision: str | None = None

    @field_validator("package_versions", "model_revisions", mode="after")
    @classmethod
    def _freeze_versions(cls, versions: Mapping[str, str]) -> Mapping[str, str]:
        return _deep_freeze(versions)

    @field_serializer("package_versions", "model_revisions")
    def _serialize_versions(self, versions: Mapping[str, str]) -> dict[str, str]:
        return _deep_thaw(versions)


class JobRecord(_FrozenModel):
    schema_version: Literal[2] = 2
    evidence_mode: Literal["transcript_only"] | None = None
    id: str = Field(min_length=1)
    status: JobStage = JobStage.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error: str | None = None
    request: JobRequest | None = None
    prepared_media: PreparedMedia | None = None
    resolved_content_context: ContentContext | None = None
    language: str | None = None
    transcript: tuple[TranscriptSegment, ...] = ()
    timeline: tuple[TimelineEntry, ...] = ()
    text_timeline: tuple[Mapping[str, Any], ...] = ()
    aggregate_signals: SignalSet | None = None
    progress: Progress = Field(default=None, validate_default=True)
    ortb_content: Mapping[str, Any] | None = None
    provenance: ModelProvenance | None = None

    @field_validator("ortb_content", mode="after")
    @classmethod
    def _freeze_ortb_content(cls, content: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        return None if content is None else _deep_freeze(content)

    @field_validator("text_timeline", mode="after")
    @classmethod
    def _freeze_text_timeline(
        cls, entries: tuple[Mapping[str, Any], ...]
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(_deep_freeze(entry) for entry in entries)

    @field_serializer("ortb_content")
    def _serialize_ortb_content(self, content: Mapping[str, Any] | None) -> Any:
        return None if content is None else _deep_thaw(content)

    @field_serializer("text_timeline")
    def _serialize_text_timeline(self, entries: tuple[Mapping[str, Any], ...]) -> Any:
        return tuple(_deep_thaw(entry) for entry in entries)

    @field_validator("progress", mode="before")
    @classmethod
    def _derives_missing_progress(cls, progress: object, info: ValidationInfo) -> object:
        status = info.data.get("status", JobStage.QUEUED)
        return _default_progress(status) if progress is None else progress

    @model_validator(mode="after")
    def _keeps_status_and_progress_coherent(self) -> JobRecord:
        if self.progress.stage is not self.status:
            raise ValueError("progress.stage must match job status")
        return self

    @model_validator(mode="after")
    def _allows_export_only_when_complete(self) -> JobRecord:
        if self.ortb_content is not None and self.status is not JobStage.COMPLETE:
            raise ValueError("ortb_content is only permitted for complete jobs")
        return self


class BatchRecord(_FrozenModel):
    """Durable batch-processing state plus independent child job outcomes.

    COMPLETE means every child finished processing; the outcome counts decide
    whether every child actually succeeded.
    """

    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    status: JobStage = JobStage.QUEUED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_directory: Path
    progress: Progress = Field(default=None, validate_default=True)
    child_job_ids: tuple[str, ...] = ()
    active_child_job_id: str | None = None
    child_outcomes: Mapping[str, JobStage] = Field(default_factory=lambda: MappingProxyType({}))
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    cancelled_count: int = Field(default=0, ge=0)
    error: str | None = None

    @field_validator("child_outcomes", mode="after")
    @classmethod
    def _freeze_child_outcomes(cls, outcomes: Mapping[str, JobStage]) -> Mapping[str, JobStage]:
        return MappingProxyType(dict(outcomes))

    @field_serializer("child_outcomes")
    def _serialize_child_outcomes(self, outcomes: Mapping[str, JobStage]) -> dict[str, str]:
        return {key: value.value for key, value in outcomes.items()}

    @field_validator("progress", mode="before")
    @classmethod
    def _derives_missing_progress(cls, progress: object, info: ValidationInfo) -> object:
        status = info.data.get("status", JobStage.QUEUED)
        return _default_progress(status) if progress is None else progress

    @model_validator(mode="after")
    def _keeps_status_and_progress_coherent(self) -> BatchRecord:
        if self.progress.stage is not self.status:
            raise ValueError("progress.stage must match batch status")
        if len(self.child_job_ids) != len(set(self.child_job_ids)):
            raise ValueError("child_job_ids must be unique")
        if not set(self.child_outcomes).issubset(self.child_job_ids):
            raise ValueError("child outcomes must belong to this batch")
        if any(
            outcome not in {JobStage.COMPLETE, JobStage.FAILED, JobStage.CANCELLED}
            for outcome in self.child_outcomes.values()
        ):
            raise ValueError("child outcomes must be terminal job states")
        expected = (
            sum(outcome is JobStage.COMPLETE for outcome in self.child_outcomes.values()),
            sum(outcome is JobStage.FAILED for outcome in self.child_outcomes.values()),
            sum(outcome is JobStage.CANCELLED for outcome in self.child_outcomes.values()),
        )
        if expected != (self.success_count, self.failure_count, self.cancelled_count):
            raise ValueError("persisted batch counts must match child outcomes")
        if self.active_child_job_id is not None and self.active_child_job_id not in self.child_job_ids:
            raise ValueError("active child must belong to this batch")
        if self.status is JobStage.COMPLETE and len(self.child_outcomes) != len(self.child_job_ids):
            raise ValueError("completed batch must retain every child outcome")
        if self.status is JobStage.CANCELLED and len(self.child_outcomes) != len(self.child_job_ids):
            raise ValueError("cancelled batch must retain every child outcome")
        return self

    @property
    def all_children_succeeded(self) -> bool:
        return (
            self.status is JobStage.COMPLETE
            and self.success_count == len(self.child_job_ids)
            and self.failure_count == 0
            and self.cancelled_count == 0
        )
