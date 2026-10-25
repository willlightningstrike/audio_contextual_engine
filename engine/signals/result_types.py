"""Immutable, versioned MLX-free whole-file and timeline result values."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from engine.domain import TimeRange
from engine.signals.text_types import ClassificationPayload


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


def _duration(ranges: tuple[TimeRange, ...]) -> float:
    """Measure a range union, so overlapping transcript spans count once."""
    intervals = sorted((item.start_seconds, item.end_seconds) for item in ranges)
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _validate_bounded_ranges(
    ranges: tuple[TimeRange, ...], *, source_seconds: float, label: str
) -> None:
    for item in ranges:
        if item.end_seconds > source_seconds:
            raise ValueError(f"{label} range exceeds source duration")


class CoverageSummary(_StrictFrozenModel):
    source_seconds: float = Field(ge=0)
    analysed_seconds: float = Field(ge=0)
    failed_ranges: tuple[TimeRange, ...] = ()
    excluded_ranges: tuple[TimeRange, ...] = ()

    @model_validator(mode="after")
    def _valid_coverage_bounds(self) -> CoverageSummary:
        if self.analysed_seconds > self.source_seconds:
            raise ValueError("analysed_seconds cannot exceed source_seconds")
        _validate_bounded_ranges(
            self.failed_ranges, source_seconds=self.source_seconds, label="failed"
        )
        _validate_bounded_ranges(
            self.excluded_ranges, source_seconds=self.source_seconds, label="excluded"
        )
        return self


class AudioTranscriptProvenance(_StrictFrozenModel):
    source: Literal["audio_transcript"] = "audio_transcript"
    language: Literal["en"] = "en"
    audio_stream_index: StrictInt | None = Field(default=None, ge=0)
    transcriber: str | None = None


class TextTimelineEntry(_StrictFrozenModel):
    time_range: TimeRange
    payload: ClassificationPayload
    evidence_ranges: tuple[TimeRange, ...] = Field(min_length=1)
    provenance: AudioTranscriptProvenance

    @model_validator(mode="after")
    def _evidence_is_inside_unit(self) -> TextTimelineEntry:
        if self.time_range.end_seconds <= self.time_range.start_seconds:
            raise ValueError("classification unit must have positive duration")
        for item in self.evidence_ranges:
            if (
                item.end_seconds <= item.start_seconds
                or item.start_seconds < self.time_range.start_seconds
                or item.end_seconds > self.time_range.end_seconds
            ):
                raise ValueError("evidence range must be positive and inside its classification unit")
        return self


class KeywordAggregate(_StrictFrozenModel):
    term: str = Field(min_length=1)
    support_seconds: float = Field(gt=0)
    model_scores: tuple[float, ...] = Field(min_length=1)

    @field_validator("term")
    @classmethod
    def _nonblank_term(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("keyword term must not be blank")
        return value

    @field_validator("model_scores")
    @classmethod
    def _scores_are_original_model_values(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("keyword model scores must be within [0, 1]")
        return values


class EntityAggregate(_StrictFrozenModel):
    text: str = Field(min_length=1)
    type: str = Field(min_length=1)
    mention_count: StrictInt = Field(ge=1)
    support_seconds: float = Field(gt=0)

    @field_validator("text", "type")
    @classmethod
    def _nonblank_entity_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity text and type must not be blank")
        return value


class AggregateSentiment(_StrictFrozenModel):
    label: Literal["negative", "neutral", "positive"]
    score: float = Field(ge=-1, le=1)
    aggregation_convention: Literal["duration_weighted_signed_score_v1"] = (
        "duration_weighted_signed_score_v1"
    )

    @model_validator(mode="after")
    def _label_follows_signed_score(self) -> AggregateSentiment:
        expected = "negative" if self.score < 0 else "positive" if self.score > 0 else "neutral"
        if self.label != expected:
            raise ValueError("sentiment label must follow the sign of the signed score")
        return self


class AggregateResult(_StrictFrozenModel):
    primary_category: str = Field(min_length=1)
    secondary_categories: tuple[str, ...] = Field(max_length=2)
    # Auditable role-specific support durations, not confidence or calibrated truth.
    primary_role_support_seconds: tuple[tuple[str, float], ...]
    secondary_role_support_seconds: tuple[tuple[str, float], ...] = ()
    keywords: tuple[KeywordAggregate, ...] = Field(max_length=20)
    entities: tuple[EntityAggregate, ...] = Field(max_length=50)
    # None only for a declared result with no verified transcript window.
    sentiment: AggregateSentiment | None
    coverage: CoverageSummary
    policy_version: Literal[
        "english-audio-aggregation-v1", "english-audio-aggregation-v2",
        "english-audio-aggregation-v3",
    ] = "english-audio-aggregation-v1"
    # v3 only: how the categories were chosen and the support each was given.
    topic_structure: Literal["single", "parent_rollup", "multi_topic", "declared_music"] | None = None
    selected_support_seconds: tuple[tuple[str, float], ...] = ()
    # declared_music only: user-declared categories, which lead and have no support.
    declared_categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _category_roles_and_support_are_valid(self) -> AggregateResult:
        categories = (self.primary_category, *self.secondary_categories)
        if any(not value.strip() for value in categories):
            raise ValueError("category IDs must not be blank")
        if len(categories) != len(set(categories)):
            raise ValueError("primary and secondary category IDs must be unique")
        for label, values in (
            ("primary", self.primary_role_support_seconds),
            ("secondary", self.secondary_role_support_seconds),
        ):
            ids = tuple(item[0] for item in values)
            if len(ids) != len(set(ids)) or any(not item.strip() for item in ids):
                raise ValueError(f"{label} support IDs must be unique non-blank strings")
            if any(not value > 0 for _, value in values):
                raise ValueError(f"{label} support durations must be positive")
        declared = self.topic_structure == "declared_music"
        if declared != bool(self.declared_categories):
            raise ValueError("only declared_music results record declared categories")
        if self.sentiment is None and not declared:
            raise ValueError("sentiment is required unless categories were declared")
        if not declared and not self.primary_role_support_seconds:
            raise ValueError("primary-role support is required")
        if self.policy_version == "english-audio-aggregation-v3":
            if self.topic_structure is None:
                raise ValueError("v3 results require a topic structure")
            if categories[:len(self.declared_categories)] != self.declared_categories:
                raise ValueError("declared categories must lead the category list")
            supported = categories[len(self.declared_categories):]
            if tuple(item[0] for item in self.selected_support_seconds) != supported:
                raise ValueError("selected support must list the chosen categories in order")
            if any(not value > 0 for _, value in self.selected_support_seconds):
                raise ValueError("selected support durations must be positive")
            return self
        if self.topic_structure is not None or self.selected_support_seconds:
            raise ValueError("only v3 results record a topic structure")
        if self.primary_category not in dict(self.primary_role_support_seconds):
            raise ValueError("primary category must have positive primary-role support")
        if not set(self.secondary_categories).issubset(
            dict(self.secondary_role_support_seconds)
        ):
            raise ValueError("secondary categories must have positive secondary-role support")
        return self
