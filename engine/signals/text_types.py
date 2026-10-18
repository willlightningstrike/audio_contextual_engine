"""Strict MLX-free values for constrained contextual text classification."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from engine.domain import TimeRange


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class GenerationResult(_StrictFrozenModel):
    raw: str
    finish_reason: str = Field(min_length=1)
    seconds: float = Field(ge=0)
    token_count: int = Field(ge=0)


class KeywordValue(_StrictFrozenModel):
    term: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("term")
    @classmethod
    def _nonblank_term(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("keyword term must not be blank")
        return value


class EntityValue(_StrictFrozenModel):
    text: str = Field(min_length=1)
    type: str = Field(min_length=1)
    mention_count: int = Field(ge=1)

    @field_validator("text", "type")
    @classmethod
    def _nonblank_entity_fields(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity text and type must not be blank")
        return value


class SentimentValue(_StrictFrozenModel):
    label: Literal["negative", "neutral", "positive"]
    score: float = Field(ge=-1, le=1)


class ClassificationPayload(_StrictFrozenModel):
    primary_category: str = Field(min_length=1)
    secondary_categories: tuple[str, ...] = Field(max_length=2)
    keywords: tuple[KeywordValue, ...] = Field(max_length=8)
    entities: tuple[EntityValue, ...] = Field(max_length=8)
    sentiment: SentimentValue

    @model_validator(mode="after")
    def _has_unique_category_roles(self) -> ClassificationPayload:
        selected = (self.primary_category, *self.secondary_categories)
        if any(not value.strip() for value in selected):
            raise ValueError("category IDs must not be blank")
        if len(selected) != len(set(selected)):
            raise ValueError("primary and secondary category IDs must be unique")
        return self


class CategoryCitation(_StrictFrozenModel):
    category_id: str = Field(min_length=1)
    segment_indices: tuple[Annotated[StrictInt, Field(ge=0)], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_indices(self) -> CategoryCitation:
        if len(self.segment_indices) != len(set(self.segment_indices)):
            raise ValueError("duplicate cited segment index")
        return self


class GenerationAttempt(_StrictFrozenModel):
    attempt: int = Field(ge=1, le=2)
    stage: Literal["classification", "primary", "payload"] = "classification"
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw: str | None = None
    finish_reason: str | None = None
    seconds: float | None = Field(default=None, ge=0)
    token_count: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    reason: str | None = None


class ClassificationResult(_StrictFrozenModel):
    time_range: TimeRange
    payload: ClassificationPayload | None
    attempts: tuple[GenerationAttempt, ...]
    error_code: str | None = None
    category_citations: tuple[CategoryCitation, ...] = ()
    policy_version: Literal["english-audio-topic-sufficiency-v1"] | None = None
    verification_policy_version: Literal[
        "english-transcript-verifier-v1", "english-transcript-verifier-v2"
    ] | None = None

    @model_validator(mode="after")
    def _is_success_or_explicit_error(self) -> ClassificationResult:
        if (self.payload is None) == (self.error_code is None):
            raise ValueError("classification result must contain payload or error, not both")
        if not self.attempts:
            raise ValueError("classification result must retain at least one attempt")
        return self
