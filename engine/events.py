"""Immutable, pipeline-owned progress snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
import math

from pydantic import BaseModel, ConfigDict, Field

from engine.lifecycle import JobStage


class Progress(BaseModel):
    """A durable status snapshot; callers supply pipeline-computed percentages."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    stage: JobStage
    percent: float = Field(ge=0, le=100)
    message: str = Field(min_length=1)
    revision: int = Field(default=0, ge=0)
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    current_file: str | None = None
    file_index: int | None = Field(default=None, ge=0)
    current_chunk: int | None = Field(default=None, ge=0)
    total_chunks: int | None = Field(default=None, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def advance(self, *, stage: JobStage, percent: float, message: str, **updates: object) -> Progress:
        """Return the next revision without allowing percentage regression."""
        if not math.isfinite(percent):
            raise ValueError("percent must be finite")
        bounded_percent = min(100.0, max(0.0, percent))
        payload = self.model_dump()
        payload.update(updates)
        payload.update(
            stage=stage,
            percent=max(self.percent, bounded_percent),
            message=message,
            revision=self.revision + 1,
            updated_at=datetime.now(UTC),
        )
        return type(self).model_validate(payload)
