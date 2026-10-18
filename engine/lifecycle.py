"""Shared MLX-free lifecycle values used by durable records and events."""

from enum import Enum


class JobStage(str, Enum):
    QUEUED = "queued"
    EXTRACTING_AUDIO = "extracting_audio"
    DETECTING_LANGUAGE = "detecting_language"
    TRANSCRIBING = "transcribing"
    ANALYSING = "analysing"
    EXPORTING = "exporting"
    CANCELLING = "cancelling"
    BATCH_PROCESSING = "batch_processing"
    CANCELLED = "cancelled"
    COMPLETE = "complete"
    FAILED = "failed"
