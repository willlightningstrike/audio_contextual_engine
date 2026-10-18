"""MLX-free contracts for bounded speech recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from engine.cancel import CancellationToken
from engine.domain import TimeRange, TranscriptSegment


class Transcriber(Protocol):
    def detect_language_probabilities(
        self, audio: Path, window: TimeRange, token: CancellationToken
    ) -> Mapping[str, float]: ...

    def transcribe_window(
        self, audio: Path, window: TimeRange, language: str, token: CancellationToken
    ) -> list[TranscriptSegment]: ...
