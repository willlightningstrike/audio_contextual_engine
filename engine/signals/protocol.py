"""MLX-free contracts used by signal services."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from engine.cancel import CancellationToken
from engine.domain import TimeRange
from engine.signals.text_types import GenerationResult
from engine.transcribe.types import ScreenResult


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TextGenerator(Protocol):
    def generate_text(self, prompt: str, *, max_tokens: int, token: CancellationToken) -> str: ...


class ConstrainedTextGenerator(Protocol):
    def generate_json(
        self,
        prompt: str,
        schema: dict,
        *,
        max_tokens: int,
        token: CancellationToken,
    ) -> GenerationResult: ...


class AudioGenerator(Protocol):
    def generate_audio(
        self,
        audio: Path,
        window: TimeRange,
        prompt: str,
        *,
        max_tokens: int,
        token: CancellationToken,
    ) -> str: ...


class VocalScreener(Protocol):
    def screen(self, audio: Path, token: CancellationToken) -> ScreenResult: ...
