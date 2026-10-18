"""Shared local text/audio adapter backed by one resident Qwen2-Audio model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from engine.cancel import CancellationToken
from engine.domain import TimeRange
from engine.models import collect_mlx_evidence, require_resolved_local_directory
from engine.models.asr import _require_bounded_wav


MAX_AUDIO_SECONDS = 30.0


def _default_load(path: Path) -> Any:
    from mlx_audio.stt import load

    return load(str(path), lazy=False, strict=True)


def _default_text_generate(
    language_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    verbose: bool,
) -> str:
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    return generate(
        language_model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temperature),
        verbose=verbose,
    )


class MlxTextAudioAdapter:
    """Serve text and bounded-audio prompts from one loaded Qwen2-Audio object."""

    def __init__(
        self,
        snapshot_path: Path,
        *,
        load_fn: Callable[[Path], Any] = _default_load,
        text_generate_fn: Callable[..., Any] = _default_text_generate,
    ) -> None:
        self._snapshot_path = require_resolved_local_directory(snapshot_path)
        self._load_fn = load_fn
        self._text_generate_fn = text_generate_fn
        self._loaded: Any | None = None
        self._last_generation_result: Any | None = None

    @property
    def model(self) -> Any:
        if self._loaded is None:
            self._loaded = self._load_fn(self._snapshot_path)
        return self._loaded

    def generate_text(self, prompt: str, *, max_tokens: int, token: CancellationToken) -> str:
        self._validate_request(prompt, max_tokens)
        token.raise_if_cancelled()
        model = self.model
        processor = getattr(model, "_processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        language_model = getattr(model, "language_model", None)
        if tokenizer is None or language_model is None:
            raise ValueError("text/audio model does not expose its resident language model and tokenizer")
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        result = self._text_generate_fn(
            language_model,
            tokenizer,
            formatted,
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
        )
        self._last_generation_result = result
        token.raise_if_cancelled()
        return self._text(result)

    def generate_audio(
        self,
        audio: Path,
        window: TimeRange,
        prompt: str,
        *,
        max_tokens: int,
        token: CancellationToken,
    ) -> str:
        self._validate_request(prompt, max_tokens)
        if window.end_seconds - window.start_seconds > MAX_AUDIO_SECONDS:
            raise ValueError("model audio windows must not exceed 30 seconds")
        bounded_audio = _require_bounded_wav(audio)
        token.raise_if_cancelled()
        result = self.model.generate(
            str(bounded_audio),
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
        )
        self._last_generation_result = result
        token.raise_if_cancelled()
        return self._text(result)

    @staticmethod
    def _validate_request(prompt: str, max_tokens: int) -> None:
        if not isinstance(prompt, str) or not prompt.strip() or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("prompt must be non-empty and max_tokens must be positive")

    @staticmethod
    def _text(result: Any) -> str:
        if not isinstance(result, str):
            result = getattr(result, "text", result)
        if not isinstance(result, str):
            raise ValueError("invalid text/audio model output")
        return result

    def runtime_evidence(self) -> dict[str, Any]:
        model = self.model
        return collect_mlx_evidence(
            model,
            {
                "audio_tower_input_weight": model.audio_tower.conv1.weight,
                "language_model_embedding_weight": model.language_model.model.embed_tokens.weight,
            },
        )
