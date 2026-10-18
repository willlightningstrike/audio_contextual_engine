"""Experimental local Qwen3-Omni text/audio adapter using MLX-VLM streaming."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from engine.cancel import CancellationToken
from engine.domain import TimeRange
from engine.models import collect_mlx_evidence, require_resolved_local_directory
from engine.models.asr import _require_bounded_wav, _require_bounded_window


def _default_load(path: Path) -> tuple[Any, Any]:
    from mlx_vlm import load

    return load(str(path), lazy=False, strict=True)


def _default_chat_template(
    processor: Any, config: Any, prompt: str, **kwargs: Any
) -> str:
    from mlx_vlm import apply_chat_template

    return apply_chat_template(processor, config, prompt, **kwargs)


def _default_stream_generate(
    model: Any, processor: Any, prompt: str, **kwargs: Any
) -> Iterator[Any]:
    from mlx_vlm import stream_generate

    return stream_generate(model, processor, prompt, **kwargs)


class Qwen3OmniAdapter:
    """Serve text and one bounded WAV from one resident Qwen3-Omni model."""

    def __init__(
        self,
        snapshot_path: Path,
        *,
        load_fn: Callable[[Path], tuple[Any, Any]] = _default_load,
        chat_template_fn: Callable[..., str] = _default_chat_template,
        stream_generate_fn: Callable[..., Iterator[Any]] = _default_stream_generate,
    ) -> None:
        self._snapshot_path = require_resolved_local_directory(snapshot_path)
        self._load_fn = load_fn
        self._chat_template_fn = chat_template_fn
        self._stream_generate_fn = stream_generate_fn
        self._loaded: tuple[Any, Any] | None = None

    @property
    def _model_and_processor(self) -> tuple[Any, Any]:
        if self._loaded is None:
            loaded = self._load_fn(self._snapshot_path)
            if not isinstance(loaded, tuple) or len(loaded) != 2:
                raise ValueError("Qwen3-Omni loader must return model and processor")
            self._loaded = loaded
        return self._loaded

    @property
    def model(self) -> Any:
        return self._model_and_processor[0]

    @property
    def processor(self) -> Any:
        return self._model_and_processor[1]

    def generate_text(
        self, prompt: str, *, max_tokens: int, token: CancellationToken
    ) -> str:
        return self._generate(
            prompt,
            audio=None,
            max_tokens=max_tokens,
            token=token,
        )

    def generate_audio(
        self,
        audio: Path,
        window: TimeRange,
        prompt: str,
        *,
        max_tokens: int,
        token: CancellationToken,
    ) -> str:
        _require_bounded_window(window)
        bounded_audio = _require_bounded_wav(audio)
        return self._generate(
            prompt,
            audio=bounded_audio,
            max_tokens=max_tokens,
            token=token,
        )

    def _generate(
        self,
        prompt: str,
        *,
        audio: Path | None,
        max_tokens: int,
        token: CancellationToken,
    ) -> str:
        self._validate_request(prompt, max_tokens)
        token.raise_if_cancelled()
        model, processor = self._model_and_processor
        token.raise_if_cancelled()
        formatted = self._chat_template_fn(
            processor,
            model.config,
            prompt,
            num_audios=0 if audio is None else 1,
            enable_thinking=False,
        )
        if not isinstance(formatted, str):
            raise ValueError("Qwen3-Omni chat template returned an invalid prompt")
        token.raise_if_cancelled()
        stream = self._stream_generate_fn(
            model,
            processor,
            formatted,
            audio=None if audio is None else str(audio),
            image=None,
            video=None,
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
            skip_special_tokens=True,
        )
        parts: list[str] = []
        try:
            token.raise_if_cancelled()
            for chunk in stream:
                token.raise_if_cancelled()
                text = getattr(chunk, "text", None)
                if not isinstance(text, str):
                    raise ValueError("invalid Qwen3-Omni output chunk")
                parts.append(text)
            token.raise_if_cancelled()
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        return "".join(parts)

    @staticmethod
    def _validate_request(prompt: str, max_tokens: int) -> None:
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("prompt must be non-empty and max_tokens must be positive")

    def runtime_evidence(self) -> dict[str, Any]:
        model = self.model
        return collect_mlx_evidence(
            model,
            {
                "audio_tower_input_weight": model.thinker.audio_tower.conv2d1.weight,
                "language_model_embedding_weight": model.thinker.language_model.model.embed_tokens.weight,
                "vision_tower_input_weight": model.thinker.vision_tower.patch_embed.proj.weight,
            },
        )
