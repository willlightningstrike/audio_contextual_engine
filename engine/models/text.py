"""Lazy-imported MLX-LM adapter for bounded JSON-schema generation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from engine.cancel import CancellationToken
from engine.models import require_resolved_local_directory
from engine.signals.text_types import GenerationResult


class GenerationTimeoutError(TimeoutError):
    """Constrained generation exceeded its wall-clock bound."""


def _default_runtime() -> SimpleNamespace:
    """Import Metal-backed packages only when constructing the real adapter."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import outlines
    from outlines.backends import get_json_schema_logits_processor

    return SimpleNamespace(
        load=load,
        stream_generate=stream_generate,
        make_sampler=make_sampler,
        wrap=outlines.from_mlxlm,
        get_processor=get_json_schema_logits_processor,
        eval=mx.eval,
        clear_cache=mx.clear_cache,
    )


class MlxConstrainedTextAdapter:
    """Own one local Qwen3.6 model and constrain every request independently."""

    def __init__(
        self,
        model_path: Path,
        *,
        runtime: Any | None = None,
        timeout_seconds: float = 180.0,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        self.model_path = require_resolved_local_directory(model_path)
        self._runtime = runtime or _default_runtime()
        self._timeout_seconds = float(timeout_seconds)
        self._model, self._tokenizer = self._runtime.load(
            str(self.model_path), lazy=False
        )
        self._wrapped = self._runtime.wrap(self._model, self._tokenizer)

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        *,
        max_tokens: int,
        token: CancellationToken,
    ) -> GenerationResult:
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 1024
        ):
            raise ValueError("max_tokens must be an integer in 1..1024")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be non-empty text")
        if not isinstance(schema, dict) or not schema:
            raise ValueError("schema must be a non-empty object")
        if self._model is None or self._tokenizer is None or self._wrapped is None:
            raise RuntimeError("text adapter has been released")

        token.raise_if_cancelled()
        inner_processor = self._runtime.get_processor(
            "outlines_core", self._wrapped, json.dumps(schema)
        )

        def processor(tokens: Any, logits: Any) -> Any:
            # mlx-lm #1156: materialize lazy tokens before stateful decoding.
            self._runtime.eval(tokens)
            return inner_processor(tokens, logits)

        # Qwen3.6 defaults to an open <think> block, which the JSON grammar cannot emit.
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token.raise_if_cancelled()
        stream = self._runtime.stream_generate(
            self._model,
            self._tokenizer,
            formatted,
            max_tokens=max_tokens,
            sampler=self._runtime.make_sampler(0.0),
            logits_processors=[processor],
        )
        pieces: list[str] = []
        token_count = 0
        finish_reason: str | None = None
        started = time.monotonic()
        try:
            for chunk in stream:
                token.raise_if_cancelled()
                if time.monotonic() - started > self._timeout_seconds:
                    raise GenerationTimeoutError(
                        "constrained generation exceeded its timeout"
                    )
                pieces.append(chunk.text)
                token_count = int(chunk.generation_tokens)
                if chunk.finish_reason is not None:
                    finish_reason = str(chunk.finish_reason)
                token.raise_if_cancelled()
        finally:
            stream.close()
        token.raise_if_cancelled()
        return GenerationResult(
            raw="".join(pieces),
            finish_reason=finish_reason or "unknown",
            seconds=float(time.monotonic() - started),
            token_count=token_count,
        )

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        self._wrapped = None
        self._runtime.clear_cache()
