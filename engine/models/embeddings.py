"""MLX-native local embedding adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from engine.models import collect_mlx_evidence, require_resolved_local_directory


def _default_load(path: Path) -> Any:
    from mlx_embeddings.utils import load

    return load(str(path), lazy=False)


class MlxEmbeddingAdapter:
    def __init__(self, snapshot_path: Path, *, load_fn: Callable[[Path], Any] = _default_load) -> None:
        self._snapshot_path = require_resolved_local_directory(snapshot_path)
        self._load_fn = load_fn
        self._model: Any | None = None
        self._last_tensor: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_fn(self._snapshot_path)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not values or any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("texts must be a non-empty sequence of non-empty strings")
        loaded = self.model
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, tokenizer = loaded
            encoded = tokenizer(values, padding=True, truncation=True, return_tensors="np")
            from mlx import core as mx

            tensors = {name: mx.array(value) for name, value in encoded.items()}
            output = model(**tensors)
            result = output.text_embeds
            mx.eval(result)
            result = result.astype(mx.float32)
        else:
            # Injection seam for ordinary model-free tests only.
            result = loaded.encode(values)
        module_name = type(result).__module__
        if not module_name.startswith("mlx"):
            raise TypeError("embedding upstream result must be an MLX tensor")
        self._last_tensor = result
        array = np.asarray(result, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(values) or array.shape[1] == 0 or not np.isfinite(array).all():
            raise ValueError("invalid embedding matrix")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embedding vectors must be non-zero")
        return np.ascontiguousarray(array / norms, dtype=np.float32)

    def runtime_evidence(self) -> dict[str, Any]:
        model = self.model[0] if isinstance(self.model, tuple) else self.model
        tensors = {} if self._last_tensor is None else {"text_embeddings": self._last_tensor}
        return collect_mlx_evidence(model, tensors)

    def release(self) -> None:
        loaded = self._model is not None
        self._model = None
        self._last_tensor = None
        if loaded:
            from mlx import core as mx

            mx.clear_cache()
