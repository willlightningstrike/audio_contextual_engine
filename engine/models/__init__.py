"""MLX model adapters live in this package."""

from pathlib import Path
from collections.abc import Mapping
from typing import Any


def require_resolved_local_directory(path: Path) -> Path:
    """Reject repository IDs, relative paths, missing paths, and symlinks."""
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("model snapshot must be a resolved local directory")
    resolved = candidate.resolve(strict=True)
    if candidate != resolved:
        raise ValueError("model snapshot must be a resolved local directory")
    return resolved


def _leaves(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def collect_mlx_evidence(model: Any, tensors: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect types without importing MLX outside model-adapter modules."""
    parameters = model.parameters()
    leaves = _leaves(parameters)
    parameter_types = [f"{type(value).__module__}.{type(value).__qualname__}" for value in leaves]
    tensor_types = {name: f"{type(value).__module__}.{type(value).__qualname__}" for name, value in tensors.items()}
    return {
        "parameter_leaf_count": len(leaves),
        "parameter_leaf_types": parameter_types[:8],
        "parameter_leaves_are_mlx": bool(leaves) and all(type(value).__module__.startswith("mlx") for value in leaves),
        "tensor_types": tensor_types,
        "tensors_are_mlx": bool(tensors) and all(type(value).__module__.startswith("mlx") for value in tensors.values()),
    }


def mlx_memory_statistics() -> dict[str, int]:
    """Read version-pinned MLX allocator counters inside the model boundary."""
    import mlx.core as mx

    return {
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }
