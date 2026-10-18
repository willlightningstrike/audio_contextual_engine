"""Strict model-response normalization outside model adapters."""

from __future__ import annotations

import ast
import json
from typing import Any


MAX_MODEL_RESPONSE_BYTES = 16_384


def _reject_non_rfc_constant(value: str) -> Any:
    raise ValueError(f"non-RFC JSON constant: {value}")


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def canonical_json_container(value: str) -> str:
    """Return strict JSON for an RFC document or a safe literal dict/list fallback."""
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_MODEL_RESPONSE_BYTES:
        raise ValueError("model response must be a JSON container")
    try:
        decoded = json.loads(value, parse_constant=_reject_non_rfc_constant)
    except (json.JSONDecodeError, ValueError):
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("model response must be a JSON container") from exc
    if not isinstance(decoded, (dict, list)) or not _is_json_value(decoded):
        raise ValueError("model response must be a JSON container")
    try:
        return json.dumps(
            decoded,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("model response must be a JSON container") from exc
