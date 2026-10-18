"""One canonical UTF-8 representation shared by storage and export callers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from engine.ortb.validate import validate_content
from engine.standards import load_taxonomy


def serialize_content(content: Mapping[str, Any]) -> bytes:
    """Serialize a content object deterministically as compact UTF-8 JSON."""
    if not isinstance(content, Mapping) or any(not isinstance(key, str) for key in content):
        raise ValueError("OpenRTB content must be a string-keyed object")
    validate_content(content, taxonomy=load_taxonomy())
    try:
        return json.dumps(
            dict(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("OpenRTB content must contain finite JSON values") from exc
