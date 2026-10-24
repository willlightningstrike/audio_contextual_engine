"""Validation for the OpenRTB 2.6 Content subset emitted by this application."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from engine.standards import CATTAX_CONTENT_3_1

_CONTEXTS = frozenset({1, 3, 6, 7})
_SEGMENT_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_CONTENT_FIELDS = frozenset(
    {"id", "title", "url", "cat", "cattax", "kwarray", "language", "len", "context", "data"}
)


def validate_content(content: Mapping[str, Any], *, taxonomy: Sequence[Any]) -> None:
    """Reject malformed output and category IDs absent from the pinned taxonomy."""
    if not isinstance(content, Mapping):
        raise ValueError("OpenRTB content must be an object")
    if set(content) - _CONTENT_FIELDS:
        raise ValueError("content contains fields outside the supported OpenRTB subset")
    if "id" in content and (not isinstance(content["id"], str) or not content["id"].strip()):
        raise ValueError("id must be a non-empty string when present")
    if "context" in content:
        context = content["context"]
        if isinstance(context, bool) or not isinstance(context, int) or context not in _CONTEXTS:
            raise ValueError("context must be one of 1, 3, 6, or 7")
    if content.get("language", "en") != "en":
        raise ValueError("language must be English ('en')")
    if "len" in content:
        length = content["len"]
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("len must be a non-negative integer")
    if "keywords" in content:
        raise ValueError("keywords and kwarray are mutually exclusive")
    if any(key in content for key in ("genre", "genres")):
        raise ValueError("genres are not inferred or emitted")
    for key in ("title", "url"):
        if key in content and (not isinstance(content[key], str) or not content[key].strip()):
            raise ValueError(f"{key} must be a non-empty string when present")
    if "kwarray" in content:
        values = content["kwarray"]
        if not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError("kwarray must be a list of non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError("kwarray values must be unique")

    has_categories = "cat" in content
    has_taxonomy = "cattax" in content
    if has_categories != has_taxonomy:
        raise ValueError("cat and cattax must either both be present or both be omitted")
    if has_categories:
        cattax = content["cattax"]
        if isinstance(cattax, bool) or not isinstance(cattax, int) or cattax != CATTAX_CONTENT_3_1:
            raise ValueError("cattax must identify IAB Content Taxonomy 3.1 (9)")
        categories = content["cat"]
        if not isinstance(categories, list) or not 1 <= len(categories) <= 3:
            raise ValueError("cat must contain a primary and at most two secondary IDs")
        if any(not isinstance(item, str) or not item for item in categories):
            raise ValueError("category IDs must be non-empty strings")
        if len(categories) != len(set(categories)):
            raise ValueError("category IDs must be unique")
        valid_ids = {row.id for row in taxonomy}
        if any(item not in valid_ids for item in categories):
            raise ValueError("category ID is not present in the vendored taxonomy")

    role_segments: list[tuple[str, str]] = []
    if "data" in content:
        data = content["data"]
        if not isinstance(data, list):
            raise ValueError("data must be a list")
        for group in data:
            if not isinstance(group, Mapping) or not isinstance(group.get("name"), str) or not group["name"]:
                raise ValueError("each data item must have a non-empty name")
            segments = group.get("segment")
            if not isinstance(segments, list):
                raise ValueError("each data item must contain a segment list")
            for segment in segments:
                if not isinstance(segment, Mapping):
                    raise ValueError("segments must be objects")
                segment_id, value = segment.get("id"), segment.get("value")
                if not isinstance(segment_id, str) or not _SEGMENT_ID.fullmatch(segment_id):
                    raise ValueError("segment id must be a safe string identifier")
                if not isinstance(value, str):
                    raise ValueError("segment values must be strings")
                if any(isinstance(part, float) and not math.isfinite(part) for part in segment.values()):
                    raise ValueError("segment values must be finite")
                if segment_id in {"primary_category", "secondary_category"}:
                    role_segments.append((segment_id, value))

    if has_categories:
        expected_roles = [("primary_category", content["cat"][0])]
        expected_roles.extend(("secondary_category", item) for item in content["cat"][1:])
        if role_segments != expected_roles:
            raise ValueError("category role segments must match cat order and IDs")
    elif role_segments:
        raise ValueError("category role segments require cat and cattax")
