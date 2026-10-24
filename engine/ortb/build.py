"""Build the application's English audio-derived OpenRTB Content object."""

from __future__ import annotations

import math
import re
from typing import Any

from engine.domain import ContentContext, SourceMetadata
from engine.signals.result_types import AggregateResult
from engine.standards import CATTAX_CONTENT_3_1, load_taxonomy
from engine.ortb.validate import validate_content

_SAFE_SEGMENT_ID = re.compile(r"[^a-zA-Z0-9_-]+")


def _score_text(score: float) -> str:
    return repr(score)


def _entity_segment_id(entity_type: str) -> str:
    normalized = _SAFE_SEGMENT_ID.sub("_", entity_type.strip().lower()).strip("_-")
    if not normalized:
        normalized = "unknown"
    return f"entity_{normalized}"


def _base_content(
    *,
    metadata: SourceMetadata,
    context: ContentContext,
    duration_seconds: float,
) -> dict[str, Any]:
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise ValueError("duration_seconds must be a finite non-negative number")
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("duration_seconds must be a finite non-negative number")
    try:
        if isinstance(context, bool):
            raise ValueError("boolean is not a content context")
        resolved_context = ContentContext(context)
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported OpenRTB content context") from exc

    content: dict[str, Any] = {
        "context": int(resolved_context),
        "language": "en",
        "len": int(duration_seconds),
    }
    if metadata.title is not None and metadata.title.strip():
        content["title"] = metadata.title.strip()
    if metadata.url is not None and metadata.url.strip():
        content["url"] = metadata.url.strip()

    return content


def build_content(
    aggregate: AggregateResult,
    *,
    metadata: SourceMetadata,
    context: ContentContext,
    duration_seconds: float,
) -> dict[str, Any]:
    """Build semantic OpenRTB Content with a primary and optional secondaries."""
    if not isinstance(aggregate, AggregateResult):
        raise ValueError("semantic content requires a validated AggregateResult")
    content = _base_content(
        metadata=metadata, context=context, duration_seconds=duration_seconds
    )
    categories = [aggregate.primary_category, *aggregate.secondary_categories]
    content["cattax"] = CATTAX_CONTENT_3_1
    content["cat"] = categories
    keywords = list(dict.fromkeys(item.term for item in aggregate.keywords))
    if keywords:
        content["kwarray"] = keywords
    segments = [
        {
            "id": "sentiment",
            "value": f"{aggregate.sentiment.label}:{_score_text(aggregate.sentiment.score)}",
        }
    ]
    segments.extend(
        {"id": _entity_segment_id(entity.type), "value": entity.text}
        for entity in aggregate.entities
    )
    segments.append({"id": "primary_category", "value": aggregate.primary_category})
    segments.extend(
        {"id": "secondary_category", "value": category}
        for category in aggregate.secondary_categories
    )
    if aggregate.policy_version == "english-audio-aggregation-v3":
        # Consumers need to know when categories are co-equal or rolled up, and
        # when technically failed ranges were excluded from the decision.
        segments.extend([
            {"id": "aggregation_policy", "value": aggregate.policy_version},
            {"id": "topic_structure", "value": aggregate.topic_structure},
            {"id": "coverage", "value": (
                "partial_technical_failures" if aggregate.coverage.failed_ranges else "complete"
            )},
        ])
    content["data"] = [{"name": "contextual_engine", "segment": segments}]
    validate_content(content, taxonomy=load_taxonomy())
    return content


def build_generic_content(
    *, metadata: SourceMetadata, context: ContentContext, duration_seconds: float
) -> dict[str, Any]:
    """Build metadata-only Content; semantic analysis must use ``build_content``."""
    content = _base_content(
        metadata=metadata, context=context, duration_seconds=duration_seconds
    )
    validate_content(content, taxonomy=load_taxonomy())
    return content
