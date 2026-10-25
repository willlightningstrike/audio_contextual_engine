"""Deterministic whole-file aggregation for validated audio-transcript results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from engine.signals.result_types import (
    AggregateResult,
    AggregateSentiment,
    CoverageSummary,
    EntityAggregate,
    KeywordAggregate,
    TextTimelineEntry,
    _duration,
)
from engine.signals.topic_sufficiency import TopicDecision
from engine.standards import TaxonomyRow


def _unit_key(entry: TextTimelineEntry) -> tuple[float, float]:
    return entry.time_range.start_seconds, entry.time_range.end_seconds


def _overlaps(first, second) -> bool:
    return first.start_seconds < second.end_seconds and second.start_seconds < first.end_seconds


def _ancestors(category_id: str, by_id: dict[str, TaxonomyRow]) -> set[str]:
    ancestors: set[str] = set()
    current = by_id.get(category_id)
    while current is not None and current.parent_id is not None:
        parent_id = current.parent_id
        if parent_id in ancestors:
            raise ValueError("taxonomy contains a parent cycle")
        ancestors.add(parent_id)
        current = by_id.get(parent_id)
    return ancestors


def _redundant(first: str, second: str, ancestors: dict[str, set[str]]) -> bool:
    return first == second or first in ancestors.get(second, set()) or second in ancestors.get(first, set())


def aggregate(
    entries: tuple[TextTimelineEntry, ...],
    *,
    coverage: CoverageSummary,
    taxonomy: Sequence[TaxonomyRow],
) -> AggregateResult:
    """Combine role-preserving classifier results without inventing confidence."""
    if not entries:
        raise ValueError("cannot aggregate without classified transcript evidence")
    if not isinstance(taxonomy, Sequence) or not taxonomy:
        raise ValueError("taxonomy must be a non-empty sequence")
    by_id = {row.id: row for row in taxonomy}
    if len(by_id) != len(taxonomy) or any(not isinstance(row.id, str) or not row.id for row in taxonomy):
        raise ValueError("taxonomy IDs must be unique non-empty strings")
    ancestors = {category_id: _ancestors(category_id, by_id) for category_id in by_id}

    ordered = tuple(sorted(entries, key=_unit_key))
    previous_end = -1.0
    for entry in ordered:
        start = entry.time_range.start_seconds
        end = entry.time_range.end_seconds
        if start < previous_end:
            raise ValueError("classification units must not overlap")
        if end > coverage.source_seconds:
            raise ValueError("classification unit exceeds source duration")
        previous_end = end
        for category_id in (entry.payload.primary_category, *entry.payload.secondary_categories):
            if category_id not in by_id:
                raise ValueError(f"unknown taxonomy category ID: {category_id}")
        for evidence in entry.evidence_ranges:
            if any(_overlaps(evidence, failed) for failed in coverage.failed_ranges):
                raise ValueError("classified evidence overlaps a failed range")
            if any(_overlaps(evidence, excluded) for excluded in coverage.excluded_ranges):
                raise ValueError("classified evidence overlaps an excluded range")

    primary_support: dict[str, float] = defaultdict(float)
    secondary_support: dict[str, float] = defaultdict(float)
    keyword_support: dict[str, float] = defaultdict(float)
    keyword_scores: dict[str, list[float]] = defaultdict(list)
    keyword_display: dict[str, set[str]] = defaultdict(set)
    entity_mentions: dict[tuple[str, str], int] = defaultdict(int)
    entity_support: dict[tuple[str, str], float] = defaultdict(float)
    entity_display: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    weighted_sentiment = 0.0
    sentiment_duration = 0.0

    for entry in ordered:
        evidence_seconds = _duration(entry.evidence_ranges)
        if evidence_seconds <= 0:
            raise ValueError("classification entry has no positive transcript support")
        primary_id = entry.payload.primary_category
        primary_support[primary_id] += evidence_seconds
        for category_id in entry.payload.secondary_categories:
            secondary_support[category_id] += evidence_seconds

        seen_keywords: set[str] = set()
        for item in entry.payload.keywords:
            normalized = item.term.strip().casefold()
            seen_keywords.add(normalized)
            keyword_scores[normalized].append(item.confidence)
            keyword_display[normalized].add(item.term.strip())
        for normalized in seen_keywords:
            keyword_support[normalized] += evidence_seconds

        seen_entities: set[tuple[str, str]] = set()
        for item in entry.payload.entities:
            key = (item.text.strip().casefold(), item.type.strip().casefold())
            entity_mentions[key] += item.mention_count
            entity_display[key].add((item.text.strip(), item.type.strip()))
            seen_entities.add(key)
        for key in seen_entities:
            entity_support[key] += evidence_seconds

        weighted_sentiment += entry.payload.sentiment.score * evidence_seconds
        sentiment_duration += evidence_seconds

    if not primary_support:
        raise ValueError("no usable primary-category support")
    primary_id = min(primary_support, key=lambda item: (-primary_support[item], item))
    ranked_secondaries = sorted(
        secondary_support,
        key=lambda item: (-secondary_support[item], item),
    )
    selected = [primary_id]
    secondaries: list[str] = []
    for category_id in ranked_secondaries:
        if any(_redundant(category_id, selected_id, ancestors) for selected_id in selected):
            continue
        secondaries.append(category_id)
        selected.append(category_id)
        if len(secondaries) == 2:
            break

    keywords = tuple(
        KeywordAggregate(
            term=min(keyword_display[key], key=lambda value: (value.casefold(), value)),
            support_seconds=keyword_support[key],
            model_scores=tuple(keyword_scores[key]),
        )
        for key in sorted(keyword_support, key=lambda item: (-keyword_support[item], item))[:20]
    )
    entities = tuple(
        EntityAggregate(
            text=min(
                entity_display[key],
                key=lambda value: (value[0].casefold(), value[0], value[1].casefold(), value[1]),
            )[0],
            type=min(
                entity_display[key],
                key=lambda value: (value[0].casefold(), value[0], value[1].casefold(), value[1]),
            )[1],
            mention_count=entity_mentions[key],
            support_seconds=entity_support[key],
        )
        for key in sorted(entity_mentions, key=lambda item: (-entity_mentions[item], item))[:50]
    )
    score = weighted_sentiment / sentiment_duration
    score = max(-1.0, min(1.0, score))
    label = "negative" if score < 0 else "positive" if score > 0 else "neutral"
    return AggregateResult(
        primary_category=primary_id,
        secondary_categories=tuple(secondaries),
        primary_role_support_seconds=tuple(
            (category_id, primary_support[category_id])
            for category_id in sorted(primary_support)
        ),
        secondary_role_support_seconds=tuple(
            (category_id, secondary_support[category_id])
            for category_id in sorted(secondary_support)
        ),
        keywords=keywords,
        entities=entities,
        sentiment=AggregateSentiment(label=label, score=score),
        coverage=coverage,
    )


def aggregate_topic_sufficient(
    entries: tuple[TextTimelineEntry, ...],
    *,
    decision: TopicDecision,
    coverage: CoverageSummary,
    taxonomy: Sequence[TaxonomyRow],
) -> AggregateResult:
    """Promote a cited whole-file primary while preserving secondary roles."""
    if decision.outcome != "eligible" or decision.primary_category is None:
        raise ValueError("v2 aggregation requires an eligible primary")
    base = aggregate(entries, coverage=coverage, taxonomy=taxonomy)
    by_id = {row.id: row for row in taxonomy}
    if decision.policy_version == "english-audio-aggregation-v3":
        # The v3 decision already chose and de-duplicated the categories.
        if any(category_id not in by_id for category_id, _ in decision.selected_support_seconds):
            raise ValueError("unknown taxonomy category in v3 decision")
        values = base.model_dump(mode="python")
        values.update(
            primary_category=decision.primary_category,
            secondary_categories=decision.secondary_categories,
            primary_role_support_seconds=decision.primary_role_support_seconds,
            secondary_role_support_seconds=decision.secondary_role_support_seconds,
            topic_structure=decision.topic_structure,
            selected_support_seconds=decision.selected_support_seconds,
            policy_version="english-audio-aggregation-v3",
        )
        return AggregateResult.model_validate(values)
    if decision.primary_category not in by_id:
        raise ValueError("unknown taxonomy primary category")
    ancestors = {category_id: _ancestors(category_id, by_id) for category_id in by_id}
    selected = [decision.primary_category]
    for category_id, seconds in sorted(
        decision.secondary_role_support_seconds, key=lambda item: (-item[1], item[0])
    ):
        if category_id not in by_id:
            raise ValueError("unknown taxonomy secondary category")
        if seconds > 0 and not any(_redundant(category_id, used, ancestors) for used in selected):
            selected.append(category_id)
        if len(selected) == 3:
            break
    values = base.model_dump(mode="python")
    values.update(
        primary_category=selected[0],
        secondary_categories=tuple(selected[1:]),
        primary_role_support_seconds=decision.primary_role_support_seconds,
        secondary_role_support_seconds=decision.secondary_role_support_seconds,
        policy_version="english-audio-aggregation-v2",
    )
    return AggregateResult.model_validate(values)


MUSIC_ID = "338"
MUSIC_VIDEO_ID = "650"


def aggregate_declared_music(
    entries: tuple[TextTimelineEntry, ...],
    *,
    decision: TopicDecision,
    coverage: CoverageSummary,
    taxonomy: Sequence[TaxonomyRow],
    video: bool,
) -> AggregateResult:
    """Lead with the user-declared Music (and Music Video) labels.

    These labels come from the user's content context, never from the audio.
    Any eligible transcript topics follow them while room remains.
    """
    if decision.policy_version != "english-audio-aggregation-v3":
        raise ValueError("declared music requires a v3 topic decision")
    declared = (MUSIC_ID, MUSIC_VIDEO_ID) if video else (MUSIC_ID,)
    by_id = {row.id: row for row in taxonomy}
    if any(category_id not in by_id for category_id in declared):
        raise ValueError("declared music categories are missing from the taxonomy")
    ancestors = {category_id: _ancestors(category_id, by_id) for category_id in by_id}
    # A rolled-up parent such as Entertainment would only restate Music.
    supported = tuple(
        (category_id, seconds) for category_id, seconds in decision.selected_support_seconds
        if not any(_redundant(category_id, used, ancestors) for used in declared)
    )[:3 - len(declared)] if decision.outcome == "eligible" else ()
    base = aggregate(entries, coverage=coverage, taxonomy=taxonomy) if entries else None
    categories = (*declared, *(category_id for category_id, _ in supported))
    return AggregateResult(
        primary_category=categories[0],
        secondary_categories=categories[1:],
        primary_role_support_seconds=decision.primary_role_support_seconds,
        secondary_role_support_seconds=decision.secondary_role_support_seconds,
        keywords=base.keywords if base else (),
        entities=base.entities if base else (),
        sentiment=base.sentiment if base else None,
        coverage=coverage,
        policy_version="english-audio-aggregation-v3",
        topic_structure="declared_music",
        selected_support_seconds=supported,
        declared_categories=declared,
    )
