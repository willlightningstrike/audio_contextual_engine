from __future__ import annotations

import pytest

from engine.domain import TimeRange
from engine.signals.text_types import (
    ClassificationPayload,
    EntityValue,
    KeywordValue,
    SentimentValue,
)
from engine.standards import TaxonomyRow


def _row(category_id: str, parent: str | None = None) -> TaxonomyRow:
    path = (category_id,) if parent is None else (parent, category_id)
    return TaxonomyRow(
        id=category_id,
        name=category_id,
        tier=len(path),
        parent_id=parent,
        path=path,
    )


def _payload(
    primary: str,
    secondary: tuple[str, ...] = (),
    *,
    keywords: tuple[KeywordValue, ...] = (),
    entities: tuple[EntityValue, ...] = (),
    sentiment: SentimentValue | None = None,
) -> ClassificationPayload:
    return ClassificationPayload(
        primary_category=primary,
        secondary_categories=secondary,
        keywords=keywords,
        entities=entities,
        sentiment=sentiment or SentimentValue(label="neutral", score=0.0),
    )


def _entry(start, end, evidence, payload):
    from engine.signals.result_types import AudioTranscriptProvenance, TextTimelineEntry

    return TextTimelineEntry(
        time_range=TimeRange(start_seconds=start, end_seconds=end),
        payload=payload,
        evidence_ranges=tuple(
            TimeRange(start_seconds=left, end_seconds=right) for left, right in evidence
        ),
        provenance=AudioTranscriptProvenance(language="en"),
    )


def _coverage(source: float, analysed: float, failed=(), excluded=()):
    from engine.signals.result_types import CoverageSummary

    return CoverageSummary(
        source_seconds=source,
        analysed_seconds=analysed,
        failed_ranges=tuple(
            TimeRange(start_seconds=left, end_seconds=right) for left, right in failed
        ),
        excluded_ranges=tuple(
            TimeRange(start_seconds=left, end_seconds=right) for left, right in excluded
        ),
    )


def test_primary_uses_primary_support_only():
    from engine.signals.aggregate import aggregate

    taxonomy = (_row("A"), _row("B"))
    result = aggregate(
        (
            _entry(0, 90, ((0, 10),), _payload("A")),
            _entry(90, 110, ((90, 105),), _payload("B")),
        ),
        coverage=_coverage(110, 25),
        taxonomy=taxonomy,
    )

    assert result.primary_category == "B"
    assert dict(result.primary_role_support_seconds) == {"A": 10, "B": 15}


def test_secondaries_preserve_roles():
    from engine.signals.aggregate import aggregate

    taxonomy = (_row("A"), _row("B"), _row("C"))
    result = aggregate(
        (
            _entry(0, 10, ((0, 10),), _payload("A", ("C",))),
            _entry(10, 20, ((10, 20),), _payload("B")),
        ),
        coverage=_coverage(20, 20),
        taxonomy=taxonomy,
    )

    assert result.primary_category == "A"
    assert result.secondary_categories == ("C",)


def test_overlapping_evidence_is_not_double_counted():
    from engine.signals.aggregate import aggregate

    taxonomy = (_row("A"),)
    entry = _entry(0, 10, ((0, 5), (3, 8)), _payload("A"))
    result = aggregate((entry,), coverage=_coverage(10, 8), taxonomy=taxonomy)

    assert dict(result.primary_role_support_seconds) == {"A": 8}


def test_overlapping_classification_units_are_rejected():
    from engine.signals.aggregate import aggregate

    entries = (
        _entry(0, 6, ((0, 5),), _payload("A")),
        _entry(4, 10, ((4, 9),), _payload("A")),
    )
    with pytest.raises(ValueError, match="classification units must not overlap"):
        aggregate(entries, coverage=_coverage(10, 9), taxonomy=(_row("A"),))


def test_gaps_do_not_receive_labels():
    from engine.signals.aggregate import aggregate

    result = aggregate(
        (_entry(0, 30, ((5, 10), (20, 25)), _payload("A")),),
        coverage=_coverage(30, 10, excluded=((0, 5), (10, 20), (25, 30))),
        taxonomy=(_row("A"), _row("B")),
    )

    assert result.primary_category == "A"
    assert dict(result.primary_role_support_seconds) == {"A": 10}
    assert result.coverage.excluded_ranges == tuple(
        TimeRange(start_seconds=start, end_seconds=end)
        for start, end in ((0, 5), (10, 20), (25, 30))
    )


def test_keyword_scores_are_preserved_without_confidence_aggregation():
    from engine.signals.aggregate import aggregate

    result = aggregate(
        (
            _entry(
                0,
                5,
                ((0, 5),),
                _payload("A", keywords=(KeywordValue(term=" Camera ", confidence=0.2),)),
            ),
            _entry(
                5,
                10,
                ((5, 10),),
                _payload("A", keywords=(KeywordValue(term="camera", confidence=0.9),)),
            ),
        ),
        coverage=_coverage(10, 10),
        taxonomy=(_row("A"),),
    )

    assert len(result.keywords) == 1
    assert result.keywords[0].term.strip().casefold() == "camera"
    assert result.keywords[0].model_scores == (0.2, 0.9)


def test_entity_counts_only_sum_disjoint_analysis_units():
    from engine.signals.aggregate import aggregate

    entity = EntityValue(text=" Acme ", type="Brand", mention_count=2)
    result = aggregate(
        (
            _entry(0, 5, ((0, 5),), _payload("A", entities=(entity,))),
            _entry(5, 10, ((5, 10),), _payload("A", entities=(entity,))),
        ),
        coverage=_coverage(10, 10),
        taxonomy=(_row("A"),),
    )

    assert len(result.entities) == 1
    assert result.entities[0].mention_count == 4
    assert result.entities[0].text.strip().casefold() == "acme"


def test_sentiment_signed_score_is_not_confidence():
    from engine.signals.aggregate import aggregate

    result = aggregate(
        (
            _entry(
                0,
                10,
                ((0, 10),),
                _payload("A", sentiment=SentimentValue(label="negative", score=-0.6)),
            ),
        ),
        coverage=_coverage(10, 10),
        taxonomy=(_row("A"),),
    )

    assert result.sentiment.label == "negative"
    assert result.sentiment.score == pytest.approx(-0.6)
    assert result.sentiment.aggregation_convention == "duration_weighted_signed_score_v1"


def test_primary_tie_breaks_by_lexical_taxonomy_id():
    from engine.signals.aggregate import aggregate

    result = aggregate(
        (
            _entry(0, 5, ((0, 5),), _payload("Z")),
            _entry(5, 10, ((5, 10),), _payload("A")),
        ),
        coverage=_coverage(10, 10),
        taxonomy=(_row("Z"), _row("A")),
    )

    assert result.primary_category == "A"


def test_hierarchy_redundant_secondary_is_removed():
    from engine.signals.aggregate import aggregate

    result = aggregate(
        (
            _entry(0, 10, ((0, 10),), _payload("P", ("C", "D"))),
        ),
        coverage=_coverage(10, 10),
        taxonomy=(_row("P"), _row("C", "P"), _row("D")),
    )

    assert result.secondary_categories == ("D",)


def test_v2_secondaries_use_only_secondary_role_citations():
    from engine.signals.aggregate import aggregate_topic_sufficient
    from engine.signals.topic_sufficiency import TopicDecision

    decision = TopicDecision(
        outcome="eligible", primary_category="A", transcript_seconds=30, audio_seconds=30,
        primary_role_support_seconds=(("A", 20.0), ("B", 10.0)),
        secondary_role_support_seconds=(("C", 2.0), ("D", 5.0)),
    )
    result = aggregate_topic_sufficient((
        _entry(0, 20, ((0, 20),), _payload("A", ("C", "D"))),
        _entry(20, 30, ((20, 30),), _payload("B")),
    ), decision=decision, coverage=_coverage(30, 30),
       taxonomy=(_row("A"), _row("B"), _row("C"), _row("D")))

    assert result.policy_version == "english-audio-aggregation-v2"
    assert result.primary_category == "A"
    assert result.secondary_categories == ("D", "C")
    assert dict(result.secondary_role_support_seconds) == {"C": 2, "D": 5}
    assert "B" not in result.secondary_categories


def test_v2_rejects_insufficient_decision():
    from engine.signals.aggregate import aggregate_topic_sufficient
    from engine.signals.topic_sufficiency import TopicDecision

    decision = TopicDecision(
        outcome="insufficient", primary_category=None, transcript_seconds=10, audio_seconds=10,
        primary_role_support_seconds=(), secondary_role_support_seconds=(),
    )
    with pytest.raises(ValueError, match="eligible primary"):
        aggregate_topic_sufficient((
            _entry(0, 10, ((0, 10),), _payload("A")),
        ), decision=decision, coverage=_coverage(10, 10), taxonomy=(_row("A"),))
