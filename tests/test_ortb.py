from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.domain import ContentContext, SourceMetadata
from engine.signals.result_types import (
    AggregateResult,
    AggregateSentiment,
    CoverageSummary,
    EntityAggregate,
    KeywordAggregate,
)
from engine.standards import load_taxonomy


FIXTURES = Path(__file__).parent / "fixtures" / "golden"


def aggregate(
    *, primary: str = "80DV8O", secondaries: tuple[str, ...] = ("222",),
    sentiment_score: float = -0.6,
    keyword: str = "coffee brewing", entity: str = "Hario Switch", entity_type: str = "product",
) -> AggregateResult:
    return AggregateResult(
        primary_category=primary,
        secondary_categories=secondaries,
        primary_role_support_seconds=((primary, 32.0),),
        secondary_role_support_seconds=tuple((item, 12.0) for item in secondaries),
        keywords=(KeywordAggregate(term=keyword, support_seconds=12.0, model_scores=(0.9,)),),
        entities=(EntityAggregate(text=entity, type=entity_type, mention_count=2, support_seconds=8.0),),
        sentiment=AggregateSentiment(
            label="negative" if sentiment_score < 0 else "positive" if sentiment_score > 0 else "neutral",
            score=sentiment_score,
        ),
        coverage=CoverageSummary(source_seconds=60.0, analysed_seconds=60.0),
    )


def test_tv_content_matches_golden_and_keeps_category_roles():
    from engine.ortb.build import build_content

    actual = build_content(
        aggregate(),
        metadata=SourceMetadata(title="Hario Switch review", url="https://example.test/review"),
        context=ContentContext.VIDEO,
        duration_seconds=60.9,
    )

    assert actual == json.loads((FIXTURES / "content-tv.json").read_text())
    assert actual["cattax"] == 7
    assert actual["cat"] == ["80DV8O", "222"]
    assert actual["context"] == 1
    assert all(isinstance(item["value"], str) for group in actual["data"] for item in group["segment"])


def test_english_vocals_content_matches_golden_without_genre_inference():
    from engine.ortb.build import build_content

    actual = build_content(
        aggregate(
            primary="338", secondaries=(), sentiment_score=0.0,
            keyword="someone like you", entity="Adele", entity_type="artist",
        ),
        metadata=SourceMetadata(title="Adele - Someone Like You"),
        context=ContentContext.MUSIC,
        duration_seconds=271.0,
    )

    assert actual == json.loads((FIXTURES / "content-english-vocals.json").read_text())
    assert actual["context"] == 3
    assert not {"genre", "genres"}.intersection(actual)


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        (ContentContext.VIDEO, 1),
        (ContentContext.MUSIC, 3),
        (ContentContext.OTHER, 6),
        (ContentContext.UNKNOWN, 7),
    ],
)
def test_context_uses_openrtb_values(context, expected):
    from engine.ortb.build import build_content

    content = build_content(
        aggregate(), metadata=SourceMetadata(), context=context, duration_seconds=1
    )
    assert content["context"] == expected


def test_generic_empty_category_omits_cat_and_cattax_and_title_stays_metadata():
    from engine.ortb.build import build_content, build_generic_content

    content = build_generic_content(
        metadata=SourceMetadata(title="Coffee Brewing Review"),
        context=ContentContext.OTHER,
        duration_seconds=3.2,
    )

    assert content["title"] == "Coffee Brewing Review"
    assert "cat" not in content and "cattax" not in content
    assert "kwarray" not in content
    with pytest.raises(ValueError, match="AggregateResult"):
        build_content(
            None,  # type: ignore[arg-type]
            metadata=SourceMetadata(title="Coffee Brewing Review"),
            context=ContentContext.OTHER,
            duration_seconds=3.2,
        )


def test_every_emitted_category_must_be_in_vendored_taxonomy():
    from engine.ortb.build import build_content
    from engine.ortb.validate import validate_content

    content = build_content(
        aggregate(), metadata=SourceMetadata(), context=ContentContext.MUSIC, duration_seconds=4
    )
    validate_content(content, taxonomy=load_taxonomy())
    invalid = {**content, "cat": ["not-a-taxonomy-id"]}
    with pytest.raises(ValueError, match="taxonomy"):
        validate_content(invalid, taxonomy=load_taxonomy())


def test_serializer_is_stable_utf8():
    from engine.ortb.export import serialize_content

    first = serialize_content({"title": "Café", "context": 6})
    assert first == serialize_content({"context": 6, "title": "Café"})
    assert b"Caf\xc3\xa9" in first
    with pytest.raises(ValueError):
        serialize_content({"context": 6, "len": float("nan")})


def test_serializer_rechecks_taxonomy_before_export():
    from engine.ortb.export import serialize_content

    with pytest.raises(ValueError, match="taxonomy"):
        serialize_content({"cattax": 7, "cat": ["unknown"], "context": 6})


def test_negative_sentiment_segment_preserves_its_sign():
    from engine.ortb.build import build_content

    content = build_content(
        aggregate(), metadata=SourceMetadata(), context=ContentContext.MUSIC, duration_seconds=10
    )
    segments = content["data"][0]["segment"]
    assert {item["id"]: item["value"] for item in segments}["sentiment"] == "negative:-0.6"


@pytest.mark.parametrize(
    "content",
    [
        {"context": 8},
        {"cat": ["80DV8O"]},
        {"cattax": 7},
        {"cat": ["80DV8O"], "cattax": 7.0},
        {
            "cat": ["80DV8O"],
            "cattax": 7,
            "data": [{"name": "contextual_engine", "segment": [{"id": "primary_category", "value": "255"}]}],
        },
    ],
)
def test_validator_rejects_nonconforming_context_or_partial_category_fields(content):
    from engine.ortb.validate import validate_content

    with pytest.raises(ValueError):
        validate_content(content, taxonomy=load_taxonomy())


@pytest.mark.parametrize(
    ("structure", "failed", "coverage"),
    [("multi_topic", (), "complete"),
     ("parent_rollup", ((10.0, 20.0),), "partial_technical_failures")],
)
def test_v3_content_discloses_topic_structure_and_coverage(structure, failed, coverage):
    from engine.domain import TimeRange
    from engine.ortb.build import build_content

    v3 = aggregate().model_copy(update={
        "policy_version": "english-audio-aggregation-v3",
        "topic_structure": structure,
        "selected_support_seconds": (("80DV8O", 32.0), ("222", 12.0)),
        "coverage": CoverageSummary(
            source_seconds=60.0, analysed_seconds=50.0,
            failed_ranges=tuple(TimeRange(start_seconds=a, end_seconds=b) for a, b in failed),
        ),
    })
    content = build_content(
        AggregateResult.model_validate(v3.model_dump(mode="python")),
        metadata=SourceMetadata(), context=ContentContext.VIDEO, duration_seconds=60.0,
    )
    segments = {item["id"]: item["value"] for item in content["data"][0]["segment"]}
    assert segments["aggregation_policy"] == "english-audio-aggregation-v3"
    assert segments["topic_structure"] == structure
    assert segments["coverage"] == coverage
    assert content["cat"] == ["80DV8O", "222"]


def test_v3_result_requires_structure_and_matching_support():
    values = aggregate().model_dump(mode="python")
    values["policy_version"] = "english-audio-aggregation-v3"
    with pytest.raises(ValueError):
        AggregateResult.model_validate(values)
    values["topic_structure"] = "single"
    values["selected_support_seconds"] = (("222", 12.0),)
    with pytest.raises(ValueError):
        AggregateResult.model_validate(values)
