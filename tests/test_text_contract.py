from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.signals.taxonomy import TaxonomyCandidate
from engine.domain import TimeRange, TranscriptSegment
from engine.transcribe.types import AnalysisWindow


def _candidate(
    taxonomy_id: str,
    name: str,
    path: tuple[str, ...],
) -> TaxonomyCandidate:
    return TaxonomyCandidate(
        taxonomy_id=taxonomy_id,
        name=name,
        path=path,
        similarity=0.5,
    )


CANDIDATES = (
    _candidate("633", "Cameras and Camcorders", ("Technology", "Cameras")),
    _candidate("255", "Photography", ("Hobbies", "Photography")),
    _candidate("264", "Content Production", ("Hobbies", "Content Production")),
    _candidate(
        "268",
        "Video Production",
        ("Hobbies", "Content Production", "Video Production"),
    ),
)


VALID = {
    "primary_category": "633",
    "secondary_categories": ["255"],
    "keywords": [{"term": "camera", "confidence": 0.9}],
    "entities": [{"text": "Sony", "type": "Brand", "mention_count": 1}],
    "sentiment": {"label": "neutral", "score": 0.0},
}


@pytest.fixture
def two_segment_window():
    return AnalysisWindow(
        time_range=TimeRange(start_seconds=0, end_seconds=4),
        segments=(
            TranscriptSegment(
                time_range=TimeRange(start_seconds=0, end_seconds=2),
                text="camera settings", confidence=0.9, language="en",
            ),
            TranscriptSegment(
                time_range=TimeRange(start_seconds=2, end_seconds=4),
                text="photography", confidence=0.9, language="en",
            ),
        ),
    )


def _cited_value():
    return {
        "primary_category": "633", "secondary_categories": ["255"],
        "keywords": [], "entities": [],
        "sentiment": {"label": "neutral", "score": 0},
        "category_evidence": [
            {"category_id": "633", "segment_indices": [0]},
            {"category_id": "255", "segment_indices": [1]},
        ],
    }


def test_cited_payload_requires_exact_selected_category_indices(two_segment_window):
    from engine.signals.classify import validate_cited_classification_output

    payload, citations = validate_cited_classification_output(
        json.dumps(_cited_value()), CANDIDATES, two_segment_window
    )
    assert payload.secondary_categories == ("255",)
    assert [(item.category_id, item.segment_indices) for item in citations] == [
        ("633", (0,)), ("255", (1,)),
    ]


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("category_evidence"),
    lambda value: value["category_evidence"][1].update(category_id="633"),
    lambda value: value["category_evidence"][0].update(segment_indices=[0, 0]),
    lambda value: value["category_evidence"][0].update(segment_indices=[True]),
    lambda value: value["category_evidence"][0].update(segment_indices=[2]),
    lambda value: value["category_evidence"][1].update(category_id="264"),
    lambda value: value.update(category_evidence=[{"category_id": "255", "segment_indices": [1]}]),
    lambda value: value.update(category_evidence=[]),
    lambda value: value.update(primary_category="__INSUFFICIENT_TOPIC_EVIDENCE__"),
])
def test_cited_payload_rejects_incomplete_or_invalid_evidence(
    two_segment_window, mutation
):
    from engine.signals.classify import (
        InvalidClassificationOutput, validate_cited_classification_output,
    )

    value = _cited_value()
    mutation(value)
    with pytest.raises(InvalidClassificationOutput):
        validate_cited_classification_output(json.dumps(value), CANDIDATES, two_segment_window)


def test_cited_payload_drops_an_uncited_secondary(two_segment_window):
    from engine.signals.classify import validate_cited_classification_output

    value = _cited_value()
    value["category_evidence"].pop()
    payload, citations = validate_cited_classification_output(
        json.dumps(value), CANDIDATES, two_segment_window
    )
    assert payload.primary_category == "633"
    assert payload.secondary_categories == ()
    assert [item.category_id for item in citations] == ["633"]


def test_cited_payload_rejects_duplicate_json_keys(two_segment_window):
    from engine.signals.classify import (
        InvalidClassificationOutput, validate_cited_classification_output,
    )

    raw = json.dumps(_cited_value()).replace(
        '"category_id": "633"', '"category_id": "633", "category_id": "633"'
    )
    with pytest.raises(InvalidClassificationOutput):
        validate_cited_classification_output(raw, CANDIDATES, two_segment_window)


def test_cited_parser_rejects_segment_outside_window():
    from engine.signals.classify import (
        InvalidClassificationOutput, validate_cited_classification_output,
    )

    window = AnalysisWindow(
        time_range=TimeRange(start_seconds=0, end_seconds=2),
        segments=(TranscriptSegment(
            time_range=TimeRange(start_seconds=2, end_seconds=4),
            text="camera", confidence=0.9, language="en",
        ),),
    )
    value = _cited_value()
    value["secondary_categories"] = []
    value["category_evidence"] = [{"category_id": "633", "segment_indices": [0]}]
    with pytest.raises(InvalidClassificationOutput):
        validate_cited_classification_output(json.dumps(value), CANDIDATES, window)


def test_old_five_field_parser_still_accepts_historical_fixture():
    from engine.signals.classify import validate_classification_output

    assert validate_classification_output(json.dumps(VALID), CANDIDATES).primary_category == "633"


def test_prompts_serialize_indexed_transcript_without_retrieval_scores(two_segment_window):
    from engine.signals.prompts import (
        build_locked_classification_prompt, build_primary_classification_prompt,
    )

    expected = json.dumps([
        {"index": 0, "start_seconds": 0.0, "end_seconds": 2.0, "text": "camera settings"},
        {"index": 1, "start_seconds": 2.0, "end_seconds": 4.0, "text": "photography"},
    ], ensure_ascii=False)
    for prompt in (
        build_primary_classification_prompt(two_segment_window, CANDIDATES, topic_sufficiency=True),
        build_locked_classification_prompt(
            two_segment_window, CANDIDATES, "633", topic_sufficiency=True
        ),
    ):
        assert expected in prompt
        assert "similarity" not in prompt
        assert "source_path" not in prompt
        assert "video frame" not in prompt.lower()


def test_new_schemas_advertise_abstention_and_exact_citation_objects():
    from engine.signals.prompts import (
        TOPIC_ABSTENTION_TOKEN, locked_classification_schema,
        primary_classification_schema,
    )

    primary = primary_classification_schema(CANDIDATES, topic_sufficiency=True)
    assert primary["properties"]["primary_category"]["enum"] == [
        "633", "255", "264", "268", TOPIC_ABSTENTION_TOKEN,
    ]
    locked = locked_classification_schema(
        CANDIDATES, "633", topic_sufficiency=True, segment_count=2
    )
    assert "category_evidence" in locked["required"]
    branches = locked["properties"]["category_evidence"]["anyOf"]
    assert [branch["minItems"] for branch in branches] == [1, 2, 3]
    assert [branch["maxItems"] for branch in branches] == [1, 2, 3]
    primary, secondary = branches[2]["prefixItems"][:2]
    assert primary["properties"]["category_id"] == {"type": "string", "const": "633"}
    assert "633" not in secondary["properties"]["category_id"]["enum"]
    for item in (primary, secondary):
        assert item["required"] == ["category_id", "segment_indices"]
        assert item["additionalProperties"] is False
        assert item["properties"]["segment_indices"] == {
            "type": "array", "minItems": 1, "maxItems": 2,
            "items": {"type": "integer", "enum": [0, 1]},
        }
    with pytest.raises(ValueError):
        locked_classification_schema(CANDIDATES, "633", topic_sufficiency=True)


def test_cited_schema_constraints_are_enforced_by_the_decoder():
    import re
    from outlines_core.json_schema import build_regex_from_schema
    from engine.signals.prompts import locked_classification_schema

    schema = locked_classification_schema(
        CANDIDATES, "633", topic_sufficiency=True, segment_count=2
    )
    pattern = build_regex_from_schema(json.dumps(schema["properties"]["category_evidence"]))
    ok = '[{"category_id": "633", "segment_indices": [0]}, {"category_id": "264", "segment_indices": [1]}]'
    assert re.fullmatch(pattern, ok)
    for bad in (
        '[{"category_id": "633", "segment_indices": [2]}]',
        '[{"category_id": "264", "segment_indices": [0]}]',
        '[{"category_id": "633", "segment_indices": [0]}, {"category_id": "633", "segment_indices": [1]}]',
    ):
        assert not re.fullmatch(pattern, bad)


def test_opt_in_cited_prompt_requires_category_role_order(two_segment_window):
    from engine.signals.prompts import build_locked_classification_prompt

    prompt = build_locked_classification_prompt(
        two_segment_window, CANDIDATES, "633", topic_sufficiency=True
    )
    assert "primary first, then secondaries in secondary_categories order" in prompt


def test_default_prompt_and_schema_keep_legacy_five_field_contract(two_segment_window):
    from engine.signals.prompts import (
        TOPIC_ABSTENTION_TOKEN, build_locked_classification_prompt,
        build_primary_classification_prompt, locked_classification_schema,
        primary_classification_schema,
    )

    primary_schema = primary_classification_schema(CANDIDATES)
    locked_schema = locked_classification_schema(CANDIDATES, "633")
    primary_prompt = build_primary_classification_prompt(two_segment_window, CANDIDATES)
    locked_prompt = build_locked_classification_prompt(two_segment_window, CANDIDATES, "633")

    assert TOPIC_ABSTENTION_TOKEN not in primary_schema["properties"]["primary_category"]["enum"]
    assert TOPIC_ABSTENTION_TOKEN not in primary_prompt
    assert locked_schema["required"] == list(VALID)
    assert "category_evidence" not in locked_prompt
    assert "Indexed transcript segments" not in primary_prompt + locked_prompt
    assert json.dumps(["camera settings photography"], ensure_ascii=False) in primary_prompt


def test_old_classification_results_deserialize_with_new_defaults():
    from engine.signals.text_types import ClassificationResult

    historical = {
        "time_range": {"start_seconds": 0, "end_seconds": 4},
        "payload": VALID,
        "attempts": [{
            "attempt": 1, "prompt_sha256": "a" * 64,
            "raw": json.dumps(VALID), "finish_reason": "stop",
            "seconds": 1.0, "token_count": 20,
        }],
        "error_code": None,
    }
    result = ClassificationResult.model_validate_json(json.dumps(historical))
    assert result.category_citations == ()
    assert result.policy_version is None


def test_strict_contract_accepts_the_verified_model_shape_without_legacy_coercion():
    from engine.signals.classify import validate_classification_output

    payload = validate_classification_output(json.dumps(VALID), CANDIDATES)

    assert payload.primary_category == "633"
    assert payload.secondary_categories == ("255",)
    assert payload.keywords[0].term == "camera"
    assert payload.keywords[0].confidence == 0.9
    assert payload.entities[0].type == "Brand"
    assert payload.entities[0].mention_count == 1
    assert payload.sentiment.label == "neutral"
    assert payload.sentiment.score == 0.0
    assert not hasattr(payload.sentiment, "confidence")


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n" + json.dumps(VALID) + "\n```",
        "Here is the result: " + json.dumps(VALID),
        json.dumps(VALID)[:-1],
        '{"primary_category":"633","primary_category":"255",'
        '"secondary_categories":[],"keywords":[],"entities":[],'
        '"sentiment":{"label":"neutral","score":0.0}}',
        json.dumps(VALID).replace("0.9", "NaN"),
    ],
)
def test_non_json_wrappers_truncation_duplicate_keys_and_nan_are_rejected(raw: str):
    from engine.signals.classify import InvalidClassificationOutput
    from engine.signals.classify import validate_classification_output

    with pytest.raises(InvalidClassificationOutput):
        validate_classification_output(raw, CANDIDATES)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("keywords", 0, "confidence"), "0.9"),
        (("entities", 0, "mention_count"), "1"),
        (("entities", 0, "mention_count"), True),
        (("sentiment", "score"), "0.0"),
    ],
)
def test_quoted_numeric_values_and_boolean_integers_are_rejected(path, value):
    from engine.signals.classify import InvalidClassificationOutput
    from engine.signals.classify import validate_classification_output

    payload = json.loads(json.dumps(VALID))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(InvalidClassificationOutput):
        validate_classification_output(json.dumps(payload), CANDIDATES)


@pytest.mark.parametrize(
    "mutation",
    [
        {"primary_category": "999"},
        {"secondary_categories": ["999"]},
        {"secondary_categories": ["255", "255"]},
        {"secondary_categories": ["633"]},
        {"secondary_categories": ["255", "268", "264"]},
        {"keywords": [{"term": "", "confidence": 0.9}]},
        {"keywords": [{"term": "camera", "confidence": 1.1}]},
        {"keywords": [{"term": f"keyword-{index}", "confidence": 0.5} for index in range(9)]},
        {"entities": [{"text": "", "type": "Brand", "mention_count": 1}]},
        {"entities": [{"text": "Sony", "type": "", "mention_count": 1}]},
        {"entities": [{"text": "Sony", "type": "Brand", "mention_count": 0}]},
        {"entities": [{"text": f"entity-{index}", "type": "Brand", "mention_count": 1} for index in range(9)]},
        {"sentiment": {"label": "mixed", "score": 0.0}},
        {"sentiment": {"label": "neutral", "score": -1.1}},
    ],
)
def test_invalid_ids_duplicates_bounds_and_empty_text_are_rejected(mutation):
    from engine.signals.classify import InvalidClassificationOutput
    from engine.signals.classify import validate_classification_output

    payload = {**VALID, **mutation}
    with pytest.raises(InvalidClassificationOutput):
        validate_classification_output(json.dumps(payload), CANDIDATES)


def test_extra_or_missing_fields_are_rejected():
    from engine.signals.classify import InvalidClassificationOutput
    from engine.signals.classify import validate_classification_output

    extra = {**VALID, "explanation": "not part of the contract"}
    missing = dict(VALID)
    del missing["entities"]

    for payload in (extra, missing):
        with pytest.raises(InvalidClassificationOutput):
            validate_classification_output(json.dumps(payload), CANDIDATES)


def test_every_candidate_ancestor_descendant_pair_is_a_hierarchy_error():
    from engine.signals.classify import HierarchyRedundancyError
    from engine.signals.classify import validate_classification_output

    hierarchical = tuple(
        _candidate(str(index), name, tuple(f"level-{part}" for part in range(index)))
        for index, name in ((1, "root"), (2, "child"), (3, "grandchild"), (4, "leaf"))
    )
    for ancestor_index in range(len(hierarchical)):
        for descendant_index in range(ancestor_index + 1, len(hierarchical)):
            payload = {
                **VALID,
                "primary_category": hierarchical[ancestor_index].taxonomy_id,
                "secondary_categories": [hierarchical[descendant_index].taxonomy_id],
            }
            with pytest.raises(HierarchyRedundancyError):
                validate_classification_output(json.dumps(payload), hierarchical)


def test_strict_values_reject_nonfinite_direct_python_inputs():
    from engine.signals.text_types import KeywordValue, SentimentValue

    with pytest.raises(ValidationError):
        KeywordValue(term="camera", confidence=math.inf)
    with pytest.raises(ValidationError):
        SentimentValue(label="neutral", score=math.nan)
