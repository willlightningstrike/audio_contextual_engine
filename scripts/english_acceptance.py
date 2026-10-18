"""Freeze and audit a qualitative, English-only Phase 1 acceptance run.

The semantic decision is intentionally prediction-first. This module never
assigns category truth from an observed prediction and keeps post-result
reviews outside the frozen protocol.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import re
import subprocess
import signal
import sys
import threading
import time
from collections.abc import Mapping
from queue import Empty, Queue
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmarks" / "acceptance-english-v1.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "runs.local" / "english-acceptance-v1" / "acceptance-report.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FROZEN_PROTOCOL_SHA256 = "97820db9e1cf2de3a560bb3c19183fbc2c3041ea9d4dd1534ed1692f4065112d"
_TV_PODCAST_SUPPLEMENT_SHA256 = "af82328445856c110613b29d4edc257f3fd7aaccabea577d7b457497ca8afacb"
_TV_FOLLOWUP_SHA256 = "13def206ff7b1d0302eee82e428de255da50cc1a9f7bd9d941a35900739fee98"
_CORRECTIVE_SUPPLEMENT_SHA256 = "e3c42074cbce20bcfc4dea8c74798743853db6a9f0f95b46c23bbaa125a17092"
_TOPIC_SUFFICIENCY_SHA256 = "ab58fa88518056f3233d84b74d9e98ae986ed53564d4cdf23bf74a7caf26d463"
_TOPIC_PROTOCOL_ID = "english-phase1-topic-sufficiency-v1"
_TOPIC_CASE_IDS = {
    "tv-minus01-2-dialogue-600-720",
    "tv-minus01-corrective-60-180",
    "backstreet-vocal-90-120",
    "sony-camera-speech-30-90",
    "hario-coffee-speech-120-300",
    "whitney-houston-full-audio",
}
_MINIMUM_PHYSICAL_MEMORY_BYTES = 64 * 1024**3
_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_ROUTING_OUTCOMES = {
    "accepted_english",
    "rejected_non_english",
    "rejected_unconfirmed_vocal",
    "rejected_no_transcript",
    "indeterminate",
}
_RESOURCE_FIELDS = (
    "wall_seconds",
    "peak_process_rss_bytes",
    "mlx_peak_memory_bytes",
    "host_memory_pressure_before",
    "host_memory_pressure_samples_percent",
    "host_memory_pressure_peak_percent",
    "swap_bytes_before",
    "swap_peak_bytes",
    "swap_bytes_after",
)
_RESOURCE_NUMERIC_FIELDS = (
    "wall_seconds",
    "peak_process_rss_bytes",
    "mlx_peak_memory_bytes",
    "swap_bytes_before",
    "swap_peak_bytes",
    "swap_bytes_after",
)
_CANCELLATION_STAGES = {"extraction", "asr", "classification", "batch"}
_CANCELLATION_LIMIT_SECONDS = {
    "extraction": 5.0,
    "asr": 35.0,
    "classification": 35.0,
    "batch": 35.0,
}


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    payload = {key: item for key, item in value.items() if key != "protocol_sha256"}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_digest(protocol: dict[str, Any]) -> str:
    """Hash every frozen protocol field except the digest field itself."""
    return hashlib.sha256(_canonical_bytes(protocol)).hexdigest()


def prediction_review_digest(case_id: str, attempt: dict[str, Any]) -> str | None:
    """Bind a qualitative decision to the exact input and result evidence."""
    payload = {
        "case_id": case_id,
        "source_sha256": attempt.get("source_sha256"),
        "input_sha256": attempt.get("input_sha256"),
        "requested_range": attempt.get("requested_range"),
        "predictions": attempt.get("predictions"),
        "transcript_coverage": attempt.get("transcript_coverage"),
        "errors": attempt.get("errors"),
    }
    if "outcome" in attempt:
        payload.update({
            "outcome": attempt.get("outcome"),
            "topic_policy": attempt.get("topic_policy"),
            "category_citations": attempt.get("category_citations"),
            "transcript_windows": attempt.get("transcript_windows"),
            "classification_results": attempt.get("classification_results"),
        })
    try:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical).hexdigest()


def validate_protocol(protocol: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(protocol, dict):
        return ["protocol must be a JSON object"]
    if protocol.get("schema_version") != 1:
        errors.append("unsupported acceptance protocol schema_version")
    pinned_digests = {
        "english-phase1-acceptance-v1": _FROZEN_PROTOCOL_SHA256,
        "english-phase1-tv-podcast-supplement-v1": _TV_PODCAST_SUPPLEMENT_SHA256,
        "english-phase1-tv-followup-v1": _TV_FOLLOWUP_SHA256,
        "english-phase1-corrective-supplement-v1": _CORRECTIVE_SUPPLEMENT_SHA256,
        _TOPIC_PROTOCOL_ID: _TOPIC_SUFFICIENCY_SHA256,
    }
    protocol_id = protocol.get("protocol_id")
    if protocol_id not in pinned_digests:
        errors.append("protocol_id must identify a code-pinned English acceptance protocol")
    if protocol.get("frozen_before_inference") is not True:
        errors.append("protocol must be frozen before inference")
    declared_digest = protocol.get("protocol_sha256")
    if not isinstance(declared_digest, str) or not _SHA256.fullmatch(declared_digest):
        errors.append("protocol_sha256 is missing or malformed")
    else:
        try:
            if declared_digest != protocol_digest(protocol):
                errors.append("protocol digest mismatch: frozen fields changed")
            if declared_digest != pinned_digests.get(protocol_id):
                errors.append("protocol digest does not match the code-pinned acceptance protocol")
        except (TypeError, ValueError):
            errors.append("protocol contains values that cannot be canonically hashed")

    resource_requirements = protocol.get("resource_requirements")
    if not isinstance(resource_requirements, dict):
        errors.append("resource_requirements is required")
    else:
        minimum_memory = resource_requirements.get("physical_memory_bytes_minimum")
        if (
            isinstance(minimum_memory, bool)
            or not isinstance(minimum_memory, int)
            or minimum_memory < _MINIMUM_PHYSICAL_MEMORY_BYTES
        ):
            errors.append("physical_memory_bytes_minimum must be at least 64 GiB")
        stages = resource_requirements.get("live_cancellation_stages")
        if (
            not isinstance(stages, list)
            or any(not isinstance(item, str) for item in stages)
            or len(stages) != len(set(stages))
            or set(stages) != _CANCELLATION_STAGES
        ):
            errors.append("live_cancellation_stages must include extraction, asr, classification, and batch once each")

    policy = protocol.get("semantic_policy")
    if not isinstance(policy, dict):
        errors.append("semantic_policy is required")
    else:
        if policy.get("method") != "qualitative_prediction_first":
            errors.append("semantic method must be qualitative_prediction_first")
        if policy.get("numeric_accuracy_threshold") is not None:
            errors.append("qualitative semantic acceptance cannot declare a numeric accuracy threshold")
        if policy.get("review_status") != "pending":
            errors.append("the frozen semantic review status must remain pending")

    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("protocol requires a non-empty cases list")
        return errors
    required_strata = protocol.get("required_strata")
    if (
        not isinstance(required_strata, list)
        or not required_strata
        or any(not isinstance(item, str) or not item.strip() for item in required_strata)
        or len(required_strata) != len(set(required_strata))
    ):
        errors.append("required_strata must contain unique non-empty strings")
    seen: set[str] = set()
    represented_strata: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not _CASE_ID.fullmatch(case_id):
            errors.append(f"{prefix} case_id must be a filesystem-safe lowercase slug")
        elif case_id in seen:
            errors.append(f"duplicate case_id: {case_id}")
        else:
            seen.add(case_id)
        if not isinstance(case.get("source_path"), str) or not case["source_path"].strip():
            errors.append(f"{prefix} source_path must be non-empty")
        if not isinstance(case.get("display_name"), str) or not case["display_name"].strip():
            errors.append(f"{prefix} display_name must be non-empty")
        if not isinstance(case.get("input_path"), str) or not case["input_path"].strip():
            errors.append(f"{prefix} input_path must be non-empty")
        source_hash = case.get("source_sha256")
        if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
            errors.append(f"{prefix} source_sha256 must be lowercase SHA-256")
        input_hash = case.get("input_sha256")
        if not isinstance(input_hash, str) or not _SHA256.fullmatch(input_hash):
            errors.append(f"{prefix} input_sha256 must be lowercase SHA-256")
        time_range = case.get("source_range")
        if not isinstance(time_range, dict):
            errors.append(f"{prefix} source_range is required")
        else:
            start = time_range.get("start_seconds")
            end = time_range.get("end_seconds")
            if (
                isinstance(start, bool)
                or not isinstance(start, (int, float))
                or not math.isfinite(start)
                or isinstance(end, bool)
                or not isinstance(end, (int, float))
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                errors.append(f"{prefix} source_range must be a finite, positive range")
            else:
                analysis_duration = case.get("analysis_duration_seconds")
                if (
                    isinstance(analysis_duration, bool)
                    or not isinstance(analysis_duration, (int, float))
                    or not math.isfinite(analysis_duration)
                    or analysis_duration <= 0
                    or not math.isclose(
                        float(analysis_duration), float(end) - float(start), abs_tol=0.05
                    )
                ):
                    errors.append(f"{prefix} analysis_duration_seconds must match its frozen source range")
        context_override = case.get("context_override")
        if context_override is not None and (
            isinstance(context_override, bool) or context_override not in {1, 3, 6, 7}
        ):
            errors.append(f"{prefix} context_override must be a supported OpenRTB context or null")
        if not isinstance(case.get("stratum"), str) or not case["stratum"].strip():
            errors.append(f"{prefix} stratum must be non-empty")
        else:
            represented_strata.add(case["stratum"])
        if case.get("semantic_reference") is not None:
            errors.append(
                f"{prefix} contains a semantic_reference; post-result reviews must remain separate, not blind labels"
            )
        routing_reference = case.get("routing_reference")
        if routing_reference is not None:
            if not isinstance(routing_reference, dict):
                errors.append(f"{prefix} routing_reference must be an object or null")
            elif (
                routing_reference.get("expected_outcome") not in _ROUTING_OUTCOMES
                or routing_reference.get("provenance") != "user_confirmed_prior"
                or routing_reference.get("blind") is not False
            ):
                errors.append(f"{prefix} routing_reference must identify non-blind prior user confirmation")
    if isinstance(required_strata, list) and all(isinstance(item, str) for item in required_strata):
        missing_strata = sorted(set(required_strata) - represented_strata)
        if missing_strata:
            errors.append("required strata have no frozen cases: " + ", ".join(missing_strata))
    if protocol_id == _TOPIC_PROTOCOL_ID:
        if seen != _TOPIC_CASE_IDS or len(cases) != 6:
            errors.append("topic-sufficiency protocol requires its exact six preselected cases")
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                continue
            if case.get("non_blind") is not True:
                errors.append(f"case[{index}] must be explicitly non_blind")
            if not isinstance(case.get("selection_provenance"), str) or not case["selection_provenance"].strip():
                errors.append(f"case[{index}] selection_provenance is required")
            if case.get("expected_topic_routing") not in {"topical_positive", "undetermined"}:
                errors.append(f"case[{index}] expected_topic_routing is required")
    return errors


def _valid_prediction(value: object, taxonomy_ids: set[str] | None = None) -> bool:
    if not isinstance(value, dict):
        return False
    primary = value.get("primary_category")
    secondaries = value.get("secondary_categories")
    keywords = value.get("keywords")
    entities = value.get("entities")
    sentiments = value.get("sentiment")
    if not isinstance(primary, str) or not primary.strip():
        return False
    if (
        not isinstance(secondaries, list)
        or len(secondaries) > 2
        or any(not isinstance(item, str) or not item.strip() for item in secondaries)
        or primary in secondaries
        or len(secondaries) != len(set(secondaries))
    ):
        return False
    if taxonomy_ids is None or primary not in taxonomy_ids or any(
        item not in taxonomy_ids for item in secondaries
    ):
        return False
    if not isinstance(keywords, list) or any(
        not isinstance(item, str) or not item.strip() for item in keywords
    ):
        return False
    if not isinstance(entities, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("text"), str)
        or not item["text"].strip()
        or not isinstance(item.get("type"), str)
        or not item["type"].strip()
        for item in entities
    ):
        return False
    if not isinstance(sentiments, list) or not sentiments:
        return False
    for item in sentiments:
        if not isinstance(item, dict):
            return False
        if item.get("label") not in {"negative", "neutral", "positive"}:
            return False
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        if not math.isfinite(score) or not -1 <= score <= 1:
            return False
    return True


def _range_covers(actual: object, expected: object) -> bool:
    if not isinstance(actual, list) or not isinstance(expected, dict):
        return False
    start = expected.get("start_seconds")
    end = expected.get("end_seconds")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    if any(
        not isinstance(item, dict)
        or isinstance(item.get("start_seconds"), bool)
        or not isinstance(item.get("start_seconds"), (int, float))
        or isinstance(item.get("end_seconds"), bool)
        or not isinstance(item.get("end_seconds"), (int, float))
        or item["start_seconds"] < float(start) - 1e-6
        or item["end_seconds"] > float(end) + 1e-6
        for item in actual
    ):
        return False
    valid: list[tuple[float, float]] = []
    for item in actual:
        if not isinstance(item, dict):
            continue
        left, right = item.get("start_seconds"), item.get("end_seconds")
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isfinite(left)
            and math.isfinite(right)
            and right > left
        ):
            valid.append((float(left), float(right)))
    valid.sort()
    cursor = float(start)
    for left, right in valid:
        if left > cursor + 1e-6:
            return False
        if right > cursor:
            cursor = right
        if cursor >= float(end) - 1e-6:
            return True
    return False


def _status(status: str, detail: str | None = None, **values: object) -> dict[str, object]:
    value: dict[str, object] = {"status": status, **values}
    if detail is not None:
        value["detail"] = detail
    return value


def evaluate_topic_attempts(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Keep known development outcomes visible without making an accuracy claim."""
    development = [deepcopy(item) for item in attempts if item.get("non_blind") is True]
    blind = [deepcopy(item) for item in attempts if item.get("non_blind") is False]
    return {
        "blind_semantic_denominator": len(blind),
        "development_cases": development,
    }


def _valid_topic_generation_attempts(result: dict[str, Any]) -> bool:
    """Check that saved classification claims retain their actual generation stages."""
    from pydantic import ValidationError
    from engine.signals.classify import InvalidClassificationOutput, _parse_json_object, _strict_payload
    from engine.signals.prompts import TOPIC_ABSTENTION_TOKEN
    from engine.signals.text_types import GenerationAttempt

    abstention = result.get("error_code") == "INSUFFICIENT_TOPIC_EVIDENCE"
    saved = result.get("attempts")
    if not isinstance(saved, list) or len(saved) != (1 if abstention else 2):
        return False
    parsed: list[dict[str, Any]] = []
    for index, value in enumerate(saved, start=1):
        if not isinstance(value, dict):
            return False
        try:
            generation = GenerationAttempt.model_validate_json(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError, ValidationError):
            return False
        if (
            generation.attempt != index
            or generation.stage != ("primary" if index == 1 else "payload")
            or generation.finish_reason != "stop"
            or generation.error_code is not None
            or generation.reason is not None
            or generation.seconds is None
            or generation.token_count is None
        ):
            return False
        try:
            parsed.append(_parse_json_object(generation.raw))
        except InvalidClassificationOutput:
            return False
    if abstention:
        return parsed[0] == {"primary_category": TOPIC_ABSTENTION_TOKEN}
    payload = result.get("payload")
    citations = result.get("category_citations")
    if not isinstance(payload, dict) or not isinstance(citations, list):
        return False
    if parsed[0] != {"primary_category": payload.get("primary_category")}:
        return False
    raw_payload = parsed[1]
    if set(raw_payload) != {
        "primary_category", "secondary_categories", "keywords", "entities",
        "sentiment", "category_evidence",
    } or raw_payload["category_evidence"] != citations:
        return False
    try:
        validated = _strict_payload({key: value for key, value in raw_payload.items() if key != "category_evidence"})
    except InvalidClassificationOutput:
        return False
    return validated.primary_category == payload.get("primary_category") and list(validated.secondary_categories) == payload.get("secondary_categories")


def _valid_topic_decision_attempt(attempt: dict[str, Any], case: dict[str, Any]) -> bool:
    """Recompute a saved v2 decision from its indexed transcript evidence."""
    from pydantic import ValidationError
    from engine.signals.topic_sufficiency import TopicDecision

    eligible = attempt.get("outcome") == "category"
    insufficient = attempt.get("outcome") == "insufficient_topic_evidence"
    if not (eligible or insufficient):
        return False
    if attempt.get("job_status") != ("complete" if eligible else "failed"):
        return False
    if eligible and attempt.get("export_byte_identical") is not True:
        return False
    if insufficient and (attempt.get("export_byte_identical") is not None or attempt.get("predictions") is not None):
        return False
    predictions = attempt.get("predictions")
    policy = attempt.get("topic_policy")
    if not isinstance(policy, dict) or set(policy) != {
        "policy_version", "outcome", "primary_category", "transcript_seconds",
        "audio_seconds", "primary_role_support_seconds", "secondary_role_support_seconds",
    }:
        return False
    if eligible and not isinstance(predictions, dict):
        return False
    secondaries = predictions.get("secondary_categories") if eligible else []
    if not isinstance(secondaries, list) or any(not isinstance(item, str) for item in secondaries):
        return False
    try:
        decision = TopicDecision.model_validate_json(json.dumps(policy, allow_nan=False))
    except (TypeError, ValueError, ValidationError):
        return False
    if decision.outcome != ("eligible" if eligible else "insufficient"):
        return False
    if decision.primary_category != (predictions.get("primary_category") if eligible else None):
        return False
    if decision.transcript_seconds <= 0 or not math.isclose(
        decision.audio_seconds,
        case["source_range"]["end_seconds"] - case["source_range"]["start_seconds"],
        abs_tol=0.05,
    ):
        return False
    primary_support = dict(decision.primary_role_support_seconds)
    secondary_support = dict(decision.secondary_role_support_seconds)
    if eligible and not set(secondaries).issubset(secondary_support):
        return False

    def valid_range(value: object, outer: dict[str, Any]) -> bool:
        if not isinstance(value, dict):
            return False
        start, end = value.get("start_seconds"), value.get("end_seconds")
        return (
            not isinstance(start, bool) and isinstance(start, (int, float)) and math.isfinite(start)
            and not isinstance(end, bool) and isinstance(end, (int, float)) and math.isfinite(end)
            and outer["start_seconds"] <= start < end <= outer["end_seconds"]
        )

    source_range = case["source_range"]
    windows = attempt.get("transcript_windows")
    results = attempt.get("classification_results")
    citations = attempt.get("category_citations")
    if not isinstance(windows, list) or not windows or not isinstance(results, list) or not isinstance(citations, list):
        return False
    if len(windows) != len(results):
        return False
    transcript_ranges: list[dict[str, Any]] = []
    cited_ranges: dict[tuple[str, str], list[dict[str, Any]]] = {}
    expected_citations: list[dict[str, Any]] = []
    for window, result in zip(windows, results):
        if not isinstance(window, dict) or not isinstance(result, dict):
            return False
        analysis_range = window.get("time_range")
        segments = window.get("segment_ranges")
        if not valid_range(analysis_range, source_range) or not isinstance(segments, list) or not segments:
            return False
        if any(not valid_range(segment, analysis_range) for segment in segments):
            return False
        transcript_ranges.extend(segments)
        if result.get("time_range") != analysis_range or result.get("policy_version") != "english-audio-topic-sufficiency-v1":
            return False
        payload = result.get("payload")
        result_citations = result.get("category_citations")
        if not isinstance(result_citations, list):
            return False
        if not _valid_topic_generation_attempts(result):
            return False
        if result.get("error_code") == "INSUFFICIENT_TOPIC_EVIDENCE":
            if payload is not None or result_citations:
                return False
            continue
        if result.get("error_code") is not None or not isinstance(payload, dict):
            return False
        primary = payload.get("primary_category")
        secondary = payload.get("secondary_categories")
        if not isinstance(primary, str) or not isinstance(secondary, list) or any(not isinstance(item, str) for item in secondary):
            return False
        selected = {primary, *secondary}
        if len(selected) != len(secondary) + 1 or len(result_citations) != len(selected):
            return False
        seen_roles: set[tuple[str, str]] = set()
        for item in result_citations:
            if not isinstance(item, dict):
                return False
            category_id = item.get("category_id")
            indices = item.get("segment_indices")
            if not isinstance(category_id, str) or category_id not in selected or not isinstance(indices, list) or not indices:
                return False
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(segments) for index in indices) or len(indices) != len(set(indices)):
                return False
            role = "primary" if category_id == primary else "secondary"
            if (role, category_id) in seen_roles:
                return False
            seen_roles.add((role, category_id))
            ranges = [segments[index] for index in indices]
            cited_ranges.setdefault((role, category_id), []).extend(ranges)
            expected_citations.append({
                "category_id": category_id, "role": role,
                "analysis_range": analysis_range, "segment_indices": indices,
                "citation_ranges": ranges,
            })
        if {category_id for _, category_id in seen_roles} != selected:
            return False
    if citations != expected_citations:
        return False
    if not math.isclose(decision.transcript_seconds, _union_duration(transcript_ranges), abs_tol=0.05):
        return False
    measured = {
        role: {category_id: _union_duration(ranges) for (item_role, category_id), ranges in cited_ranges.items() if item_role == role}
        for role in ("primary", "secondary")
    }
    if set(primary_support) != set(measured["primary"]) or set(secondary_support) != set(measured["secondary"]):
        return False
    if any(not math.isclose(seconds, measured[role][category_id], abs_tol=0.05)
           for role, supports in (("primary", primary_support), ("secondary", secondary_support))
           for category_id, seconds in supports.items()):
        return False
    qualifying = {category_id for category_id, seconds in measured["primary"].items()
                  if seconds / decision.transcript_seconds > 0.5 and seconds / decision.audio_seconds >= 0.25}
    return qualifying == ({decision.primary_category} if eligible else set())


def evaluate_protocol(
    protocol: dict[str, Any],
    attempts: Sequence[dict[str, Any]],
    *,
    post_result_reviews: Sequence[dict[str, Any]] = (),
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score operational evidence without inventing semantic ground truth."""
    run_metadata = dict(run_metadata or {})
    errors = validate_protocol(protocol)
    topic_protocol = protocol.get("protocol_id") == _TOPIC_PROTOCOL_ID
    evidence_input_errors = run_metadata.get("evidence_input_errors", [])
    if not isinstance(evidence_input_errors, list) or any(
        not isinstance(item, str) for item in evidence_input_errors
    ):
        evidence_input_errors = ["saved evidence input error metadata is malformed"]
    blocking_reason = None if not run_metadata else run_metadata.get("reason")
    environment_blocked = blocking_reason in {
        "METAL_UNAVAILABLE",
        "UNSUPPORTED_HOST",
        "USER_CANCELLED",
    }
    user_cancelled_run = blocking_reason == "USER_CANCELLED"
    cases = protocol.get("cases") if isinstance(protocol.get("cases"), list) else []
    case_ids = [
        case.get("case_id")
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    ]
    by_id: dict[str, list[dict[str, Any]]] = {item: [] for item in case_ids if isinstance(item, str)}
    unknown_attempts: list[dict[str, Any]] = []
    for raw in attempts:
        attempt = deepcopy(raw)
        case_id = attempt.get("case_id") if isinstance(attempt, dict) else None
        if not isinstance(case_id, str) or case_id not in by_id:
            unknown_attempts.append(attempt)
        else:
            by_id[case_id].append(attempt)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    duplicate: list[str] = []
    identity_errors: list[str] = []
    partial: list[str] = []
    unattempted: list[str] = []
    execution_errors: list[str] = []
    schema_errors: list[str] = []
    export_failures: list[str] = []
    export_pending: list[str] = []
    export_checked: list[str] = []
    routing_total = 0
    routing_correct = 0
    topic_positive_failed: list[str] = []
    topic_positive_pending: list[str] = []
    resource_missing: list[str] = []
    resource_invalid: list[str] = []
    cancellation_stages_seen: set[str] = set()
    cancellation_failures: list[str] = []
    has_predictions = any(
        isinstance(attempt, dict) and attempt.get("predictions") is not None
        for attempts_for_case in by_id.values()
        for attempt in attempts_for_case
    )
    taxonomy_ids: set[str] = set()
    if has_predictions:
        try:
            from engine.standards import load_taxonomy

            taxonomy_ids = {item.id for item in load_taxonomy()}
        except Exception:
            errors.append("IAB Content Taxonomy 3.0 could not be loaded for prediction validation")
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        selected = by_id.get(case_id, []) if isinstance(case_id, str) else []
        if not selected:
            missing.append(str(case_id))
        if len(selected) > 1:
            duplicate.append(str(case_id))
        case_attempt_rows: list[dict[str, Any]] = []
        expected_hash = case.get("source_sha256")
        expected_range = case.get("source_range")
        for attempt in selected:
            attempt_errors = attempt.get("errors", [])
            if not isinstance(attempt_errors, list):
                attempt_errors = [{"code": "MALFORMED_ERRORS_FIELD"}]
                execution_errors.append(f"{case_id}: malformed errors field")
            source_matches = attempt.get("source_sha256") == expected_hash
            input_matches = attempt.get("input_sha256") == case.get("input_sha256")
            range_matches = attempt.get("requested_range") == expected_range
            if not source_matches or not input_matches or not range_matches:
                identity_errors.append(str(case_id))
            if topic_protocol and attempt.get("non_blind") is not case.get("non_blind"):
                identity_errors.append(str(case_id))
            attempted = attempt.get("attempted") is True
            execution_status = attempt.get("execution_status", "not_attempted")
            handled_insufficiency = (
                topic_protocol
                and execution_status == "handled_topic_insufficiency"
                and attempt.get("outcome") == "insufficient_topic_evidence"
                and attempt_errors == [{"code": "INSUFFICIENT_TOPIC_EVIDENCE"}]
                and _valid_topic_decision_attempt(attempt, case)
            )
            if not attempted or execution_status not in {"complete", "handled_out_of_scope"} and not handled_insufficiency:
                is_expected_interruption = (
                    environment_blocked
                    and (
                        execution_status == "not_attempted"
                        or user_cancelled_run and execution_status == "cancelled"
                    )
                )
                if not is_expected_interruption:
                    execution_errors.append(f"{case_id}: {execution_status}")
            elif attempted and execution_status == "complete" and attempt_errors:
                execution_errors.append(f"{case_id}: completed attempt contains explicit errors")
            is_full = _range_covers(attempt.get("processed_ranges"), expected_range)
            if not attempted or user_cancelled_run and execution_status == "cancelled":
                unattempted.append(str(case_id))
            elif not is_full:
                partial.append(str(case_id))
            predictions = attempt.get("predictions")
            schema_status = "not_applicable"
            if predictions is not None:
                schema_status = (
                    "passed"
                    if taxonomy_ids and _valid_prediction(predictions, taxonomy_ids)
                    else "failed"
                )
                if schema_status == "failed":
                    schema_errors.append(str(case_id))
            elif execution_status == "complete":
                schema_status = "failed"
                schema_errors.append(str(case_id))
            if topic_protocol and execution_status == "complete":
                if not _valid_topic_decision_attempt(attempt, case):
                    schema_status = "failed"
                    schema_errors.append(str(case_id))
            if topic_protocol and execution_status == "handled_topic_insufficiency" and not handled_insufficiency:
                schema_status = "failed"
                schema_errors.append(str(case_id))
            export_match = attempt.get("export_byte_identical")
            if export_match is False:
                export_failures.append(str(case_id))
            elif execution_status == "complete" and export_match is not True:
                export_pending.append(str(case_id))
            elif export_match is True:
                export_checked.append(str(case_id))
            reference = case.get("routing_reference")
            if isinstance(reference, dict):
                routing_total += 1
                if attempt.get("routing_outcome") == reference.get("expected_outcome"):
                    routing_correct += 1
            resource = attempt.get("resource")
            if not isinstance(resource, dict) or any(
                field not in resource or resource[field] is None for field in _RESOURCE_FIELDS
            ):
                resource_missing.append(str(case_id))
            elif any(
                isinstance(resource.get(field), bool)
                or not isinstance(resource.get(field), (int, float))
                or not math.isfinite(resource[field])
                or resource[field] < 0
                for field in _RESOURCE_NUMERIC_FIELDS
            ):
                resource_invalid.append(str(case_id))
            elif (
                isinstance(resource.get("host_memory_pressure_peak_percent"), bool)
                or not isinstance(resource.get("host_memory_pressure_peak_percent"), (int, float))
                or not math.isfinite(resource["host_memory_pressure_peak_percent"])
                or not 0 <= resource["host_memory_pressure_peak_percent"] <= 100
                or not isinstance(resource.get("host_memory_pressure_samples_percent"), list)
                or not resource["host_memory_pressure_samples_percent"]
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 100
                    for value in resource["host_memory_pressure_samples_percent"]
                )
            ):
                resource_missing.append(str(case_id))
            probes = attempt.get("cancellation_probes", [])
            if isinstance(probes, list):
                for item in probes:
                    if not isinstance(item, dict) or item.get("outcome") != "cancelled":
                        continue
                    stage = item.get("stage")
                    elapsed = item.get("request_to_stop_seconds")
                    valid_elapsed = (
                        not isinstance(elapsed, bool)
                        and isinstance(elapsed, (int, float))
                        and math.isfinite(elapsed)
                        and elapsed >= 0
                    )
                    if (
                        stage in _CANCELLATION_STAGES
                        and item.get("process_stopped") is True
                        and valid_elapsed
                        and elapsed <= _CANCELLATION_LIMIT_SECONDS[stage]
                    ):
                        cancellation_stages_seen.add(stage)
                    elif stage in _CANCELLATION_STAGES:
                        cancellation_failures.append(str(stage))
            row = {
                "case_id": case_id,
                "display_name": case.get("display_name"),
                "stratum": case.get("stratum"),
                "attempted": attempted,
                "job_status": attempt.get("job_status"),
                "routing_outcome": attempt.get("routing_outcome"),
                "coverage_status": (
                    "not_run" if not attempted
                    else "interrupted" if execution_status == "cancelled"
                    else "complete" if is_full else "partial"
                ),
                "schema_status": schema_status,
                "export_byte_identical": export_match,
                "predictions": predictions,
                "outcome": attempt.get("outcome") if topic_protocol else None,
                "topic_policy": attempt.get("topic_policy") if topic_protocol else None,
                "category_citations": attempt.get("category_citations") if topic_protocol else None,
                "transcript_windows": attempt.get("transcript_windows") if topic_protocol else None,
                "classification_results": attempt.get("classification_results") if topic_protocol else None,
                "window_predictions": attempt.get("window_predictions") if topic_protocol else None,
                "non_blind": attempt.get("non_blind") if topic_protocol else None,
                "review_provenance": attempt.get("review_provenance") if topic_protocol else None,
                "transcript_coverage": attempt.get("transcript_coverage"),
                "candidate_coverage": attempt.get("candidate_coverage"),
                "timestamp_correction_frequency": attempt.get("timestamp_correction_frequency"),
                "prediction_sha256": (
                    prediction_review_digest(case_id, selected[0])
                    if isinstance(case_id, str) and len(selected) == 1
                    else None
                ),
                "errors": attempt_errors,
                "resource": resource,
                "cancellation_probes": probes,
            }
            case_attempt_rows.append(row)
        if topic_protocol and case.get("expected_topic_routing") == "topical_positive":
            routing_total += 1
            if len(selected) == 1 and (
                selected[0].get("routing_outcome") == "accepted_english"
                and selected[0].get("execution_status") == "complete"
                and selected[0].get("outcome") == "category"
            ):
                routing_correct += 1
            elif environment_blocked and len(selected) == 1 and (
                selected[0].get("attempted") is False
                or user_cancelled_run and selected[0].get("execution_status") == "cancelled"
            ):
                topic_positive_pending.append(str(case_id))
            else:
                topic_positive_failed.append(str(case_id))
        rows.append(
            {
                "case_id": case_id,
                "display_name": case.get("display_name"),
                "stratum": case.get("stratum"),
                "attempt_count": len(selected),
                "coverage_status": (
                    "missing" if not selected
                    else "not_run" if all(
                        row["coverage_status"] == "not_run" for row in case_attempt_rows
                    )
                    else "interrupted" if all(
                        row["coverage_status"] in {"not_run", "interrupted"}
                        for row in case_attempt_rows
                    ) and any(row["coverage_status"] == "interrupted" for row in case_attempt_rows)
                    else "partial" if any(
                        row["coverage_status"] != "complete" for row in case_attempt_rows
                    ) else "complete"
                ),
                "schema_status": (
                    "failed" if any(row["schema_status"] == "failed" for row in case_attempt_rows)
                    else "passed" if any(row["schema_status"] == "passed" for row in case_attempt_rows)
                    else "not_applicable"
                ),
                "attempts": case_attempt_rows,
                "errors": [error for row in case_attempt_rows for error in row["errors"]],
            }
        )

    if unknown_attempts:
        errors.append("attempts contain unknown case IDs")
    integrity_errors = errors + evidence_input_errors
    protocol_gate = _status(
        "failed" if integrity_errors else "passed",
        "; ".join(integrity_errors) if integrity_errors else None,
        failures=integrity_errors,
    )
    coverage_failures = missing + duplicate + identity_errors + partial
    coverage_status = (
        "failed" if coverage_failures
        else "pending" if environment_blocked and unattempted
        else "passed"
    )
    coverage_gate = _status(
        coverage_status,
        (
            "input set, hashes, or full source ranges do not match the frozen protocol"
            if coverage_failures
            else "the source set is frozen, but inference did not run or finish because the environment was unavailable or the run was cancelled"
            if coverage_status == "pending"
            else None
        ),
        planned_cases=len(cases),
        attempted_cases=sum(
            attempt.get("attempted") is True
            for case_id in case_ids
            for attempt in by_id.get(case_id, [])
        ),
        missing_case_ids=missing,
        duplicate_case_ids=duplicate,
        identity_mismatch_case_ids=identity_errors,
        partial_case_ids=partial,
        not_attempted_case_ids=unattempted,
    )
    execution_status = (
        "failed" if execution_errors
        else "pending" if environment_blocked and unattempted
        else "passed"
    )
    execution_gate = _status(
        execution_status,
        (
            "one or more cases failed before a handled result was retained"
            if execution_errors
            else "MLX inference did not run or finish because the environment was unavailable or the run was cancelled"
            if execution_status == "pending"
            else None
        ),
        failures=execution_errors,
    )
    schema_gate = _status(
        "failed" if schema_errors else "passed" if any(
            row["schema_status"] == "passed" for row in rows
        ) else "not_applicable",
        "completed predictions must obey the strict primary-plus-zero-to-two-secondary schema"
        if schema_errors else None,
        invalid_case_ids=schema_errors,
    )
    if topic_protocol and routing_total:
        routing_gate = _status(
            "failed" if routing_correct < routing_total - len(topic_positive_pending)
            else "pending" if topic_positive_pending else "passed",
            "predeclared topical-positive controls must route to an accepted English category"
            if topic_positive_failed else None,
            denominator=routing_total,
            correct=routing_correct,
            rate=(routing_correct / routing_total) if routing_total else None,
            failed_topic_positive_case_ids=topic_positive_failed,
            pending_topic_positive_case_ids=topic_positive_pending,
        )
    elif routing_total:
        routing_gate = _status(
            "passed" if routing_correct == routing_total else "failed",
            None if routing_correct == routing_total else "routing did not match prior non-blind references",
            denominator=routing_total,
            correct=routing_correct,
            rate=(routing_correct / routing_total) if routing_total else None,
        )
    else:
        routing_gate = _status(
            "not_applicable", "no routing reference labels were frozen", denominator=0, rate=None
        )
    if export_failures:
        export_gate = _status(
            "failed",
            "stored export bytes differ from the shared canonical serializer",
            mismatched_case_ids=export_failures,
        )
    elif export_pending:
        export_gate = _status(
            "pending",
            "completed cases are missing a byte-for-byte export comparison",
            unchecked_case_ids=export_pending,
        )
    elif export_checked:
        export_gate = _status("passed", checked_case_ids=export_checked)
    else:
        export_gate = _status("not_applicable", "no completed export was produced")
    reviews = [deepcopy(item) for item in post_result_reviews]
    reviewed_case_ids: set[str] = set()
    review_errors: list[str] = []
    review_rejections: list[str] = []
    for review in reviews:
        if not isinstance(review, dict):
            review_errors.append("review must be an object")
            continue
        case_id = review.get("case_id")
        decision = review.get("review")
        if not isinstance(case_id, str) or case_id not in by_id:
            review_errors.append("review contains an unknown case_id")
        elif case_id in reviewed_case_ids:
            review_errors.append(f"duplicate review for {case_id}")
        else:
            reviewed_case_ids.add(case_id)
        if not isinstance(decision, str) or decision not in {"approved", "revision_requested"}:
            review_errors.append(f"{case_id}: review must be approved or revision_requested")
        elif decision == "revision_requested":
            review_rejections.append(str(case_id))
        if review.get("provenance") != "post_result_user_review":
            review_errors.append(f"{case_id}: review provenance must be post_result_user_review")
        expected_digest = next(
            (
                row["attempts"][0].get("prediction_sha256")
                for row in rows
                if row.get("case_id") == case_id
                and row.get("attempt_count") == 1
                and row.get("attempts")
            ),
            None,
        )
        if (
            not isinstance(review.get("prediction_sha256"), str)
            or expected_digest is None
            or review.get("prediction_sha256") != expected_digest
        ):
            review_errors.append(f"{case_id}: review is not bound to the exact prediction evidence")
    unreviewed_case_ids = sorted(set(case_ids) - reviewed_case_ids)
    false_abstentions = sorted(
        case["case_id"] for case in cases
        if topic_protocol and isinstance(case, dict)
        and case.get("expected_topic_routing") == "topical_positive"
        and any(attempt.get("outcome") == "insufficient_topic_evidence" for attempt in by_id.get(case["case_id"], []))
    )
    if review_errors:
        semantic_gate = _status(
            "failed",
            "post-result qualitative review records are malformed or inconsistent",
            method="qualitative_prediction_first",
            numeric_threshold=None,
            errors=review_errors,
        )
    elif review_rejections or false_abstentions:
        semantic_gate = _status(
            "failed",
            "qualitative review rejected a prediction or a pre-confirmed topical positive abstained",
            method="qualitative_prediction_first",
            numeric_threshold=None,
            revision_requested_case_ids=sorted(review_rejections),
            false_abstention_case_ids=false_abstentions,
        )
    elif unreviewed_case_ids:
        semantic_gate = _status(
            "pending",
            "qualitative prediction review has not been completed for every case",
            method="qualitative_prediction_first",
            numeric_threshold=None,
            unreviewed_case_ids=unreviewed_case_ids,
        )
    else:
        semantic_gate = _status(
            "passed",
            method="qualitative_prediction_first",
            numeric_threshold=None,
            reviewed_case_ids=sorted(reviewed_case_ids),
            decision_basis="explicit post-result qualitative user approval; not a numeric accuracy claim",
        )
    missing_resource = sorted(set(resource_missing))
    cancellation_missing = sorted(_CANCELLATION_STAGES - cancellation_stages_seen)
    required_memory = protocol.get("resource_requirements", {}).get(
        "physical_memory_bytes_minimum", _MINIMUM_PHYSICAL_MEMORY_BYTES
    )
    if (
        isinstance(required_memory, bool)
        or not isinstance(required_memory, int)
        or required_memory < _MINIMUM_PHYSICAL_MEMORY_BYTES
    ):
        required_memory = _MINIMUM_PHYSICAL_MEMORY_BYTES
    host_memory = run_metadata.get("host_physical_memory_bytes")
    cold_warm = run_metadata.get("cold_warm_timing")
    cold_warm_values = (
        (cold_warm.get("cold_wall_seconds"), cold_warm.get("warm_wall_seconds"))
        if isinstance(cold_warm, dict)
        else (None, None)
    )
    cold_warm_invalid = any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in cold_warm_values
        if value is not None
    )
    cold_warm_missing = (
        not isinstance(cold_warm, dict)
        or any(value is None for value in cold_warm_values)
        or cold_warm.get("warm_probe_status") in {None, "cancelled"}
    )
    cold_warm_failed = (
        isinstance(cold_warm, dict)
        and cold_warm.get("warm_probe_status") is not None
        and cold_warm.get("warm_probe_status") not in {"complete", "cancelled"}
    ) or (
        isinstance(cold_warm, dict)
        and bool(cases)
        and isinstance(cases[0], dict)
        and cold_warm.get("case_id") != cases[0].get("case_id")
    )
    host_memory_failure = (
        not environment_blocked
        and isinstance(host_memory, int)
        and not isinstance(host_memory, bool)
        and host_memory < required_memory
    )
    host_memory_missing = (
        not environment_blocked
        and (isinstance(host_memory, bool) or not isinstance(host_memory, int))
    )
    if host_memory_failure:
        resources_gate = _status(
            "failed",
            "live inference ran on a host below the frozen physical-memory minimum",
            host_physical_memory_bytes=host_memory,
            minimum_physical_memory_bytes=required_memory,
        )
    elif cancellation_failures:
        resources_gate = _status(
            "failed",
            "one or more live cancellations did not stop within the frozen boundary limit",
            failing_stages=sorted(cancellation_failures),
            missing_cancellation_stages=cancellation_missing,
        )
    elif resource_invalid or cold_warm_invalid or cold_warm_failed:
        resources_gate = _status(
            "failed",
            "one or more resource measurements are invalid",
            invalid_case_ids=sorted(set(resource_invalid)),
            cold_warm_timing_invalid=cold_warm_invalid,
            cold_warm_probe_failed=cold_warm_failed,
        )
    elif missing_resource or host_memory_missing or cold_warm_missing:
        resources_gate = _status(
            "pending",
            "required host and MLX measurements are incomplete",
            missing_case_ids=missing_resource,
            missing_cancellation_stages=cancellation_missing,
            host_physical_memory_bytes=host_memory,
            minimum_physical_memory_bytes=required_memory,
            cold_warm_timing_missing=cold_warm_missing,
        )
    elif cancellation_missing:
        resources_gate = _status(
            "pending",
            "live cancellation evidence is required at each named pipeline boundary",
            missing_case_ids=[],
            missing_cancellation_stages=cancellation_missing,
        )
    else:
        resources_gate = _status("passed", missing_case_ids=[], missing_cancellation_stages=[])

    gates = {
        "protocol_integrity": protocol_gate,
        "execution": execution_gate,
        "coverage": coverage_gate,
        "routing": routing_gate,
        "schema": schema_gate,
        "export_integrity": export_gate,
        "semantic": semantic_gate,
        "resources": resources_gate,
    }
    if any(gate["status"] == "failed" for gate in gates.values()):
        decision = "fail"
    elif any(gate["status"] == "pending" for gate in gates.values()):
        decision = "pending"
    else:
        decision = "pass"
    supplement_only = protocol.get("protocol_id") in {
        "english-phase1-tv-podcast-supplement-v1",
        "english-phase1-tv-followup-v1",
        "english-phase1-corrective-supplement-v1",
        _TOPIC_PROTOCOL_ID,
    }
    if supplement_only and decision == "pass":
        decision = "pending"
    blind_reference_count = sum(
        1 for case in cases if isinstance(case, dict) and case.get("semantic_reference") is not None
    )
    result = {
        "schema_version": 1,
        "protocol_id": protocol.get("protocol_id") if isinstance(protocol, dict) else None,
        "protocol_sha256": protocol.get("protocol_sha256") if isinstance(protocol, dict) else None,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "release_decision": decision,
        "release_scope": "supplemental_format_coverage" if supplement_only else "full_phase1",
        "gates": gates,
        "cases": rows,
        "attempts": [deepcopy(item) for item in attempts],
        "unknown_attempts": unknown_attempts,
        "missing_case_ids": missing,
        "protocol_errors": errors,
        "evidence_input_errors": evidence_input_errors,
        "semantic_review_separation": {
            "blind_reference_count": blind_reference_count,
            "post_result_review_count": len(reviews),
        },
        "post_result_reviews": reviews,
        "run_metadata": run_metadata,
    }
    if topic_protocol:
        normalized_attempts = [
            {**attempt, "non_blind": case.get("non_blind")}
            for case in cases if isinstance(case, dict)
            for attempt in by_id.get(case.get("case_id"), [])
        ]
        result.update(evaluate_topic_attempts(normalized_attempts))
        result["release_scope"] = "topic_sufficiency_challenge_only"
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_source_root() -> Path:
    if ROOT.parent.name == ".worktrees":
        return ROOT.parent.parent.resolve()
    return ROOT.resolve()


def _resolve_source(value: str) -> Path:
    candidate = (ROOT / value).resolve(strict=True)
    allowed = _allowed_source_root()
    if candidate != allowed and allowed not in candidate.parents:
        raise ValueError("source_path resolves outside the project media root")
    if not candidate.is_file():
        raise ValueError("source_path is not a regular file")
    return candidate


def _source_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("media duration is not a finite positive number")
    return duration


def _physical_memory_bytes() -> int:
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def _host_preflight(minimum_memory_bytes: int) -> tuple[bool, dict[str, Any], list[str]]:
    system = platform.system()
    machine = platform.machine().lower()
    mac_version = platform.mac_ver()[0]
    try:
        mac_major = int(mac_version.split(".", maxsplit=1)[0])
    except (AttributeError, ValueError):
        mac_major = 0
    memory_bytes = _physical_memory_bytes()
    failures: list[str] = []
    if system != "Darwin":
        failures.append("macOS required")
    if machine != "arm64":
        failures.append("Apple Silicon arm64 required")
    if mac_major < 14:
        failures.append("macOS 14 or newer required")
    if memory_bytes < minimum_memory_bytes:
        failures.append(f"physical memory below {minimum_memory_bytes} bytes")
    evidence = {
        "system": system,
        "macos_version": mac_version,
        "machine": machine,
        "physical_memory_bytes": memory_bytes,
        "minimum_memory_bytes": minimum_memory_bytes,
    }
    return not failures, evidence, failures


def _verify_inputs(protocol: dict[str, Any]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    resolved: dict[str, Path] = {}
    attempts: list[dict[str, Any]] = []
    for case in protocol["cases"]:
        case_id = case["case_id"]
        try:
            source = _resolve_source(case["source_path"])
            if _sha256_file(source) != case["source_sha256"]:
                raise ValueError("source hash differs from frozen protocol")
            source_duration = _source_duration(source)
            planned = case["source_range"]
            if float(planned["end_seconds"]) > source_duration + 0.05:
                raise ValueError("frozen source range exceeds the source media duration")
            analysis_input = _resolve_source(case["input_path"])
            if _sha256_file(analysis_input) != case["input_sha256"]:
                raise ValueError("analysis-input hash differs from frozen protocol")
            analysis_duration = _source_duration(analysis_input)
            expected_duration = float(planned["end_seconds"]) - float(planned["start_seconds"])
            if not math.isclose(analysis_duration, expected_duration, abs_tol=0.05):
                raise ValueError("analysis input duration differs from the frozen source range")
            resolved[case_id] = analysis_input
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            attempts.append(
                {
                    "case_id": case_id,
                    **({"non_blind": True} if case.get("non_blind") is True else {}),
                    "source_sha256": case.get("source_sha256"),
                    "input_sha256": case.get("input_sha256"),
                    "requested_range": deepcopy(case.get("source_range")),
                    "attempted": False,
                    "execution_status": "not_attempted",
                    "job_status": "not_started",
                    "routing_outcome": "indeterminate",
                    "processed_ranges": [],
                    "predictions": None,
                    "errors": [{"code": "INPUT_PREFLIGHT_FAILED", "detail": type(exc).__name__}],
                    "resource": None,
                }
            )
    return resolved, attempts


def _metal_preflight() -> tuple[bool, str | None]:
    script = (
        "import mlx.core as mx; value = mx.array([1.0], dtype=mx.float32); "
        "mx.eval(value); print('metal-ready')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, type(exc).__name__
    if result.returncode == 0 and "metal-ready" in result.stdout:
        return True, None
    combined = result.stderr + result.stdout
    if "No Metal device available" in combined:
        return False, "No Metal device available to this process"
    return False, "MLX device preflight failed"


def _not_attempted_cases(
    protocol: dict[str, Any], *, code: str, detail: str | None
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            **({"non_blind": True} if case.get("non_blind") is True else {}),
            "source_sha256": case["source_sha256"],
            "input_sha256": case["input_sha256"],
            "requested_range": deepcopy(case["source_range"]),
            "attempted": False,
            "execution_status": "not_attempted",
            "job_status": "not_started",
            "language": None,
            "routing_outcome": "indeterminate",
            "processed_ranges": [],
            "transcript_coverage": None,
            "predictions": None,
            "errors": [{"code": code, "detail": detail}],
            "resource": None,
            "cancellation_probes": [],
        }
        for case in protocol["cases"]
    ]


def _host_memory_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "physical_memory_bytes": _physical_memory_bytes(),
        "memory_pressure": None,
        "swap_used_bytes": None,
    }
    try:
        result = subprocess.run(
            ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            snapshot["memory_pressure"] = lines[-1] if lines else None
            percentages = re.findall(r"([0-9]+(?:\.[0-9]+)?)%", result.stdout)
            if percentages:
                snapshot["memory_pressure_free_percent"] = float(percentages[-1])
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        result = subprocess.run(
            ["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=3
        )
        match = re.search(r"used\s*=\s*([0-9.]+)\s*([KMGTP])", result.stdout, re.I)
        if result.returncode == 0 and match:
            multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}
            snapshot["swap_used_bytes"] = int(float(match.group(1)) * multiplier[match.group(2).upper()])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return snapshot


def _monitor_child(command: list[str], *, cwd: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    before = _host_memory_snapshot()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: Queue[str | None] = Queue()

    def read_lines() -> None:
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_lines, name=f"acceptance-output-{label}", daemon=True)
    reader.start()
    samples = [before]
    saw_eof = False
    cancellation_requested = False
    cancellation_requested_at: float | None = None
    process_was_running_at_cancel = False
    cancellation_stopped_at: float | None = None
    termination_sent_at: float | None = None
    last_progress_stage: str | None = None
    result_event: dict[str, Any] = {}
    while process.poll() is None or not saw_eof:
        try:
            line = lines.get(timeout=0.5)
        except Empty:
            line = None
        except KeyboardInterrupt:
            if not cancellation_requested:
                cancellation_requested = True
                process_was_running_at_cancel = process.poll() is None
                cancellation_requested_at = time.monotonic()
                if process_was_running_at_cancel:
                    try:
                        process.send_signal(signal.SIGINT)
                    except (OSError, ProcessLookupError):
                        pass
            line = None
        now = time.monotonic()
        if (
            cancellation_requested
            and process_was_running_at_cancel
            and cancellation_stopped_at is None
            and process.poll() is not None
        ):
            cancellation_stopped_at = now
        if cancellation_requested and process.poll() is None:
            if (
                termination_sent_at is None
                and cancellation_requested_at is not None
                and now - cancellation_requested_at >= 10.0
            ):
                try:
                    process.terminate()
                except (OSError, ProcessLookupError):
                    pass
                termination_sent_at = now
            elif termination_sent_at is not None and now - termination_sent_at >= 5.0:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
                termination_sent_at = now
        if line is None:
            if process.poll() is not None:
                saw_eof = True
        else:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") in {"initializing_models", "progress", "cancellation_requested"}:
                if isinstance(event.get("stage"), str):
                    last_progress_stage = event["stage"]
                print(
                    json.dumps(
                        {
                            "case_id": label,
                            "event": event.get("event"),
                            "stage": event.get("stage"),
                            "percent": event.get("percent"),
                            "message": event.get("message"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            elif event.get("event") == "result":
                result_event = event
        if time.monotonic() - float(samples[-1].get("_sample_time", 0)) >= 1.0:
            current = _host_memory_snapshot()
            current["_sample_time"] = time.monotonic()
            samples.append(current)
    reader.join(timeout=1)
    process.stdout.close()
    after = _host_memory_snapshot()
    clean_samples = [
        {key: value for key, value in sample.items() if key != "_sample_time"}
        for sample in samples
    ]
    pressure_values = [
        item.get("memory_pressure_free_percent")
        for item in clean_samples
        if isinstance(item.get("memory_pressure_free_percent"), (int, float))
    ]
    swap_values = [item.get("swap_used_bytes") for item in clean_samples if isinstance(item.get("swap_used_bytes"), int)]
    host = {
        "before": {key: value for key, value in before.items() if key != "_sample_time"},
        "after": after,
        "pressure_samples": pressure_values,
        "pressure_text_samples": [
            item.get("memory_pressure")
            for item in clean_samples
            if item.get("memory_pressure")
        ],
        "swap_peak_bytes": max(swap_values) if swap_values else None,
    }
    result = result_event
    result["_exit_code"] = process.returncode
    result["_cancellation_requested"] = cancellation_requested
    result["_cancel_stage"] = last_progress_stage
    result["_cancel_request_to_stop_seconds"] = (
        None
        if cancellation_requested_at is None or cancellation_stopped_at is None
        else cancellation_stopped_at - cancellation_requested_at
    )
    result["_process_stopped_after_cancel"] = (
        cancellation_requested
        and process_was_running_at_cancel
        and cancellation_stopped_at is not None
    )
    return result, host


def _union_duration(ranges: Sequence[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(item["start_seconds"]), float(item["end_seconds"]))
        for item in ranges
        if isinstance(item, dict)
        and isinstance(item.get("start_seconds"), (int, float))
        and isinstance(item.get("end_seconds"), (int, float))
        and item["end_seconds"] > item["start_seconds"]
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for left, right in intervals[1:]:
        if left <= end:
            end = max(end, right)
        else:
            total += end - start
            start, end = left, right
    return total + end - start


def _prediction_and_evidence(record: Any, taxonomy: Sequence[Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    timeline = json.loads(record.model_dump_json()).get("text_timeline", [])
    content = record.ortb_content or {}
    category_ids = content.get("cat", ())
    if not isinstance(category_ids, (list, tuple)) or not category_ids:
        return None, timeline
    names = {item.id: item.name for item in taxonomy}
    segments = content.get("data", [])
    segment_items = (
        segments[0].get("segment", [])
        if segments and isinstance(segments[0], Mapping)
        else []
    )
    sentiment: list[dict[str, object]] = []
    entities: list[dict[str, str]] = []
    for item in segment_items:
        if not isinstance(item, Mapping):
            continue
        name, value = item.get("id"), item.get("value")
        if not isinstance(value, str):
            continue
        if name == "sentiment" and ":" in value:
            label, raw_score = value.split(":", maxsplit=1)
            try:
                sentiment.append({"label": label, "score": float(raw_score)})
            except ValueError:
                sentiment.append({"label": label, "score": None})
        elif isinstance(name, str) and name.startswith("entity_"):
            entities.append({"type": name.removeprefix("entity_"), "text": value})
    keywords = content.get("kwarray", [])
    predictions = {
        "primary_category": category_ids[0],
        "primary_name": names.get(category_ids[0], "Unknown taxonomy ID"),
        "secondary_categories": list(category_ids[1:]),
        "secondary_names": [names.get(item, "Unknown taxonomy ID") for item in category_ids[1:]],
        "keywords": list(keywords) if isinstance(keywords, (list, tuple)) else [],
        "entities": entities,
        "sentiment": sentiment,
    }
    return predictions, timeline


def _offset_timeline_item(item: dict[str, Any], offset_seconds: float) -> dict[str, Any]:
    shifted = deepcopy(item)
    time_range = shifted.get("time_range")
    if isinstance(time_range, dict):
        time_range["start_seconds"] = float(time_range["start_seconds"]) + offset_seconds
        time_range["end_seconds"] = float(time_range["end_seconds"]) + offset_seconds
    evidence = shifted.get("evidence_ranges")
    if isinstance(evidence, list):
        for source_range in evidence:
            if isinstance(source_range, dict):
                source_range["start_seconds"] = float(source_range["start_seconds"]) + offset_seconds
                source_range["end_seconds"] = float(source_range["end_seconds"]) + offset_seconds
    return shifted


def _topic_citation_ranges(
    windows: Sequence[Any], diagnostics: Sequence[dict[str, Any]], source_offset: float
) -> list[dict[str, Any]]:
    """Resolve saved category citation indices against the exact transcript chunks."""
    from engine.signals.classify import analysis_windows_from_run
    from engine.transcribe.types import TranscriptionRun

    analysis = analysis_windows_from_run(TranscriptionRun(windows=tuple(windows), status="ready"))
    by_range = {
        (item.time_range.start_seconds, item.time_range.end_seconds): item
        for item in analysis
    }
    citations: list[dict[str, Any]] = []
    for event in diagnostics:
        if event.get("kind") != "result":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("policy_version") != "english-audio-topic-sufficiency-v1":
            continue
        time_range = payload.get("time_range")
        if not isinstance(time_range, dict):
            raise ValueError("saved classification result lacks a time range")
        window = by_range.get((time_range.get("start_seconds"), time_range.get("end_seconds")))
        if window is None:
            raise ValueError("saved classification range does not match transcript")
        for citation in payload.get("category_citations", []):
            if not isinstance(citation, dict) or not isinstance(citation.get("segment_indices"), list):
                raise ValueError("saved category citation is malformed")
            indices = citation["segment_indices"]
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(window.segments) for index in indices):
                raise ValueError("saved category citation index exceeds transcript")
            selected_payload = payload.get("payload")
            if not isinstance(selected_payload, dict):
                raise ValueError("saved category result lacks its selected categories")
            category_id = citation.get("category_id")
            if category_id == selected_payload.get("primary_category"):
                role = "primary"
            elif category_id in selected_payload.get("secondary_categories", []):
                role = "secondary"
            else:
                raise ValueError("saved citation category does not match its result role")
            citations.append({
                "category_id": category_id,
                "role": role,
                "analysis_range": {
                    "start_seconds": time_range["start_seconds"] + source_offset,
                    "end_seconds": time_range["end_seconds"] + source_offset,
                },
                "segment_indices": indices,
                "citation_ranges": [
                    {
                        "start_seconds": window.segments[index].time_range.start_seconds + source_offset,
                        "end_seconds": window.segments[index].time_range.end_seconds + source_offset,
                    }
                    for index in indices
                ],
            })
    return citations


def _build_attempt(case: dict[str, Any], result: dict[str, Any], host: dict[str, Any], jobs_dir: Path) -> dict[str, Any]:
    case_id = case["case_id"]
    summary = result.get("result")
    if not isinstance(summary, dict):
        summary = result
    job_id = summary.get("job_id")
    report: dict[str, Any] = {
        "case_id": case_id,
        "source_sha256": case["source_sha256"],
        "input_sha256": case["input_sha256"],
        "requested_range": deepcopy(case["source_range"]),
        "attempted": bool(job_id),
        "execution_status": "failed",
        "job_status": summary.get("status", "failed_before_job_creation"),
        "language": summary.get("language"),
        "routing_outcome": "indeterminate",
        "processed_ranges": [],
        "transcript_coverage": None,
        "predictions": None,
        "window_predictions": [],
        "candidate_coverage": None,
        "timestamp_correction_frequency": None,
        "export_byte_identical": None,
        "errors": [],
        "resource": {
            "wall_seconds": summary.get("measurements", {}).get("wall_seconds"),
            "peak_process_rss_bytes": summary.get("measurements", {}).get("peak_process_rss_bytes"),
            "mlx_peak_memory_bytes": (summary.get("measurements", {}).get("mlx_memory") or {}).get("peak_memory_bytes"),
            "host_memory_pressure_before": host.get("before", {}).get("memory_pressure"),
            "host_memory_pressure_samples_percent": [
                100.0 - value for value in host.get("pressure_samples", [])
            ],
            "host_memory_pressure_peak_percent": (
                100.0 - min(host.get("pressure_samples", []))
                if host.get("pressure_samples") else None
            ),
            "swap_bytes_before": host.get("before", {}).get("swap_used_bytes"),
            "swap_bytes_after": host.get("after", {}).get("swap_used_bytes"),
            "swap_peak_bytes": host.get("swap_peak_bytes"),
        },
        "cancellation_probes": [],
    }
    if case.get("non_blind") is True:
        report.update({
            "non_blind": True,
            "selection_provenance": case.get("selection_provenance"),
            "review_provenance": "pending_post_result_review",
            "outcome": None,
            "topic_policy": None,
            "category_citations": [],
            "transcript_windows": [],
            "classification_results": [],
        })
    if not job_id:
        code = summary.get("error", "PROBE_FAILED_BEFORE_JOB_CREATION")
        report["errors"] = [{"code": str(code)}]
        return report
    try:
        from engine.lifecycle import JobStage
        from engine.ortb.export import serialize_content
        from engine.store import JobStore
        from engine.standards import load_taxonomy

        store = JobStore(jobs_dir)
        record = store.load(str(job_id))
        windows = store.load_transcription_windows(str(job_id))
        diagnostics = store.load_analysis_diagnostics(str(job_id))
        source_offset = float(case["source_range"]["start_seconds"])
        processed = [
            {
                "start_seconds": source_offset + item.time_range.start_seconds,
                "end_seconds": source_offset + item.time_range.end_seconds,
            }
            for item in windows
        ]
        transcript_ranges = [
            segment.time_range.model_dump(mode="json")
            for item in windows
            for segment in item.segments
        ]
        failed = [
            {
                "range": {
                    "start_seconds": source_offset + item.time_range.start_seconds,
                    "end_seconds": source_offset + item.time_range.end_seconds,
                },
                "error_code": item.error_code,
                "failure_stage": item.failure_stage,
                "failure_reason": item.failure_reason,
            }
            for item in windows
            if item.error_code is not None
        ]
        report["errors"] = [
            {
                "code": str(entry["error_code"] or "WINDOW_FAILED"),
                "time_range": entry["range"],
                "failure_stage": entry["failure_stage"],
                "failure_reason": entry["failure_reason"],
            }
            for entry in failed
        ]
        report["processed_ranges"] = processed
        report["transcript_coverage"] = {
            "source_seconds": case["source_range"]["end_seconds"] - case["source_range"]["start_seconds"],
            "transcribed_seconds": _union_duration(transcript_ranges),
            "window_count": len(windows),
            "failed_windows": failed,
        }
        candidate_events = [
            item.get("payload", {})
            for item in diagnostics
            if item.get("kind") == "candidates" and isinstance(item.get("payload"), dict)
        ]
        candidate_ids = sorted(
            {
                candidate.get("taxonomy_id")
                for event in candidate_events
                for candidate in event.get("candidates", [])
                if isinstance(candidate, dict)
                and isinstance(candidate.get("taxonomy_id"), str)
            }
        )
        generation_events = [
            item.get("payload", {})
            for item in diagnostics
            if item.get("kind") == "attempt" and isinstance(item.get("payload"), dict)
        ]
        if case.get("non_blind") is True:
            from engine.signals.classify import analysis_windows_from_run
            from engine.transcribe.types import TranscriptionRun

            policy_events = [item.get("payload") for item in diagnostics if item.get("kind") == "topic_policy"]
            report["topic_policy"] = policy_events[-1] if policy_events else None
            report["transcript_windows"] = [
                {
                    "time_range": _offset_timeline_item({"time_range": item.time_range.model_dump(mode="json")}, source_offset)["time_range"],
                    "segment_ranges": [
                        _offset_timeline_item({"time_range": segment.time_range.model_dump(mode="json")}, source_offset)["time_range"]
                        for segment in item.segments
                    ],
                }
                for item in analysis_windows_from_run(TranscriptionRun(windows=tuple(windows), status="ready"))
            ]
            report["classification_results"] = [
                _offset_timeline_item(item["payload"], source_offset)
                for item in diagnostics if item.get("kind") == "result" and isinstance(item.get("payload"), dict)
            ]
            report["category_citations"] = _topic_citation_ranges(windows, diagnostics, source_offset)
        correction_count = sum(len(item.timestamp_corrections) for item in windows)
        report["candidate_coverage"] = {
            "classification_windows_with_candidates": len(candidate_events),
            "candidate_ids_seen": candidate_ids,
            "unique_candidate_count": len(candidate_ids),
            "generation_attempt_count": len(generation_events),
            "generation_attempt_error_count": sum(
                item.get("error_code") is not None for item in generation_events
            ),
            "classification_result_count": sum(item.get("kind") == "result" for item in diagnostics),
        }
        report["timestamp_correction_frequency"] = (
            None if not windows else correction_count / len(windows)
        )
        taxonomy = load_taxonomy()
        predictions, timeline = _prediction_and_evidence(record, taxonomy)
        report["predictions"] = predictions
        report["window_predictions"] = [
            _offset_timeline_item(item, source_offset)
            for item in timeline
            if isinstance(item, dict)
        ]
        report["provenance"] = None if record.provenance is None else record.provenance.model_dump(mode="json")
        if record.status is JobStage.COMPLETE:
            normalized_content = json.loads(record.model_dump_json()).get("ortb_content") or {}
            report["export_byte_identical"] = (
                store.read_export(str(job_id)) == serialize_content(normalized_content)
            )
            report["execution_status"] = "complete"
            report["routing_outcome"] = "accepted_english" if record.language == "en" else "indeterminate"
            if case.get("non_blind") is True:
                report["outcome"] = "category"
        elif record.status is JobStage.CANCELLED:
            report["execution_status"] = "cancelled"
            report["errors"] = [{"code": str(record.error or "USER_CANCELLED")}]
        elif windows and all(item.failure_reason in {"vocal_presence_uncertain", "non_english_or_unconfirmed_language"} for item in windows):
            report["execution_status"] = "handled_out_of_scope"
            reasons = {item.failure_reason for item in windows}
            report["routing_outcome"] = (
                "rejected_non_english" if reasons == {"non_english_or_unconfirmed_language"}
                else "rejected_unconfirmed_vocal" if reasons == {"vocal_presence_uncertain"}
                else "indeterminate"
            )
        else:
            report["execution_status"] = "failed"
            report["errors"] = [{"code": str(record.error or "JOB_FAILED")}]
            if case.get("non_blind") is True and record.error == "INSUFFICIENT_TOPIC_EVIDENCE" and isinstance(report.get("topic_policy"), dict) and report["topic_policy"].get("outcome") == "insufficient":
                report["execution_status"] = "handled_topic_insufficiency"
                report["outcome"] = "insufficient_topic_evidence"
                report["routing_outcome"] = "accepted_english" if record.language == "en" else "indeterminate"
    except Exception as exc:
        report["execution_status"] = "failed"
        report["errors"] = [{"code": type(exc).__name__}]
    return report


def run_live(protocol: dict[str, Any], output_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    errors = validate_protocol(protocol)
    if errors:
        return [], {"preflight": "failed", "errors": errors}
    minimum_memory = protocol["resource_requirements"]["physical_memory_bytes_minimum"]
    host_ready, host_evidence, host_failures = _host_preflight(minimum_memory)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifacts_root = output_path.parent / f"{output_path.stem}-artifacts-{run_id}"
    if not host_ready:
        detail = "; ".join(host_failures)
        return _not_attempted_cases(protocol, code="UNSUPPORTED_HOST", detail=detail), {
            "preflight": "blocked",
            "reason": "UNSUPPORTED_HOST",
            "detail": detail,
            "host": host_evidence,
            "host_physical_memory_bytes": host_evidence["physical_memory_bytes"],
            "model_downloads": 0,
        }
    resolved, input_failures = _verify_inputs(protocol)
    if input_failures:
        return input_failures, {"preflight": "failed", "reason": "input_hash_or_range_mismatch"}
    metal_ready, metal_error = _metal_preflight()
    if not metal_ready:
        return _not_attempted_cases(protocol, code="METAL_UNAVAILABLE", detail=metal_error), {
            "preflight": "blocked",
            "reason": "METAL_UNAVAILABLE",
            "detail": metal_error,
            "host": host_evidence,
            "host_physical_memory_bytes": host_evidence["physical_memory_bytes"],
            "model_downloads": 0,
        }

    attempts: list[dict[str, Any]] = []
    probe_script = ROOT / "scripts" / "run_pipeline_probe.py"
    artifacts_root.mkdir(parents=True, exist_ok=False)
    cold_warm_timing: dict[str, Any] | None = None
    for case_index, case in enumerate(protocol["cases"]):
        case_id = case["case_id"]
        try:
            jobs_dir = artifacts_root / case_id / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, str(probe_script), str(resolved[case_id]), "--jobs-dir", str(jobs_dir)]
            if case.get("context_override") is not None:
                command.extend(["--context", str(case["context_override"])])
            result, host = _monitor_child(command, cwd=ROOT, label=case_id)
            attempt = _build_attempt(case, result, host, jobs_dir)
            cold_cancelled = result.get("_cancellation_requested") is True or (
                isinstance(result.get("result"), dict)
                and result["result"].get("status") == "cancelled"
            ) or result.get("status") == "cancelled"
            cancel_result = result if cold_cancelled else None
            warm_cancelled = False
            if case_index == 0 and not cold_cancelled and attempt["execution_status"] == "complete":
                warm_jobs_dir = artifacts_root / "warm-cache-probe" / "jobs"
                warm_jobs_dir.mkdir(parents=True, exist_ok=False)
                warm_command = [
                    sys.executable,
                    str(probe_script),
                    str(resolved[case_id]),
                    "--jobs-dir",
                    str(warm_jobs_dir),
                ]
                if case.get("context_override") is not None:
                    warm_command.extend(["--context", str(case["context_override"])])
                warm_result, _warm_host = _monitor_child(
                    warm_command, cwd=ROOT, label=f"{case_id}-warm"
                )
                warm_summary = warm_result.get("result")
                if not isinstance(warm_summary, dict):
                    warm_summary = warm_result
                cold_measurements = result.get("result")
                if not isinstance(cold_measurements, dict):
                    cold_measurements = result
                cold_measurements = cold_measurements.get("measurements", {})
                warm_measurements = warm_summary.get("measurements", {})
                warm_cancelled = warm_result.get("_cancellation_requested") is True or (
                    isinstance(warm_result.get("result"), dict)
                    and warm_result["result"].get("status") == "cancelled"
                ) or warm_result.get("status") == "cancelled"
                cold_warm_timing = {
                    "case_id": case_id,
                    "cold_wall_seconds": cold_measurements.get("wall_seconds"),
                    "warm_wall_seconds": warm_measurements.get("wall_seconds"),
                    "warm_probe_status": warm_summary.get("status"),
                    "warm_cache_definition": (
                        "second fresh process over identical input and local model assets; "
                        "OS/file cache may be warm, but model weights are not resident in-process"
                    ),
                }
                if warm_cancelled:
                    cancel_result = warm_result
            cancelled = cold_cancelled or warm_cancelled
            if cold_cancelled:
                attempt["execution_status"] = "cancelled"
                if not attempt["errors"] or all(
                    error.get("code") == "JOB_FAILED" for error in attempt["errors"]
                ):
                    attempt["errors"] = [{"code": "USER_CANCELLED"}]
            if cancelled and cancel_result is not None:
                stage = cancel_result.get("_cancel_stage")
                stage_aliases = {
                    "transcribing": "asr",
                    "analysing": "classification",
                    "extracting_audio": "extraction",
                    "extracting": "extraction",
                }
                stage = stage_aliases.get(stage, stage)
                if stage in _CANCELLATION_STAGES:
                    attempt["cancellation_probes"].append(
                        {
                            "stage": stage,
                            "outcome": "cancelled",
                            "process_stopped": cancel_result.get("_process_stopped_after_cancel") is True,
                            "request_to_stop_seconds": cancel_result.get("_cancel_request_to_stop_seconds"),
                        }
                    )
            attempts.append(attempt)
            if cancelled:
                attempts.extend(
                    _not_attempted_cases(
                        {"cases": protocol["cases"][case_index + 1 :]},
                        code="RUN_CANCELLED_BY_USER",
                        detail="remaining batch cases were not launched after cancellation",
                    )
                )
                return attempts, {
                    "preflight": "passed",
                    "reason": "USER_CANCELLED",
                    "cancelled_by_user": True,
                    "run_id": run_id,
                    "host_physical_memory_bytes": host_evidence["physical_memory_bytes"],
                    "cold_warm_timing": cold_warm_timing,
                    "model_downloads": 0,
                    "artifacts_root": str(artifacts_root),
                }
        except KeyboardInterrupt:
            attempts.extend(
                _not_attempted_cases(
                    {"cases": protocol["cases"][case_index:]},
                    code="RUN_CANCELLED_BY_USER",
                    detail="remaining batch cases were not launched after cancellation",
                )
            )
            return attempts, {
                "preflight": "passed",
                "reason": "USER_CANCELLED",
                "cancelled_by_user": True,
                "cancellation_location": "outside_child_monitor",
                "run_id": run_id,
                "host_physical_memory_bytes": host_evidence["physical_memory_bytes"],
                "cold_warm_timing": cold_warm_timing,
                "model_downloads": 0,
                "artifacts_root": str(artifacts_root),
            }
    return attempts, {
        "preflight": "passed",
        "run_id": run_id,
        "host_physical_memory_bytes": host_evidence["physical_memory_bytes"],
        "cold_warm_timing": cold_warm_timing,
        "host": _host_memory_snapshot(),
        "model_downloads": 0,
        "artifacts_root": str(artifacts_root),
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or audit qualitative English Phase 1 acceptance evidence.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attempts", type=Path, help="score saved attempt records without running models")
    parser.add_argument("--post-result-reviews", type=Path, help="separate qualitative reviews; never treated as blind labels")
    parser.add_argument("--no-run", action="store_true", help="audit the protocol without loading models")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = _read_json(args.protocol)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read acceptance protocol: {type(exc).__name__}", file=sys.stderr)
        return 2
    attempts: list[dict[str, Any]] = []
    run_metadata: dict[str, Any] = {"execution_mode": "audit_only"}
    attempt_input_error: str | None = None
    if args.attempts:
        try:
            raw_attempts = _read_json(args.attempts)
            if isinstance(raw_attempts, list):
                attempts = raw_attempts
            elif isinstance(raw_attempts, dict):
                saved_attempts = raw_attempts.get("attempts")
                if isinstance(saved_attempts, list):
                    attempts = saved_attempts
                else:
                    attempt_input_error = "saved attempts report must contain an attempts array"
                saved_metadata = raw_attempts.get("run_metadata")
                if isinstance(saved_metadata, dict):
                    run_metadata.update(saved_metadata)
            else:
                print("attempts input must be a JSON array or report object", file=sys.stderr)
                return 2
            run_metadata["execution_mode"] = "saved_attempts"
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            print(f"cannot read attempts: {type(exc).__name__}", file=sys.stderr)
            return 2
        if attempt_input_error is not None:
            prior_errors = run_metadata.get("evidence_input_errors", [])
            if not isinstance(prior_errors, list) or any(
                not isinstance(item, str) for item in prior_errors
            ):
                prior_errors = []
            run_metadata["evidence_input_errors"] = [*prior_errors, attempt_input_error]
    elif not args.no_run:
        attempts, run_metadata = run_live(protocol, args.output)
    reviews: list[dict[str, Any]] = []
    if args.post_result_reviews:
        try:
            raw_reviews = _read_json(args.post_result_reviews)
            if isinstance(raw_reviews, list):
                reviews = raw_reviews
            elif isinstance(raw_reviews, dict) and isinstance(raw_reviews.get("reviews"), list):
                reviews = raw_reviews["reviews"]
            else:
                prior_errors = run_metadata.get("evidence_input_errors", [])
                if not isinstance(prior_errors, list) or any(
                    not isinstance(item, str) for item in prior_errors
                ):
                    prior_errors = []
                run_metadata["evidence_input_errors"] = [
                    *prior_errors,
                    "post-result review report must contain a reviews array",
                ]
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            print(f"cannot read post-result reviews: {type(exc).__name__}", file=sys.stderr)
            return 2
    report = evaluate_protocol(protocol, attempts, post_result_reviews=reviews, run_metadata=run_metadata)
    try:
        _write_json_exclusive(args.output, report)
    except FileExistsError:
        print("refusing to overwrite existing acceptance output", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"cannot write acceptance output: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps({"release_decision": report["release_decision"], "output": str(args.output)}))
    return 0 if report["release_decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
