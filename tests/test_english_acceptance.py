from __future__ import annotations

import importlib.util
from copy import deepcopy
import json
from pathlib import Path

import pytest


def _load_harness():
    path = Path(__file__).parents[1] / "scripts" / "english_acceptance.py"
    spec = importlib.util.spec_from_file_location("english_acceptance_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol(harness, cases: list[dict[str, object]] | None = None):
    value = {
        "schema_version": 1,
        "protocol_id": "english-phase1-acceptance-v1",
        "frozen_before_inference": True,
        "semantic_policy": {
            "method": "qualitative_prediction_first",
            "numeric_accuracy_threshold": None,
            "review_status": "pending",
        },
        "resource_requirements": {
            "physical_memory_bytes_minimum": 64 * 1024**3,
            "live_cancellation_stages": ["extraction", "asr", "classification", "batch"],
        },
        "required_strata": ["english_speech", "english_lyrics"],
        "cases": cases
        or [
            {
                "case_id": "speech-01",
                "display_name": "English speech fixture",
                "stratum": "english_speech",
                "source_path": "speech-source.mp4",
                "input_path": "speech-input.wav",
                "source_sha256": "a" * 64,
                "input_sha256": "c" * 64,
                "analysis_duration_seconds": 10.0,
                "source_range": {"start_seconds": 0.0, "end_seconds": 10.0},
                "routing_reference": None,
            },
            {
                "case_id": "song-01",
                "display_name": "English lyrics fixture",
                "stratum": "english_lyrics",
                "source_path": "song-source.mp4",
                "input_path": "song-input.wav",
                "source_sha256": "b" * 64,
                "input_sha256": "d" * 64,
                "analysis_duration_seconds": 20.0,
                "source_range": {"start_seconds": 0.0, "end_seconds": 20.0},
                "routing_reference": None,
            },
        ],
    }
    value["protocol_sha256"] = harness.protocol_digest(value)
    harness._FROZEN_PROTOCOL_SHA256 = value["protocol_sha256"]
    return value


def _attempt(case: dict[str, object], *, status: str = "complete"):
    source_range = case["source_range"]
    return {
        "case_id": case["case_id"],
        "source_sha256": case["source_sha256"],
        "input_sha256": case["input_sha256"],
        "requested_range": source_range,
        "attempted": True,
        "execution_status": "complete" if status == "complete" else "failed",
        "job_status": status,
        "language": "en" if status == "complete" else None,
        "routing_outcome": "accepted_english" if status == "complete" else "rejected",
        "processed_ranges": [source_range] if status == "complete" else [],
        "predictions": (
            {
                "primary_category": "150",
                "secondary_categories": ["151"],
                "keywords": [],
                "entities": [],
                "sentiment": [{"label": "neutral", "score": 0.0}],
            }
            if status == "complete"
            else None
        ),
        "errors": [] if status == "complete" else [{"code": "MODEL_FAILURE"}],
        "resource": {
            "wall_seconds": 12.0,
            "peak_process_rss_bytes": 1_000_000,
            "mlx_peak_memory_bytes": 2_000_000,
            "host_memory_pressure_before": "normal",
            "host_memory_pressure_samples_percent": [35.0, 37.0],
            "host_memory_pressure_peak_percent": 35.0,
            "swap_bytes_before": 0,
            "swap_peak_bytes": 0,
            "swap_bytes_after": 0,
            "cancellation_probe": "not_run",
        },
    }


def _review(harness, case, attempt, *, decision="approved"):
    return {
        "case_id": case["case_id"],
        "review": decision,
        "provenance": "post_result_user_review",
        "prediction_sha256": harness.prediction_review_digest(case["case_id"], attempt),
    }


def _topic_protocol(harness, monkeypatch):
    ids = (
        "tv-minus01-2-dialogue-600-720",
        "tv-minus01-corrective-60-180",
        "backstreet-vocal-90-120",
        "sony-camera-speech-30-90",
        "hario-coffee-speech-120-300",
        "whitney-houston-full-audio",
    )
    cases = []
    for index, case_id in enumerate(ids):
        case = deepcopy(_protocol(harness)["cases"][0])
        case.update(
            case_id=case_id,
            stratum=case_id,
            source_sha256=format(index + 1, "064x"),
            input_sha256=format(index + 11, "064x"),
            non_blind=True,
            selection_provenance="previously reviewed development material",
            routing_reference=None,
            semantic_reference=None,
            expected_topic_routing="undetermined",
        )
        if case_id in {"sony-camera-speech-30-90", "hario-coffee-speech-120-300"}:
            case["expected_topic_routing"] = "topical_positive"
        cases.append(case)
    protocol = _protocol(harness, cases)
    protocol["protocol_id"] = "english-phase1-topic-sufficiency-v1"
    protocol["required_strata"] = list(ids)
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)
    monkeypatch.setattr(harness, "_TOPIC_SUFFICIENCY_SHA256", protocol["protocol_sha256"], raising=False)
    return protocol


def _topic_attempt(case, *, outcome="insufficient_topic_evidence"):
    attempt = _attempt(case)
    category = outcome == "category"
    start = case["source_range"]["start_seconds"]
    duration = case["source_range"]["end_seconds"] - start
    attempt.update(
        non_blind=True,
        outcome=outcome,
        job_status="complete" if category else "failed",
        execution_status="complete" if category else "handled_topic_insufficiency",
        predictions=attempt["predictions"] if category else None,
        errors=[] if category else [{"code": "INSUFFICIENT_TOPIC_EVIDENCE"}],
        topic_policy={
            "policy_version": "english-audio-aggregation-v2",
            "outcome": "eligible" if category else "insufficient",
            "primary_category": "150" if category else None,
            "transcript_seconds": 0.6 * duration,
            "audio_seconds": duration,
            "primary_role_support_seconds": [["150", 0.4 * duration]] if category else [],
            "secondary_role_support_seconds": [["151", 0.2 * duration]] if category else [],
        },
        category_citations=[
            {"category_id": "150", "role": "primary", "analysis_range": case["source_range"], "segment_indices": [0],
             "citation_ranges": [{"start_seconds": 1.0, "end_seconds": 5.0}]},
            {"category_id": "151", "role": "secondary", "analysis_range": case["source_range"], "segment_indices": [0],
             "citation_ranges": [{"start_seconds": 2.0, "end_seconds": 4.0}]},
        ] if category else [],
        export_byte_identical=True if category else None,
        transcript_windows=[{
            "time_range": case["source_range"],
            "segment_ranges": [
                {"start_seconds": start + 0.1 * duration, "end_seconds": start + 0.5 * duration},
                {"start_seconds": start + 0.5 * duration, "end_seconds": start + 0.7 * duration},
            ],
        }],
        classification_results=[{
            "time_range": case["source_range"],
            "policy_version": "english-audio-topic-sufficiency-v1",
            "payload": {"primary_category": "150", "secondary_categories": ["151"]} if category else None,
            "error_code": None if category else "INSUFFICIENT_TOPIC_EVIDENCE",
            "attempts": [
                {
                    "attempt": 1, "stage": "primary", "prompt_sha256": "1" * 64,
                    "raw": json.dumps({"primary_category": "150" if category else "__INSUFFICIENT_TOPIC_EVIDENCE__"}),
                    "finish_reason": "stop", "seconds": 0.5, "token_count": 10,
                    "error_code": None, "reason": None,
                },
                *([{
                    "attempt": 2, "stage": "payload", "prompt_sha256": "2" * 64,
                    "raw": json.dumps({
                        "primary_category": "150", "secondary_categories": ["151"],
                        "keywords": [], "entities": [], "sentiment": {"label": "neutral", "score": 0.0},
                        "category_evidence": [
                            {"category_id": "150", "segment_indices": [0]},
                            {"category_id": "151", "segment_indices": [1]},
                        ],
                    }),
                    "finish_reason": "stop", "seconds": 0.75, "token_count": 20,
                    "error_code": None, "reason": None,
                }] if category else []),
            ],
            "category_citations": [
                {"category_id": "150", "segment_indices": [0]},
                {"category_id": "151", "segment_indices": [1]},
            ] if category else [],
        }],
    )
    if category:
        for citation in attempt["category_citations"]:
            citation["analysis_range"] = case["source_range"]
            citation["segment_indices"] = [0 if citation["role"] == "primary" else 1]
            citation["citation_ranges"] = [
                attempt["transcript_windows"][0]["segment_ranges"][0 if citation["role"] == "primary" else 1]
            ]
    return attempt


@pytest.mark.parametrize("change", [
    "missing", "empty", "malformed", "wrong_stage", "wrong_number", "incomplete",
    "generation_error", "missing_prompt_hash", "missing_timing", "bad_raw",
    "wrong_raw", "extra_raw_field", "duplicate_raw_key", "extra_attempt",
])
def test_topic_abstention_requires_retained_raw_primary_generation(monkeypatch, change):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case) for case in protocol["cases"]]
    result = attempts[0]["classification_results"][0]
    raw = result["attempts"][0]
    if change == "missing":
        result.pop("attempts")
    elif change == "empty":
        result["attempts"] = []
    elif change == "malformed":
        result["attempts"] = [{}]
    elif change == "wrong_stage":
        raw["stage"] = "payload"
    elif change == "wrong_number":
        raw["attempt"] = 2
    elif change == "incomplete":
        raw["finish_reason"] = "length"
    elif change == "generation_error":
        raw["error_code"] = "RuntimeError"
    elif change == "missing_prompt_hash":
        raw["prompt_sha256"] = ""
    elif change == "missing_timing":
        raw["seconds"] = None
    elif change == "bad_raw":
        raw["raw"] = "not json"
    elif change == "wrong_raw":
        raw["raw"] = '{"primary_category":"150"}'
    elif change == "extra_raw_field":
        raw["raw"] = '{"primary_category":"__INSUFFICIENT_TOPIC_EVIDENCE__","extra":1}'
    elif change == "duplicate_raw_key":
        raw["raw"] = '{"primary_category":"__INSUFFICIENT_TOPIC_EVIDENCE__","primary_category":"__INSUFFICIENT_TOPIC_EVIDENCE__"}'
    else:
        result["attempts"].append(deepcopy(raw))
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["cases"][0]["schema_status"] == "failed"
    assert report["gates"]["execution"]["status"] == "failed"


def test_topic_category_requires_retained_generation_attempts(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case, outcome="category") for case in protocol["cases"]]
    attempts[0]["classification_results"][0].pop("attempts")
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["cases"][0]["schema_status"] == "failed"


@pytest.mark.parametrize("change", [
    "bad_job_status", "bad_export", "missing_d", "nonfinite_t", "missing_support",
    "wrong_primary", "bad_classification", "missing_transcript",
])
def test_topic_insufficiency_requires_valid_saved_decision(monkeypatch, change):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case) for case in protocol["cases"]]
    attempt = attempts[0]
    if change == "bad_job_status":
        attempt["job_status"] = "complete"
    elif change == "bad_export":
        attempt["export_byte_identical"] = True
    elif change == "missing_d":
        attempt["topic_policy"].pop("audio_seconds")
    elif change == "nonfinite_t":
        attempt["topic_policy"]["transcript_seconds"] = float("nan")
    elif change == "missing_support":
        attempt["topic_policy"].pop("primary_role_support_seconds")
    elif change == "wrong_primary":
        attempt["topic_policy"]["primary_category"] = "150"
    elif change == "bad_classification":
        attempt["classification_results"][0]["error_code"] = "INVALID_MODEL_OUTPUT"
    else:
        attempt.pop("transcript_windows")
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["gates"]["schema"]["status"] == "failed"
    assert report["gates"]["execution"]["status"] == "failed"
    assert report["release_decision"] == "fail"


@pytest.mark.parametrize("change", [
    "extra_category", "swapped_role", "bad_index", "wrong_range", "missing_transcript",
    "malformed_category_id",
])
def test_topic_eligible_citations_reconcile_with_saved_transcript(monkeypatch, change):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case, outcome="category") for case in protocol["cases"]]
    attempt = attempts[0]
    if change == "extra_category":
        attempt["category_citations"].append({
            "category_id": "999", "role": "secondary", "analysis_range": attempt["requested_range"],
            "segment_indices": [1], "citation_ranges": [attempt["transcript_windows"][0]["segment_ranges"][1]],
        })
    elif change == "swapped_role":
        attempt["category_citations"][0]["role"] = "secondary"
    elif change == "bad_index":
        attempt["category_citations"][0]["segment_indices"] = [2]
    elif change == "wrong_range":
        attempt["category_citations"][0]["citation_ranges"] = [{
            "start_seconds": attempt["requested_range"]["start_seconds"] + 2,
            "end_seconds": attempt["requested_range"]["start_seconds"] + 6,
        }]
    elif change == "malformed_category_id":
        attempt["classification_results"][0]["category_citations"][0]["category_id"] = []
    else:
        attempt.pop("transcript_windows")
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["gates"]["schema"]["status"] == "failed"
    assert report["cases"][0]["schema_status"] == "failed"


def test_topic_protocol_requires_its_code_pin(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    assert harness.validate_protocol(protocol) == []
    changed = deepcopy(protocol)
    changed["protocol_sha256"] = "0" * 64
    assert any("digest" in error for error in harness.validate_protocol(changed))
    monkeypatch.setattr(harness, "_TOPIC_SUFFICIENCY_SHA256", "f" * 64)
    assert any("code-pinned" in error for error in harness.validate_protocol(protocol))


def test_topic_protocol_requires_all_six_non_blind_cases(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    protocol["cases"].pop()
    protocol["required_strata"].pop()
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)
    monkeypatch.setattr(harness, "_TOPIC_SUFFICIENCY_SHA256", protocol["protocol_sha256"])
    assert any("six" in error for error in harness.validate_protocol(protocol))


def test_topic_abstention_is_distinct_from_invalid_json_and_not_blind_accuracy(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case, outcome="category") if case.get("expected_topic_routing") == "topical_positive" else _topic_attempt(case) for case in protocol["cases"]]
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["blind_semantic_denominator"] == 0
    assert report["development_cases"][0]["outcome"] == "insufficient_topic_evidence"
    assert report["cases"][0]["schema_status"] == "not_applicable"
    assert report["gates"]["execution"]["status"] == "passed"
    assert report["release_decision"] == "pending"


def test_topic_insufficiency_keeps_supported_window_below_whole_file_cutoff(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case) for case in protocol["cases"]]
    retained = attempts[0]
    duration = retained["topic_policy"]["audio_seconds"]
    retained["classification_results"][0].update(
        payload={"primary_category": "150", "secondary_categories": ["151"]},
        error_code=None,
        category_citations=[
            {"category_id": "150", "segment_indices": [1]},
            {"category_id": "151", "segment_indices": [0]},
        ],
        attempts=deepcopy(_topic_attempt(protocol["cases"][0], outcome="category")["classification_results"][0]["attempts"]),
    )
    raw_payload = json.loads(retained["classification_results"][0]["attempts"][1]["raw"])
    raw_payload["category_evidence"] = retained["classification_results"][0]["category_citations"]
    retained["classification_results"][0]["attempts"][1]["raw"] = json.dumps(raw_payload)
    retained["category_citations"] = [
        {"category_id": "150", "role": "primary", "analysis_range": retained["requested_range"],
         "segment_indices": [1], "citation_ranges": [retained["transcript_windows"][0]["segment_ranges"][1]]},
        {"category_id": "151", "role": "secondary", "analysis_range": retained["requested_range"],
         "segment_indices": [0], "citation_ranges": [retained["transcript_windows"][0]["segment_ranges"][0]]},
    ]
    retained["topic_policy"]["primary_role_support_seconds"] = [["150", 0.2 * duration]]
    retained["topic_policy"]["secondary_role_support_seconds"] = [["151", 0.4 * duration]]
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["cases"][0]["schema_status"] == "not_applicable"
    assert report["gates"]["execution"]["status"] == "passed"


def test_topic_positive_control_abstention_fails_semantic_gate(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    report = harness.evaluate_protocol(protocol, [_topic_attempt(case) for case in protocol["cases"]])
    assert report["gates"]["semantic"]["status"] == "failed"
    assert "sony-camera-speech-30-90" in report["gates"]["semantic"]["false_abstention_case_ids"]
    assert report["release_decision"] == "fail"


def test_topic_attempt_cannot_relabel_frozen_development_case_as_blind(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case, outcome="category") for case in protocol["cases"]]
    attempts[0]["non_blind"] = False
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["gates"]["coverage"]["status"] == "failed"
    assert report["blind_semantic_denominator"] == 0
    assert report["release_decision"] == "fail"


def test_topic_metal_block_keeps_known_cases_pending_not_blind(monkeypatch):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = harness._not_attempted_cases(protocol, code="METAL_UNAVAILABLE", detail="device unavailable")
    report = harness.evaluate_protocol(protocol, attempts, run_metadata={"reason": "METAL_UNAVAILABLE"})
    assert report["gates"]["coverage"]["status"] == "pending"
    assert report["gates"]["routing"]["status"] == "pending"
    assert report["blind_semantic_denominator"] == 0
    assert report["release_decision"] == "pending"


@pytest.mark.parametrize("change", [
    "missing_t", "nonfinite_d", "missing_support", "wrong_primary",
    "wrong_outcome", "missing_citations", "wrong_citation_category", "invalid_citation_range",
    "wrong_support_duration", "missing_role",
])
def test_topic_complete_requires_measured_consistent_policy_and_citations(monkeypatch, change):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case, outcome="category") for case in protocol["cases"]]
    attempt = attempts[0]
    if change == "missing_t":
        attempt["topic_policy"].pop("transcript_seconds")
    elif change == "nonfinite_d":
        attempt["topic_policy"]["audio_seconds"] = float("nan")
    elif change == "missing_support":
        attempt["topic_policy"]["primary_role_support_seconds"] = []
    elif change == "wrong_primary":
        attempt["topic_policy"]["primary_category"] = "151"
    elif change == "wrong_outcome":
        attempt["outcome"] = "insufficient_topic_evidence"
    elif change == "missing_citations":
        attempt["category_citations"] = []
    elif change == "wrong_citation_category":
        attempt["category_citations"][0]["category_id"] = "999"
    elif change == "wrong_support_duration":
        attempt["topic_policy"]["primary_role_support_seconds"] = [["150", 5.0]]
    elif change == "missing_role":
        attempt["category_citations"][0].pop("role")
    else:
        attempt["category_citations"][0]["citation_ranges"] = [
            {"start_seconds": 9.0, "end_seconds": 12.0}
        ]
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["gates"]["schema"]["status"] == "failed"
    assert report["cases"][0]["schema_status"] == "failed"
    assert report["release_decision"] == "fail"


@pytest.mark.parametrize("routing_outcome", ["rejected_unconfirmed_vocal", "indeterminate"])
@pytest.mark.parametrize("control_index", [3, 4])
def test_predeclared_topical_control_cannot_pass_as_out_of_scope(monkeypatch, routing_outcome, control_index):
    harness = _load_harness()
    protocol = _topic_protocol(harness, monkeypatch)
    attempts = [_topic_attempt(case, outcome="category") for case in protocol["cases"]]
    control = attempts[control_index]
    control.update(
        execution_status="handled_out_of_scope", job_status="failed",
        routing_outcome=routing_outcome, predictions=None, outcome=None,
        topic_policy=None, category_citations=[], export_byte_identical=None,
    )
    report = harness.evaluate_protocol(protocol, attempts)
    assert report["gates"]["routing"]["status"] == "failed"
    assert protocol["cases"][control_index]["case_id"] in report["gates"]["routing"]["failed_topic_positive_case_ids"]
    assert report["release_decision"] == "fail"


def test_topic_citation_ranges_follow_saved_segment_indices():
    from engine.domain import TimeRange, TranscriptSegment
    from engine.transcribe.types import WindowResult

    harness = _load_harness()
    windows = (WindowResult(
        time_range=TimeRange(start_seconds=0, end_seconds=10),
        segments=(
            TranscriptSegment(text="camera", confidence=0.9, language="en", time_range=TimeRange(start_seconds=1, end_seconds=3)),
            TranscriptSegment(text="battery", confidence=0.9, language="en", time_range=TimeRange(start_seconds=4, end_seconds=6)),
        ),
    ),)
    diagnostics = [{"kind": "result", "payload": {
        "time_range": {"start_seconds": 1, "end_seconds": 6},
        "policy_version": "english-audio-topic-sufficiency-v1",
        "payload": {"primary_category": "633", "secondary_categories": []},
        "category_citations": [{"category_id": "633", "segment_indices": [1]}],
    }}]
    assert harness._topic_citation_ranges(windows, diagnostics, 30.0) == [{
        "category_id": "633",
        "role": "primary",
        "analysis_range": {"start_seconds": 31.0, "end_seconds": 36.0},
        "segment_indices": [1],
        "citation_ranges": [{"start_seconds": 34.0, "end_seconds": 36.0}],
    }]


def test_missing_case_fails_coverage_and_is_not_dropped():
    harness = _load_harness()
    protocol = _protocol(harness)

    report = harness.evaluate_protocol(protocol, [_attempt(protocol["cases"][0])])

    assert report["gates"]["coverage"]["status"] == "failed"
    assert report["missing_case_ids"] == ["song-01"]
    assert report["release_decision"] == "fail"


def test_missing_required_stratum_fails_protocol_integrity():
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol["required_strata"].append("tv_or_podcast")
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)

    report = harness.evaluate_protocol(protocol, [])

    assert report["gates"]["protocol_integrity"]["status"] == "failed"
    assert any("required strata" in error for error in report["protocol_errors"])


def test_recomputed_digest_does_not_unfreeze_approved_protocol():
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol["cases"][0]["display_name"] = "changed after approval"
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)

    report = harness.evaluate_protocol(protocol, [])

    assert report["gates"]["protocol_integrity"]["status"] == "failed"
    assert any("code-pinned" in error for error in report["protocol_errors"])


def test_tv_podcast_supplement_is_accepted_only_at_its_frozen_digest():
    harness = _load_harness()
    path = Path(__file__).parents[1] / "benchmarks" / "acceptance-english-tv-podcast-v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert harness.validate_protocol(protocol) == []

    protocol["cases"][0]["display_name"] = "changed after freezing"
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)
    assert any("code-pinned" in error for error in harness.validate_protocol(protocol))


def test_tv_followup_is_accepted_only_at_its_frozen_digest():
    harness = _load_harness()
    path = Path(__file__).parents[1] / "benchmarks" / "acceptance-english-tv-followup-v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert harness.validate_protocol(protocol) == []

    protocol["cases"][0]["display_name"] = "changed after freezing"
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)
    assert any("code-pinned" in error for error in harness.validate_protocol(protocol))


def test_corrective_supplement_is_accepted_only_at_its_frozen_digest():
    harness = _load_harness()
    path = Path(__file__).parents[1] / "benchmarks" / "acceptance-english-corrective-v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert harness.validate_protocol(protocol) == []

    protocol["cases"][0]["display_name"] = "changed after freezing"
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)
    assert any("code-pinned" in error for error in harness.validate_protocol(protocol))


def test_failed_attempt_is_retained_and_fails_required_execution_gate():
    harness = _load_harness()
    protocol = _protocol(harness)
    failed = _attempt(protocol["cases"][0], status="failed")

    report = harness.evaluate_protocol(protocol, [failed, _attempt(protocol["cases"][1])])

    assert report["attempts"] == [failed, _attempt(protocol["cases"][1])]
    assert report["gates"]["execution"]["status"] == "failed"
    assert report["cases"][0]["errors"] == [{"code": "MODEL_FAILURE"}]


def test_handled_out_of_scope_error_is_not_misreported_as_runtime_failure():
    harness = _load_harness()
    protocol = _protocol(harness)
    outside_scope = _attempt(protocol["cases"][0], status="failed")
    outside_scope.update(
        {
            "execution_status": "handled_out_of_scope",
            "processed_ranges": [protocol["cases"][0]["source_range"]],
            "predictions": None,
            "errors": [],
            "routing_outcome": "rejected_unconfirmed_vocal",
        }
    )

    report = harness.evaluate_protocol(
        protocol, [outside_scope, _attempt(protocol["cases"][1])]
    )

    assert report["gates"]["execution"]["status"] == "passed"
    assert report["gates"]["coverage"]["status"] == "passed"
    assert report["cases"][0]["schema_status"] == "not_applicable"


def test_unavailable_metal_leaves_acceptance_pending_not_semantically_failed():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = []
    for case in protocol["cases"]:
        attempt = _attempt(case, status="failed")
        attempt.update(
            {
                "attempted": False,
                "execution_status": "not_attempted",
                "job_status": "not_started",
                "processed_ranges": [],
                "predictions": None,
                "errors": [{"code": "METAL_UNAVAILABLE"}],
                "resource": None,
            }
        )
        attempts.append(attempt)

    report = harness.evaluate_protocol(
        protocol,
        attempts,
        run_metadata={"preflight": "failed", "reason": "METAL_UNAVAILABLE"},
    )

    assert report["gates"]["execution"]["status"] == "pending"
    assert report["gates"]["coverage"]["status"] == "pending"
    assert report["gates"]["semantic"]["status"] == "pending"
    assert report["release_decision"] == "pending"


def test_empty_routing_reference_denominator_is_na_not_zero_or_success():
    harness = _load_harness()
    protocol = _protocol(harness)

    report = harness.evaluate_protocol(
        protocol, [_attempt(case) for case in protocol["cases"]]
    )

    assert report["gates"]["routing"]["status"] == "not_applicable"
    assert report["gates"]["routing"]["denominator"] == 0
    assert report["gates"]["routing"]["rate"] is None


def test_protocol_drift_is_detected_before_evaluation():
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol["cases"][0]["stratum"] = "changed_after_freeze"

    report = harness.evaluate_protocol(protocol, [])

    assert report["gates"]["protocol_integrity"]["status"] == "failed"
    assert report["release_decision"] == "fail"


def test_partial_excerpt_cannot_count_as_full_case_coverage():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["processed_ranges"] = [{"start_seconds": 0.0, "end_seconds": 5.0}]

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["coverage"]["status"] == "failed"
    assert report["cases"][0]["coverage_status"] == "partial"


def test_attempt_must_match_frozen_normalized_audio_hash():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["input_sha256"] = "e" * 64

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["coverage"]["status"] == "failed"
    assert report["gates"]["coverage"]["identity_mismatch_case_ids"] == ["speech-01"]


def test_errors_on_a_completed_attempt_fail_execution_gate():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["errors"] = [{"code": "PARTIAL_TRANSCRIPTION"}]

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["execution"]["status"] == "failed"
    assert any(item.startswith("speech-01:") for item in report["gates"]["execution"]["failures"])


def test_completed_before_cancel_does_not_satisfy_live_cancellation_stage():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["cancellation_probes"] = [
        {"stage": stage, "outcome": "completed_before_cancel"}
        for stage in ("extraction", "asr", "classification", "batch")
    ]

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["resources"]["status"] == "pending"
    assert report["gates"]["resources"]["missing_cancellation_stages"] == [
        "asr", "batch", "classification", "extraction"
    ]


def test_cancellation_over_boundary_limit_fails_resource_gate():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["cancellation_probes"] = [
        {
            "stage": "asr",
            "outcome": "cancelled",
            "process_stopped": True,
            "request_to_stop_seconds": 36.0,
        }
    ]

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["resources"]["status"] == "failed"
    assert report["gates"]["resources"]["failing_stages"] == ["asr"]


def test_below_minimum_memory_fails_when_preflight_claims_success():
    harness = _load_harness()
    protocol = _protocol(harness)

    report = harness.evaluate_protocol(
        protocol,
        [_attempt(case) for case in protocol["cases"]],
        run_metadata={"preflight": "passed", "host_physical_memory_bytes": 8 * 1024**3},
    )

    assert report["gates"]["resources"]["status"] == "failed"
    assert "below" in report["gates"]["resources"]["detail"]


def test_live_run_stops_at_unsupported_host_before_reading_inputs(tmp_path: Path, monkeypatch):
    harness = _load_harness()
    protocol = _protocol(harness)
    monkeypatch.setattr(harness, "_host_preflight", lambda _: (False, {"physical_memory_bytes": 8 * 1024**3}, ["physical memory below minimum"]))
    monkeypatch.setattr(harness, "_verify_inputs", lambda _: (_ for _ in ()).throw(AssertionError("must not read inputs")))

    attempts, metadata = harness.run_live(protocol, tmp_path / "acceptance.json")

    assert len(attempts) == len(protocol["cases"])
    assert all(item["errors"][0]["code"] == "UNSUPPORTED_HOST" for item in attempts)
    assert metadata["reason"] == "UNSUPPORTED_HOST"
    assert metadata["model_downloads"] == 0


def test_live_run_measures_cold_and_same_input_warm_start(tmp_path: Path, monkeypatch):
    harness = _load_harness()
    protocol = _protocol(harness)
    monitored_labels = []
    monkeypatch.setattr(
        harness,
        "_host_preflight",
        lambda _: (True, {"physical_memory_bytes": 64 * 1024**3}, []),
    )
    monkeypatch.setattr(
        harness,
        "_verify_inputs",
        lambda value: (
            {case["case_id"]: tmp_path / case["input_path"] for case in value["cases"]},
            [],
        ),
    )
    monkeypatch.setattr(harness, "_metal_preflight", lambda: (True, None))
    monkeypatch.setattr(harness, "_build_attempt", lambda case, *_: _attempt(case))

    def monitor(_command, *, cwd, label):
        monitored_labels.append(label)
        seconds = 9.0 if label.endswith("-warm") else 12.0
        return {
            "status": "complete",
            "result": {"status": "complete", "measurements": {"wall_seconds": seconds}},
        }, {}

    monkeypatch.setattr(harness, "_monitor_child", monitor)

    attempts, metadata = harness.run_live(protocol, tmp_path / "run.json")

    assert len(attempts) == len(protocol["cases"])
    assert monitored_labels == ["speech-01", "speech-01-warm", "song-01"]
    assert metadata["cold_warm_timing"]["case_id"] == "speech-01"
    assert metadata["cold_warm_timing"]["cold_wall_seconds"] == 12.0
    assert metadata["cold_warm_timing"]["warm_wall_seconds"] == 9.0
    assert metadata["cold_warm_timing"]["warm_probe_status"] == "complete"


def test_batch_interrupt_between_child_runs_does_not_forge_process_stop_evidence(
    tmp_path: Path, monkeypatch
):
    harness = _load_harness()
    protocol = _protocol(harness)
    monkeypatch.setattr(
        harness,
        "_host_preflight",
        lambda _: (True, {"physical_memory_bytes": 64 * 1024**3}, []),
    )
    monkeypatch.setattr(
        harness,
        "_verify_inputs",
        lambda value: (
            {case["case_id"]: tmp_path / case["input_path"] for case in value["cases"]},
            [],
        ),
    )
    monkeypatch.setattr(harness, "_metal_preflight", lambda: (True, None))
    monkeypatch.setattr(harness, "_build_attempt", lambda case, *_: _attempt(case))
    monitored = 0

    def monitor(_command, *, cwd, label):
        nonlocal monitored
        monitored += 1
        if monitored == 3:  # first case, warm probe, then interrupt before the next case
            raise KeyboardInterrupt
        return {
            "status": "complete",
            "result": {"status": "complete", "measurements": {"wall_seconds": 1.0}},
        }, {}

    monkeypatch.setattr(harness, "_monitor_child", monitor)

    attempts, metadata = harness.run_live(protocol, tmp_path / "run.json")

    assert metadata["reason"] == "USER_CANCELLED"
    assert metadata["cancellation_location"] == "outside_child_monitor"
    assert any(attempt["attempted"] for attempt in attempts)
    assert all(
        probe["stage"] != "batch"
        for attempt in attempts
        for probe in attempt.get("cancellation_probes", [])
    )


def test_child_cancel_latency_stops_at_process_exit_not_post_run_cleanup(monkeypatch):
    harness = _load_harness()
    ticks = iter((10.0, 12.25, 100.0, 100.1, 150.0))
    monkeypatch.setattr(harness.time, "monotonic", lambda: next(ticks, 1000.0))

    class FakeQueue:
        def __init__(self):
            self.interrupted = False

        def put(self, _value):
            pass

        def get(self, timeout=None):
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return None

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            pass

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.stdout = FakeStdout()

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = -2

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    class FakeStdout:
        def __iter__(self):
            return iter(())

        def close(self):
            pass

    process = FakeProcess()
    snapshots = 0

    def host_snapshot():
        nonlocal snapshots
        snapshots += 1
        if snapshots == 3:
            # Simulate slow post-exit resource bookkeeping.
            monkeypatch.setattr(harness.time, "monotonic", lambda: 500.0)
        return {"physical_memory_bytes": 64 * 1024**3}

    monkeypatch.setattr(harness, "_host_memory_snapshot", host_snapshot)
    monkeypatch.setattr(harness, "Queue", FakeQueue)
    monkeypatch.setattr(harness.threading, "Thread", FakeThread)
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *args, **kwargs: process)

    result, _host = harness._monitor_child(["ignored"], cwd=Path("."), label="case")

    assert result["_process_stopped_after_cancel"] is True
    assert result["_cancel_request_to_stop_seconds"] == 2.25


def test_cli_records_malformed_saved_attempt_collections_without_crashing(tmp_path: Path):
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol_path = tmp_path / "protocol.json"
    attempts_path = tmp_path / "attempts.json"
    output_path = tmp_path / "report.json"
    protocol_path.write_text(json.dumps(protocol))
    attempts_path.write_text(json.dumps({"attempts": None}))

    code = harness.main(
        [
            "--protocol", str(protocol_path),
            "--attempts", str(attempts_path),
            "--output", str(output_path),
        ]
    )

    report = json.loads(output_path.read_text())
    assert code == 1
    assert report["release_decision"] == "fail"
    assert report["evidence_input_errors"] == [
        "saved attempts report must contain an attempts array"
    ]


def test_cli_records_malformed_saved_review_collections_without_crashing(tmp_path: Path):
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol_path = tmp_path / "protocol.json"
    attempts_path = tmp_path / "attempts.json"
    reviews_path = tmp_path / "reviews.json"
    output_path = tmp_path / "report.json"
    protocol_path.write_text(json.dumps(protocol))
    attempts_path.write_text(json.dumps({"attempts": []}))
    reviews_path.write_text(json.dumps({"reviews": None}))

    code = harness.main(
        [
            "--protocol", str(protocol_path),
            "--attempts", str(attempts_path),
            "--post-result-reviews", str(reviews_path),
            "--output", str(output_path),
        ]
    )

    report = json.loads(output_path.read_text())
    assert code == 1
    assert report["release_decision"] == "fail"
    assert report["evidence_input_errors"] == [
        "post-result review report must contain a reviews array"
    ]


def test_protocol_rejects_resource_requirement_below_64_gib():
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol["resource_requirements"]["physical_memory_bytes_minimum"] = 16 * 1024**3
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)

    errors = harness.validate_protocol(protocol)

    assert any("at least 64 GiB" in error for error in errors)


def test_protocol_rejects_filesystem_unsafe_case_id():
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol["cases"][0]["case_id"] = "../outside"
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)

    errors = harness.validate_protocol(protocol)

    assert any("filesystem-safe" in error for error in errors)


def test_build_attempt_compares_export_using_json_thawed_record_content(tmp_path: Path):
    from engine.domain import JobRecord, JobRequest
    from engine.lifecycle import JobStage
    from engine.store import JobStore

    harness = _load_harness()
    protocol = _protocol(harness)
    case = protocol["cases"][0]
    jobs_dir = tmp_path / "jobs"
    store = JobStore(jobs_dir)
    record = store.create(JobRequest(source="input.wav"))
    for stage in (
        JobStage.EXTRACTING_AUDIO,
        JobStage.DETECTING_LANGUAGE,
        JobStage.TRANSCRIBING,
        JobStage.ANALYSING,
        JobStage.EXPORTING,
    ):
        record = JobRecord.model_validate(
            {
                **record.model_dump(mode="json"),
                "status": stage,
                "progress": record.progress.advance(
                    stage=stage, percent=20, message=stage.value
                ),
            }
        )
        store.save(record)
    store.promote_result(
        record.id,
        {
            "context": 6,
            "kwarray": ["computer peripherals"],
            "language": "en",
        },
    )

    attempt = harness._build_attempt(
        case,
        {
            "result": {
                "job_id": record.id,
                "status": "complete",
                "language": "en",
                "measurements": {
                    "wall_seconds": 1.0,
                    "peak_process_rss_bytes": 1_000_000,
                    "mlx_memory": {"peak_memory_bytes": 2_000_000},
                },
            }
        },
        {},
        jobs_dir,
    )

    assert attempt["execution_status"] == "complete"
    assert attempt["errors"] == []
    assert attempt["export_byte_identical"] is True


def test_failed_job_keeps_partial_window_predictions_without_final_category():
    from engine.domain import JobRecord
    from engine.lifecycle import JobStage
    from engine.standards import load_taxonomy

    harness = _load_harness()
    partial_entry = {
        "time_range": {"start_seconds": 30.0, "end_seconds": 60.0},
        "payload": {"primary_category": "363", "secondary_categories": []},
    }
    record = JobRecord(
        id="12345678-1234-5678-1234-567812345678",
        status=JobStage.FAILED,
        error="ValueError",
        text_timeline=(partial_entry,),
    )

    prediction, timeline = harness._prediction_and_evidence(record, load_taxonomy())

    assert prediction is None
    assert timeline == [partial_entry]


def test_unhashable_case_id_does_not_crash_audit():
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol["cases"][0]["case_id"] = {"bad": []}
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)

    report = harness.evaluate_protocol(protocol, [])

    assert report["gates"]["protocol_integrity"]["status"] == "failed"
    assert report["release_decision"] == "fail"


def test_post_result_review_stays_separate_from_blind_references():
    harness = _load_harness()
    protocol = _protocol(harness)
    reviews = [_review(harness, protocol["cases"][0], _attempt(protocol["cases"][0]))]

    report = harness.evaluate_protocol(
        protocol, [_attempt(case) for case in protocol["cases"]], post_result_reviews=reviews
    )

    assert report["gates"]["semantic"]["status"] == "pending"
    assert report["gates"]["semantic"]["numeric_threshold"] is None
    assert report["semantic_review_separation"] == {
        "blind_reference_count": 0,
        "post_result_review_count": 1,
    }


def test_explicit_qualitative_approval_can_clear_semantic_gate_without_metric():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    for attempt in attempts:
        attempt["export_byte_identical"] = True
        attempt["cancellation_probes"] = [
            {
                "stage": stage,
                "outcome": "cancelled",
                "process_stopped": True,
                "request_to_stop_seconds": 2.0,
            }
            for stage in ("extraction", "asr", "classification", "batch")
        ]
    reviews = [
        _review(harness, case, attempt)
        for case, attempt in zip(protocol["cases"], attempts, strict=True)
    ]

    report = harness.evaluate_protocol(
        protocol,
        attempts,
        post_result_reviews=reviews,
        run_metadata={
            "preflight": "passed",
            "host_physical_memory_bytes": 64 * 1024**3,
            "cold_warm_timing": {
                "case_id": protocol["cases"][0]["case_id"],
                "cold_wall_seconds": 12.0,
                "warm_wall_seconds": 9.0,
                "warm_probe_status": "complete",
            },
        },
    )

    assert report["gates"]["semantic"]["status"] == "passed"
    assert report["gates"]["semantic"]["numeric_threshold"] is None
    assert report["release_decision"] == "pass"


def test_tv_supplements_cannot_clear_full_release_decision():
    harness = _load_harness()
    for protocol_id, pin_name in (
        ("english-phase1-tv-podcast-supplement-v1", "_TV_PODCAST_SUPPLEMENT_SHA256"),
        ("english-phase1-tv-followup-v1", "_TV_FOLLOWUP_SHA256"),
        ("english-phase1-corrective-supplement-v1", "_CORRECTIVE_SUPPLEMENT_SHA256"),
    ):
        protocol = _protocol(harness)
        protocol["protocol_id"] = protocol_id
        protocol["protocol_sha256"] = harness.protocol_digest(protocol)
        setattr(harness, pin_name, protocol["protocol_sha256"])
        attempts = [_attempt(case) for case in protocol["cases"]]
        for attempt in attempts:
            attempt["export_byte_identical"] = True
            attempt["cancellation_probes"] = [
                {"stage": stage, "outcome": "cancelled", "process_stopped": True, "request_to_stop_seconds": 2.0}
                for stage in ("extraction", "asr", "classification", "batch")
            ]
        reviews = [
            _review(harness, case, attempt)
            for case, attempt in zip(protocol["cases"], attempts, strict=True)
        ]

        report = harness.evaluate_protocol(
            protocol,
            attempts,
            post_result_reviews=reviews,
            run_metadata={
                "preflight": "passed",
                "host_physical_memory_bytes": 64 * 1024**3,
                "cold_warm_timing": {
                    "case_id": protocol["cases"][0]["case_id"],
                    "cold_wall_seconds": 12.0,
                    "warm_wall_seconds": 9.0,
                    "warm_probe_status": "complete",
                },
            },
        )

        assert all(gate["status"] in {"passed", "not_applicable"} for gate in report["gates"].values())
        assert report["release_decision"] == "pending"
        assert report["release_scope"] == "supplemental_format_coverage"


def test_revision_requested_qualitative_review_fails_semantic_gate():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    reviews = [
        _review(harness, case, attempt, decision="revision_requested" if index == 0 else "approved")
        for index, (case, attempt) in enumerate(zip(protocol["cases"], attempts, strict=True))
    ]

    report = harness.evaluate_protocol(protocol, attempts, post_result_reviews=reviews)

    assert report["gates"]["semantic"]["status"] == "failed"
    assert report["gates"]["semantic"]["revision_requested_case_ids"] == ["speech-01"]


def test_stale_qualitative_approval_cannot_cover_changed_predictions():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    reviews = [
        _review(harness, case, attempt)
        for case, attempt in zip(protocol["cases"], attempts, strict=True)
    ]
    attempts[0]["predictions"]["primary_category"] = "179"

    report = harness.evaluate_protocol(protocol, attempts, post_result_reviews=reviews)

    assert report["gates"]["semantic"]["status"] == "failed"
    assert any("exact prediction evidence" in error for error in report["gates"]["semantic"]["errors"])


def test_saved_attempts_without_host_memory_stay_pending():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    for attempt in attempts:
        attempt["cancellation_probes"] = [
            {
                "stage": stage,
                "outcome": "cancelled",
                "process_stopped": True,
                "request_to_stop_seconds": 2.0,
            }
            for stage in ("extraction", "asr", "classification", "batch")
        ]
    metadata = {
        "execution_mode": "saved_attempts",
        "cold_warm_timing": {
            "case_id": protocol["cases"][0]["case_id"],
            "cold_wall_seconds": 12.0,
            "warm_wall_seconds": 9.0,
            "warm_probe_status": "complete",
        },
    }

    report = harness.evaluate_protocol(protocol, attempts, run_metadata=metadata)

    assert report["gates"]["resources"]["status"] == "pending"
    assert report["gates"]["resources"]["host_physical_memory_bytes"] is None


def test_saved_attempt_bundle_preserves_run_metadata(tmp_path: Path):
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol_path = tmp_path / "protocol.json"
    attempts_path = tmp_path / "attempts.json"
    report_path = tmp_path / "report.json"
    protocol_path.write_text(json.dumps(protocol))
    attempts_path.write_text(
        json.dumps(
            {
                "attempts": [_attempt(case) for case in protocol["cases"]],
                "run_metadata": {
                    "preflight": "passed",
                    "host_physical_memory_bytes": 64 * 1024**3,
                },
            }
        )
    )

    code = harness.main(
        [
            "--protocol", str(protocol_path),
            "--attempts", str(attempts_path),
            "--output", str(report_path),
        ]
    )

    saved_report = json.loads(report_path.read_text())
    assert code != 0
    assert saved_report["run_metadata"]["host_physical_memory_bytes"] == 64 * 1024**3


def test_negative_resource_measurement_fails_resource_gate():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["resource"]["peak_process_rss_bytes"] = -1

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["resources"]["status"] == "failed"
    assert report["gates"]["resources"]["invalid_case_ids"] == ["speech-01"]


def test_post_result_approval_cannot_be_declared_a_blind_label():
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol["cases"][0]["semantic_reference"] = {
        "primary_category": "150",
        "provenance": "post_result_user_review",
    }
    protocol["protocol_sha256"] = harness.protocol_digest(protocol)

    report = harness.evaluate_protocol(protocol, [])

    assert report["gates"]["protocol_integrity"]["status"] == "failed"
    assert any("post-result" in error for error in report["protocol_errors"])


def test_invalid_schema_fails_without_suppressing_other_cases():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["predictions"]["secondary_categories"] = ["151", "152", "153"]

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["schema"]["status"] == "failed"
    assert len(report["cases"]) == 2
    assert report["cases"][1]["schema_status"] == "passed"


def test_unknown_iab_category_id_fails_schema_gate():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["predictions"]["primary_category"] = "unknown-category"

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["schema"]["status"] == "failed"
    assert report["cases"][0]["schema_status"] == "failed"


def test_byte_mismatched_export_fails_its_own_gate():
    harness = _load_harness()
    protocol = _protocol(harness)
    attempts = [_attempt(case) for case in protocol["cases"]]
    attempts[0]["export_byte_identical"] = False
    attempts[1]["export_byte_identical"] = True

    report = harness.evaluate_protocol(protocol, attempts)

    assert report["gates"]["export_integrity"]["status"] == "failed"
    assert report["gates"]["export_integrity"]["mismatched_case_ids"] == ["speech-01"]
    assert report["release_decision"] == "fail"


def test_cli_returns_nonzero_when_required_gates_fail(tmp_path: Path):
    harness = _load_harness()
    protocol = _protocol(harness)
    protocol_path = tmp_path / "protocol.json"
    output_path = tmp_path / "report.json"
    protocol_path.write_text(json.dumps(protocol))

    code = harness.main(
        ["--protocol", str(protocol_path), "--output", str(output_path), "--no-run"]
    )

    assert code != 0
    assert json.loads(output_path.read_text())["release_decision"] == "fail"
