"""UI-safe projections of durable job records and partial diagnostics."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from engine.domain import JobRecord, TimeRange
from engine.lifecycle import JobStage
from engine.signals.aggregate import whole_file_sentiment
from engine.signals.result_types import AudioTranscriptProvenance, TextTimelineEntry
from engine.signals.text_types import CategoryCitation, ClassificationPayload
from engine.signals.topic_sufficiency import TopicDecision
from engine.signals.topic_verify import KNOWN_VERIFIER_POLICIES


_TOPIC_RESULT_VERSION = "english-audio-topic-sufficiency-v1"


def _union_seconds(ranges: tuple[TimeRange, ...]) -> float:
    intervals = sorted((item.start_seconds, item.end_seconds) for item in ranges)
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _union_ranges(ranges: tuple[TimeRange, ...]) -> tuple[TimeRange, ...]:
    ordered = sorted(ranges, key=lambda item: (item.start_seconds, item.end_seconds))
    if not ordered:
        return ()
    merged = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if item.start_seconds <= previous.end_seconds:
            merged[-1] = TimeRange(
                start_seconds=previous.start_seconds,
                end_seconds=max(previous.end_seconds, item.end_seconds),
            )
        else:
            merged.append(item)
    return tuple(merged)


def _sentiment(record: JobRecord) -> dict[str, Any] | None:
    """Whole-file sentiment for the UI only; it is not part of the export."""
    if record.status is not JobStage.COMPLETE:
        return None
    try:
        entries = tuple(
            TextTimelineEntry.model_validate_json(json.dumps(entry, allow_nan=False))
            for entry in record.model_dump(mode="json")["text_timeline"]
        )
    except ValidationError:
        return None
    value = whole_file_sentiment(entries)
    return value.model_dump(mode="json") if value else None


def job_snapshot(store: object, record: JobRecord) -> dict[str, Any]:
    """Add persisted window and range outcomes without asking the UI to infer them."""
    windows = store.load_transcription_windows(record.id)  # type: ignore[attr-defined]
    diagnostics = store.load_analysis_diagnostics(record.id)  # type: ignore[attr-defined]
    verification_windows = []
    for entry in diagnostics:
        diagnostic = entry["payload"]
        if entry["kind"] != "result" or diagnostic.get("verification_policy_version") not in KNOWN_VERIFIER_POLICIES:
            continue
        error = diagnostic.get("error_code")
        time_range = diagnostic["time_range"]
        segments = tuple(segment for segment in record.transcript
                         if time_range["start_seconds"] <= segment.time_range.start_seconds
                         and segment.time_range.end_seconds <= time_range["end_seconds"])
        citations = []
        for citation in diagnostic.get("category_citations", ()):
            indices = citation.get("segment_indices", ())
            if indices and all(type(index) is int and 0 <= index < len(segments) for index in indices):
                citations.append({"category_id": citation["category_id"],
                                  "segment_indices": list(indices),
                                  "time_ranges": [segments[index].time_range.model_dump(mode="json")
                                                  for index in indices]})
        verification_windows.append({
            "time_range": time_range,
            "verification_policy_version": diagnostic["verification_policy_version"],
            "status": ("abstained" if error == "INSUFFICIENT_TOPIC_EVIDENCE"
                       else "technical_failure" if error else "supported"),
            "error_code": error, "citations": citations,
        })
    failed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for window in windows:
        if window.error_code is None:
            continue
        value = {
            "time_range": window.time_range.model_dump(mode="json"),
            "error_code": window.error_code,
            "stage": window.failure_stage,
            "reason": window.failure_reason,
        }
        # Out-of-scope audio is excluded; only technical failures are failures.
        if window.failure_reason == "vocal_presence_uncertain" or window.error_code == "NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE":
            excluded.append(value)
        else:
            failed.append(value)
    for entry in diagnostics:
        payload = entry["payload"]
        if entry["kind"] != "result" or not payload.get("error_code"):
            continue
        if (payload.get("policy_version") == _TOPIC_RESULT_VERSION
                and payload["error_code"] == "INSUFFICIENT_TOPIC_EVIDENCE"):
            continue
        failed.append(
            {
                "time_range": payload.get("time_range"),
                "error_code": payload["error_code"],
                "stage": ("verification" if payload.get("verification_policy_version") in KNOWN_VERIFIER_POLICIES
                          else "classification"),
                "reason": payload.get("error_code"),
            }
        )
    abstained = [
        {"time_range": entry["payload"]["time_range"],
         "error_code": "INSUFFICIENT_TOPIC_EVIDENCE"}
        for entry in diagnostics
        if entry["kind"] == "result"
        and entry["payload"].get("policy_version") == _TOPIC_RESULT_VERSION
        and entry["payload"].get("error_code") == "INSUFFICIENT_TOPIC_EVIDENCE"
        and isinstance(entry["payload"].get("time_range"), dict)
    ]
    policies = [
        TopicDecision.model_validate_json(json.dumps(entry["payload"], allow_nan=False))
        for entry in diagnostics if entry["kind"] == "topic_policy"
    ]
    projected_timeline = list(record.model_dump(mode="json")["text_timeline"])
    existing_ranges = {
        (item["time_range"]["start_seconds"], item["time_range"]["end_seconds"])
        for item in projected_timeline
    }
    for entry in diagnostics:
        diagnostic = entry["payload"]
        if entry["kind"] != "result" or diagnostic.get("error_code") or not diagnostic.get("payload"):
            continue
        if record.evidence_mode == "transcript_only" and diagnostic.get("verification_policy_version") not in KNOWN_VERIFIER_POLICIES:
            continue
        try:
            time_range = TimeRange.model_validate(diagnostic["time_range"])
            key = (time_range.start_seconds, time_range.end_seconds)
            if key in existing_ranges:
                continue
            window_segments = tuple(
                segment for segment in record.transcript
                if time_range.start_seconds <= segment.time_range.start_seconds
                and segment.time_range.end_seconds <= time_range.end_seconds
            )
            policy_version = diagnostic.get("policy_version")
            if policy_version == _TOPIC_RESULT_VERSION:
                payload = ClassificationPayload.model_validate_json(
                    json.dumps(diagnostic["payload"], allow_nan=False)
                )
                citations = tuple(
                    CategoryCitation.model_validate_json(json.dumps(item, allow_nan=False))
                    for item in diagnostic["category_citations"]
                )
                selected = (payload.primary_category, *payload.secondary_categories)
                if tuple(item.category_id for item in citations) != selected:
                    continue
                cited_indices = {
                    index for citation in citations for index in citation.segment_indices
                }
                if not cited_indices or max(cited_indices) >= len(window_segments):
                    continue
                evidence_ranges = _union_ranges(tuple(
                    segment.time_range for index, segment in enumerate(window_segments)
                    if index in cited_indices
                ))
            elif policy_version is None:
                evidence_ranges = tuple(
                    TimeRange(
                        start_seconds=max(time_range.start_seconds, segment.time_range.start_seconds),
                        end_seconds=min(time_range.end_seconds, segment.time_range.end_seconds),
                    )
                    for segment in record.transcript
                    if min(time_range.end_seconds, segment.time_range.end_seconds)
                    > max(time_range.start_seconds, segment.time_range.start_seconds)
                )
                payload = ClassificationPayload.model_validate_json(
                    json.dumps(diagnostic["payload"], allow_nan=False)
                )
            else:
                continue
            if not evidence_ranges:
                continue
            partial = TextTimelineEntry(
                time_range=time_range,
                payload=payload,
                evidence_ranges=evidence_ranges,
                provenance=AudioTranscriptProvenance(
                    audio_stream_index=(
                        record.request.audio_stream_index if record.request else None
                    )
                ),
            )
        except (KeyError, TypeError, ValueError):
            continue
        projected_timeline.append(partial.model_dump(mode="json"))
        existing_ranges.add(key)
    transcript_ranges = tuple(segment.time_range for segment in record.transcript)
    serialized_windows = [window.model_dump(mode="json") for window in windows]
    result = record.model_dump(mode="json")
    result["transcription_windows"] = serialized_windows
    result["text_timeline"] = projected_timeline
    result["topic_policy"] = policies[-1].model_dump(mode="json") if policies else None
    result["topic_abstentions"] = abstained
    result["sentiment"] = _sentiment(record)
    result["verification_windows"] = verification_windows
    result["coverage"] = {
        "total_windows": len(windows),
        "successful_windows": sum(window.error_code is None for window in windows),
        "transcribed_seconds": _union_seconds(transcript_ranges),
        "classified_units": len(projected_timeline),
        "failed_ranges": failed,
        "excluded_ranges": excluded,
        "abstained_ranges": abstained,
    }
    return result
