"""Cited transcript-duration gate for the English whole-file topic policy."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from engine.domain import TimeRange
from engine.signals.classify import analysis_windows_from_run
from engine.signals.result_types import _duration
from engine.signals.text_types import ClassificationResult
from engine.standards import TaxonomyRow
from engine.transcribe.types import TranscriptionRun


TopicStructure = Literal["single", "parent_rollup", "multi_topic"]

# english-audio-aggregation-v3 topic rules (shares of usable transcript time T).
PRIMARY_TRANSCRIPT_SHARE = 0.50  # strictly greater than
PRIMARY_AUDIO_SHARE = 0.25
MULTI_TOPIC_MIN_SHARE = 0.20


class TopicDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, allow_inf_nan=False)

    policy_version: Literal[
        "english-audio-aggregation-v2", "english-audio-aggregation-v3"
    ] = "english-audio-aggregation-v2"
    outcome: Literal["eligible", "insufficient"]
    primary_category: str | None
    transcript_seconds: float = Field(ge=0)
    audio_seconds: float = Field(gt=0)
    primary_role_support_seconds: tuple[tuple[str, float], ...]
    secondary_role_support_seconds: tuple[tuple[str, float], ...]
    # v3 only: how the categories were chosen and the support each was given.
    topic_structure: TopicStructure | None = None
    secondary_categories: tuple[str, ...] = Field(default=(), max_length=2)
    selected_support_seconds: tuple[tuple[str, float], ...] = ()

    @model_serializer(mode="wrap")
    def _omit_v3_fields_from_v2(self, handler):
        # Saved v2 decisions keep their original serialized form.
        data = handler(self)
        if self.policy_version == "english-audio-aggregation-v2":
            for key in ("topic_structure", "secondary_categories", "selected_support_seconds"):
                data.pop(key, None)
        return data

    @model_validator(mode="after")
    def _valid_decision(self) -> TopicDecision:
        if (self.outcome == "eligible") != (self.primary_category is not None):
            raise ValueError("eligible decision requires a primary category")
        for support in (
            self.primary_role_support_seconds, self.secondary_role_support_seconds,
            self.selected_support_seconds,
        ):
            ids = tuple(category_id for category_id, _ in support)
            if len(ids) != len(set(ids)) or any(not category_id.strip() for category_id in ids):
                raise ValueError("support categories must be unique non-blank IDs")
            if any(seconds <= 0 for _, seconds in support):
                raise ValueError("support durations must be positive")
        if self.policy_version == "english-audio-aggregation-v2":
            if self.topic_structure or self.secondary_categories or self.selected_support_seconds:
                raise ValueError("v2 decisions do not record v3 topic fields")
            if self.primary_category is not None and self.primary_category not in dict(self.primary_role_support_seconds):
                raise ValueError("primary category must have cited primary-role support")
            return self
        if (self.outcome == "eligible") != (self.topic_structure is not None):
            raise ValueError("eligible v3 decision requires a topic structure")
        selected = (
            (self.primary_category, *self.secondary_categories)
            if self.primary_category is not None else ()
        )
        if tuple(category_id for category_id, _ in self.selected_support_seconds) != selected:
            raise ValueError("selected support must list the chosen categories in order")
        if self.topic_structure in {"single", "multi_topic"} and self.primary_category not in dict(self.primary_role_support_seconds):
            raise ValueError("primary category must have cited primary-role support")
        return self


def _inside(inner: TimeRange, outer: TimeRange) -> bool:
    return (
        inner.start_seconds >= outer.start_seconds
        and inner.end_seconds <= outer.end_seconds
        and inner.end_seconds > inner.start_seconds
    )


def _overlap(first: TimeRange, second: TimeRange) -> bool:
    return first.start_seconds < second.end_seconds and second.start_seconds < first.end_seconds


def evaluate_topic_sufficiency(
    run: TranscriptionRun,
    classifications: tuple[ClassificationResult, ...],
    *,
    audio_seconds: float,
) -> TopicDecision:
    """Decide from role-specific cited speech without assigning a confidence score."""
    from engine.completion import completion_errors

    errors = completion_errors(run, classifications)
    if errors:
        raise ValueError("topic sufficiency requires complete usable audio: " + ", ".join(errors))
    if (
        isinstance(audio_seconds, bool)
        or not isinstance(audio_seconds, (int, float))
        or not math.isfinite(audio_seconds)
        or audio_seconds <= 0
    ):
        raise ValueError("audio_seconds must be positive finite seconds")
    windows = analysis_windows_from_run(run)
    if len(windows) != len(classifications) or any(
        window.time_range != result.time_range for window, result in zip(windows, classifications)
    ):
        raise ValueError("analysis result/window mismatch")

    transcript_ranges: list[TimeRange] = []
    for source_window in run.windows:
        for segment in source_window.segments:
            if segment.language != "en":
                raise ValueError("included transcript segment must be usable English")
            if not _inside(segment.time_range, source_window.time_range):
                raise ValueError("transcript segment must lie inside its source window")
            if segment.time_range.end_seconds > audio_seconds:
                raise ValueError("transcript segment exceeds audio_seconds")
            transcript_ranges.append(segment.time_range)
    total = _duration(tuple(transcript_ranges))

    primary_ranges: dict[str, list[TimeRange]] = defaultdict(list)
    secondary_ranges: dict[str, list[TimeRange]] = defaultdict(list)
    primary_window_ranges: list[tuple[int, TimeRange]] = []
    for window_number, (window, result) in enumerate(zip(windows, classifications)):
        if result.policy_version != "english-audio-topic-sufficiency-v1":
            raise ValueError("missing versioned category result")
        if result.error_code == "INSUFFICIENT_TOPIC_EVIDENCE":
            if result.payload is not None or result.category_citations:
                raise ValueError("abstention cannot contain category citations")
            continue
        if result.error_code is not None or result.payload is None:
            raise ValueError("missing versioned category result")
        selected = {result.payload.primary_category, *result.payload.secondary_categories}
        cited_ids = tuple(citation.category_id for citation in result.category_citations)
        if len(cited_ids) != len(set(cited_ids)) or set(cited_ids) != selected:
            raise ValueError("category citations must exactly match payload categories")
        for citation in result.category_citations:
            target = (
                primary_ranges
                if citation.category_id == result.payload.primary_category
                else secondary_ranges
            )
            for index in citation.segment_indices:
                if index >= len(window.segments):
                    raise ValueError("citation index outside analysis window")
                segment_range = window.segments[index].time_range
                if not _inside(segment_range, window.time_range):
                    raise ValueError("cited segment outside analysis window")
                target[citation.category_id].append(segment_range)
                if target is primary_ranges:
                    primary_window_ranges.append((window_number, segment_range))

    for index, (window_number, first) in enumerate(primary_window_ranges):
        if any(
            other_window != window_number and _overlap(first, second)
            for other_window, second in primary_window_ranges[index + 1 :]
        ):
            raise ValueError("primary-role windows overlap")

    primary_support = {
        category_id: _duration(tuple(ranges)) for category_id, ranges in primary_ranges.items()
    }
    secondary_support = {
        category_id: _duration(tuple(ranges)) for category_id, ranges in secondary_ranges.items()
    }
    eligible = [
        category_id for category_id, seconds in primary_support.items()
        if total > 0 and seconds / total > 0.50 and seconds / audio_seconds >= 0.25
    ]
    if len(eligible) > 1:
        raise ValueError("multiple eligible primary categories")
    primary = eligible[0] if eligible else None
    return TopicDecision(
        outcome="eligible" if primary is not None else "insufficient",
        primary_category=primary,
        transcript_seconds=total,
        audio_seconds=float(audio_seconds),
        primary_role_support_seconds=tuple(sorted(primary_support.items())),
        secondary_role_support_seconds=tuple(sorted(secondary_support.items())),
    )


def _role_ranges(
    run: TranscriptionRun,
    classifications: tuple[ClassificationResult, ...],
    *,
    audio_seconds: float,
    skip_technical_failures: bool,
) -> tuple[list[TimeRange], dict[str, list[TimeRange]], dict[str, list[TimeRange]]]:
    """Collect cited role ranges, validating the same invariants as v2."""
    if (
        isinstance(audio_seconds, bool)
        or not isinstance(audio_seconds, (int, float))
        or not math.isfinite(audio_seconds)
        or audio_seconds <= 0
    ):
        raise ValueError("audio_seconds must be positive finite seconds")
    windows = analysis_windows_from_run(run)
    if len(windows) != len(classifications) or any(
        window.time_range != result.time_range for window, result in zip(windows, classifications)
    ):
        raise ValueError("analysis result/window mismatch")
    transcript_ranges: list[TimeRange] = []
    primary_ranges: dict[str, list[TimeRange]] = defaultdict(list)
    secondary_ranges: dict[str, list[TimeRange]] = defaultdict(list)
    for window, result in zip(windows, classifications):
        technical = result.payload is None and result.error_code != "INSUFFICIENT_TOPIC_EVIDENCE"
        if technical and skip_technical_failures:
            continue
        if result.policy_version != "english-audio-topic-sufficiency-v1":
            raise ValueError("missing versioned category result")
        for segment in window.segments:
            if segment.language != "en":
                raise ValueError("included transcript segment must be usable English")
            if not _inside(segment.time_range, window.time_range):
                raise ValueError("transcript segment must lie inside its analysis window")
            if segment.time_range.end_seconds > audio_seconds:
                raise ValueError("transcript segment exceeds audio_seconds")
            transcript_ranges.append(segment.time_range)
        if result.error_code == "INSUFFICIENT_TOPIC_EVIDENCE":
            if result.payload is not None or result.category_citations:
                raise ValueError("abstention cannot contain category citations")
            continue
        if result.error_code is not None or result.payload is None:
            raise ValueError("missing versioned category result")
        selected = {result.payload.primary_category, *result.payload.secondary_categories}
        cited_ids = tuple(citation.category_id for citation in result.category_citations)
        if len(cited_ids) != len(set(cited_ids)) or set(cited_ids) != selected:
            raise ValueError("category citations must exactly match payload categories")
        for citation in result.category_citations:
            target = (
                primary_ranges
                if citation.category_id == result.payload.primary_category
                else secondary_ranges
            )
            for index in citation.segment_indices:
                if index >= len(window.segments):
                    raise ValueError("citation index outside analysis window")
                target[citation.category_id].append(window.segments[index].time_range)
    return transcript_ranges, primary_ranges, secondary_ranges


def evaluate_topic_policy_v3(
    run: TranscriptionRun,
    classifications: tuple[ClassificationResult, ...],
    *,
    audio_seconds: float,
    taxonomy: Sequence[TaxonomyRow],
) -> TopicDecision:
    """Decide over usable windows only; coverage caps are applied beforehand.

    In order: one dominant category; else the most specific shared parent whose
    descendants together dominate; else two or three co-equal major topics.
    """
    transcript_ranges, primary_ranges, secondary_ranges = _role_ranges(
        run, classifications, audio_seconds=audio_seconds, skip_technical_failures=True,
    )
    total = _duration(tuple(transcript_ranges))
    by_id = {row.id: row for row in taxonomy}
    if any(category_id not in by_id for category_id in (*primary_ranges, *secondary_ranges)):
        raise ValueError("unknown taxonomy category in cited support")

    def ancestors(category_id: str) -> tuple[str, ...]:
        chain: list[str] = []
        parent = by_id[category_id].parent_id
        while parent is not None:
            if parent in chain:
                raise ValueError("taxonomy contains a parent cycle")
            chain.append(parent)
            parent = by_id[parent].parent_id
        return tuple(chain)

    def redundant(first: str, second: str) -> bool:
        return first == second or first in ancestors(second) or second in ancestors(first)

    def dominant(seconds: float) -> bool:
        return (
            total > 0
            and seconds / total > PRIMARY_TRANSCRIPT_SHARE
            and seconds / audio_seconds >= PRIMARY_AUDIO_SHARE
        )

    primary_support = {cid: _duration(tuple(r)) for cid, r in primary_ranges.items()}
    secondary_support = {cid: _duration(tuple(r)) for cid, r in secondary_ranges.items()}

    def by_support(support: dict[str, float]) -> list[str]:
        return sorted(support, key=lambda cid: (-support[cid], cid))

    structure: str | None = None
    selected: list[tuple[str, float]] = []
    singles = [cid for cid in by_support(primary_support) if dominant(primary_support[cid])]
    if singles:
        structure = "single"
        selected = [(singles[0], primary_support[singles[0]])]
    else:
        rollup: dict[str, list[TimeRange]] = defaultdict(list)
        for category_id, ranges in primary_ranges.items():
            for ancestor in ancestors(category_id):
                rollup[ancestor].extend(ranges)
        parent_support = {
            cid: _duration(tuple(ranges) + tuple(primary_ranges.get(cid, ())))
            for cid, ranges in rollup.items()
        }
        parents = sorted(
            (cid for cid, seconds in parent_support.items() if dominant(seconds)),
            # Prefer the most specific parent, then the larger support.
            key=lambda cid: (-len(ancestors(cid)), -parent_support[cid], cid),
        )
        if parents:
            structure = "parent_rollup"
            selected = [(parents[0], parent_support[parents[0]])]
        else:
            majors: list[str] = []
            for cid in by_support(primary_support):
                if total > 0 and primary_support[cid] / total >= MULTI_TOPIC_MIN_SHARE and not any(
                    redundant(cid, chosen) for chosen in majors
                ):
                    majors.append(cid)
            majors = majors[:3]
            combined = _duration(tuple(r for cid in majors for r in primary_ranges[cid]))
            if len(majors) >= 2 and dominant(combined):
                structure = "multi_topic"
                selected = [(cid, primary_support[cid]) for cid in majors]

    if structure in {"single", "parent_rollup"}:
        # Secondaries rank on all verified support, so a category that won the
        # primary role in some windows can still be reported as secondary.
        combined_support = {
            cid: _duration(tuple(primary_ranges.get(cid, ())) + tuple(secondary_ranges.get(cid, ())))
            for cid in {*primary_ranges, *secondary_ranges}
        }
        for cid in by_support(combined_support):
            if len(selected) == 3:
                break
            if not any(redundant(cid, chosen) for chosen, _ in selected):
                selected.append((cid, combined_support[cid]))

    return TopicDecision(
        policy_version="english-audio-aggregation-v3",
        outcome="eligible" if structure else "insufficient",
        primary_category=selected[0][0] if structure else None,
        transcript_seconds=total,
        audio_seconds=float(audio_seconds),
        primary_role_support_seconds=tuple(sorted(primary_support.items())),
        secondary_role_support_seconds=tuple(sorted(secondary_support.items())),
        topic_structure=structure,
        secondary_categories=tuple(cid for cid, _ in selected[1:]) if structure else (),
        selected_support_seconds=tuple(selected) if structure else (),
    )
