"""Bounded, fail-closed verification of transcript category proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.signals.protocol import ConstrainedTextGenerator
from engine.signals.taxonomy import TaxonomyCandidate
from engine.signals.text_types import CategoryCitation, ClassificationResult
from engine.signals.topic_verify import (
    VERIFIER_POLICY, VerifierAttempt, VerifierJudgment, build_verifier_prompt,
    parse_verifier_judgment, verifier_max_tokens, verifier_schema,
)
from engine.transcribe.types import AnalysisWindow

MAX_ALTERNATIVE_PRIMARIES = 2
MAX_VERIFIER_CALLS = 5
VerifierProgress = Callable[[TimeRange, int, int, Literal["starting", "completed"]], None]


@dataclass(frozen=True)
class VerifiedTopics:
    policy_version: str = VERIFIER_POLICY
    primary_category: str | None = None
    secondary_categories: tuple[str, ...] = ()
    citations: tuple[CategoryCitation, ...] = ()
    attempts: tuple[VerifierAttempt, ...] = ()
    error_code: str | None = None


def transcript_candidate(candidate: TaxonomyCandidate) -> bool:
    return candidate.taxonomy_id not in {"338", "650"} and candidate.path[:1] != ("Music",)


def _overlap(first: TaxonomyCandidate, second: TaxonomyCandidate) -> bool:
    return (
        first.taxonomy_id == second.taxonomy_id
        or first.path == second.path[:len(first.path)]
        or second.path == first.path[:len(second.path)]
    )


class _TechnicalFailure(Exception):
    pass


def verify_window_topics(
    window: AnalysisWindow, proposed: ClassificationResult,
    ranked: Sequence[TaxonomyCandidate], *, generator: ConstrainedTextGenerator,
    token: CancellationToken, checkpoint: Callable[[VerifierAttempt], None],
    on_verifier_progress: VerifierProgress | None = None,
) -> VerifiedTopics:
    token.raise_if_cancelled()
    attempts: list[VerifierAttempt] = []
    by_id = {item.taxonomy_id: item for item in ranked}
    if len(by_id) != len(ranked) or (
        proposed.payload is None and proposed.error_code != "INSUFFICIENT_TOPIC_EVIDENCE"
    ):
        return VerifiedTopics(error_code="INVALID_VERIFIER_INPUT")
    payload = proposed.payload
    proposed_id = payload.primary_category if payload else None
    # Recognizable Music IDs are excluded even when not present in retrieval.
    if proposed_id and proposed_id not in by_id and proposed_id not in {"338", "650"}:
        return VerifiedTopics(error_code="INVALID_VERIFIER_INPUT")
    eligible = [item for item in ranked if transcript_candidate(item)]
    eligible_by_id = {item.taxonomy_id: item for item in eligible}
    secondary_ids = payload.secondary_categories if payload else ()
    if any(id not in by_id and id not in {"338", "650"} for id in secondary_ids):
        return VerifiedTopics(error_code="INVALID_VERIFIER_INPUT")
    order = [eligible_by_id[proposed_id]] if proposed_id in eligible_by_id else []
    order += [item for item in eligible if item.taxonomy_id != proposed_id][:1 + MAX_ALTERNATIVE_PRIMARIES - len(order)]

    def issue(candidate: TaxonomyCandidate) -> VerifierJudgment:
        token.raise_if_cancelled()
        if len(attempts) >= MAX_VERIFIER_CALLS:
            raise _TechnicalFailure("VERIFIER_CALL_BUDGET_EXCEEDED")
        prompt = build_verifier_prompt(window, candidate, [item for item in eligible if item.taxonomy_id != candidate.taxonomy_id])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if on_verifier_progress is not None:
            on_verifier_progress(window.time_range, len(attempts), MAX_VERIFIER_CALLS, "starting")
        token.raise_if_cancelled()
        try:
            generation = generator.generate_json(
                prompt, verifier_schema(candidate.taxonomy_id, len(window.segments)),
                max_tokens=verifier_max_tokens(len(window.segments)), token=token,
            )
        except JobCancelled:
            raise
        except Exception:
            attempt = VerifierAttempt(category_id=candidate.taxonomy_id, prompt_sha256=digest, error_code="GENERATION_ERROR")
            attempts.append(attempt)
            checkpoint(attempt)
            if on_verifier_progress is not None:
                on_verifier_progress(window.time_range, len(attempts), MAX_VERIFIER_CALLS, "completed")
            token.raise_if_cancelled()
            raise _TechnicalFailure("GENERATION_ERROR")
        attempt = VerifierAttempt(category_id=candidate.taxonomy_id, prompt_sha256=digest, raw=generation.raw, finish_reason=generation.finish_reason, seconds=generation.seconds, token_count=generation.token_count)
        # Persist a completed generation even if cancellation arrived during it.
        attempts.append(attempt)
        checkpoint(attempt)
        if on_verifier_progress is not None:
            on_verifier_progress(window.time_range, len(attempts), MAX_VERIFIER_CALLS, "completed")
        token.raise_if_cancelled()
        error = None
        judgment = None
        if generation.finish_reason != "stop":
            error = "INCOMPLETE_GENERATION"
        else:
            try:
                judgment = parse_verifier_judgment(generation.raw, candidate.taxonomy_id, window)
            except (ValueError, TypeError, UnicodeError):
                error = "INVALID_VERIFIER_OUTPUT"
        if error:
            attempts[-1] = attempt.model_copy(update={"error_code": error})
            raise _TechnicalFailure(error)
        assert judgment is not None
        return judgment

    accepted: list[TaxonomyCandidate] = []
    citations: list[CategoryCitation] = []
    try:
        for candidate in order:
            judgment = issue(candidate)
            if judgment.status == "supported":
                accepted.append(candidate)
                citations.append(CategoryCitation(category_id=judgment.category_id, segment_indices=judgment.segment_indices))
                break
        if accepted:
            for id in secondary_ids[:2]:
                candidate = eligible_by_id.get(id)
                if candidate is None or any(_overlap(candidate, item) for item in accepted):
                    continue
                judgment = issue(candidate)
                if judgment.status == "supported":
                    accepted.append(candidate)
                    citations.append(CategoryCitation(category_id=judgment.category_id, segment_indices=judgment.segment_indices))
    except _TechnicalFailure as exc:
        return VerifiedTopics(attempts=tuple(attempts), error_code=str(exc))
    token.raise_if_cancelled()
    return VerifiedTopics(primary_category=accepted[0].taxonomy_id if accepted else None, secondary_categories=tuple(item.taxonomy_id for item in accepted[1:]), citations=tuple(citations), attempts=tuple(attempts))
