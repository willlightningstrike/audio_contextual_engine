"""Experimental transcript-only classification with bounded topic verification."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from engine.cancel import CancellationToken
from engine.signals.classify import (
    CandidateRecord, analysis_windows_from_run, classify_locked_payload, classify_window,
)
from engine.signals.protocol import ConstrainedTextGenerator
from engine.signals.taxonomy import TaxonomyIndex
from engine.signals.text_types import ClassificationResult, GenerationAttempt
from engine.signals.topic_verify import VERIFIER_POLICY, VerifierAttempt
from engine.signals.verified_topics import VerifierProgress, verify_window_topics
from engine.transcribe.types import TranscriptionRun


def classify_and_verify_transcription_run(
    run: TranscriptionRun,
    *,
    taxonomy_index: TaxonomyIndex,
    generator: ConstrainedTextGenerator,
    token: CancellationToken,
    candidate_checkpoint: Callable[[CandidateRecord], Any],
    classification_attempt_checkpoint: Callable[[GenerationAttempt], Any],
    verifier_attempt_checkpoint: Callable[[VerifierAttempt], Any],
    result_checkpoint: Callable[[ClassificationResult], Any],
    on_verifier_progress: VerifierProgress,
) -> tuple[ClassificationResult, ...]:
    """Promote only verifier-supported categories; retain all completed raw attempts."""
    results: list[ClassificationResult] = []
    for window in analysis_windows_from_run(run):
        token.raise_if_cancelled()
        candidates = tuple(taxonomy_index.shortlist(
            " ".join(segment.text for segment in window.segments), top_k=15, token=token,
        ))
        token.raise_if_cancelled()
        candidate_checkpoint(CandidateRecord(time_range=window.time_range, candidates=candidates))
        token.raise_if_cancelled()
        proposed = classify_window(
            window, candidates, generator=generator, token=token,
            checkpoint=classification_attempt_checkpoint,
        )
        result = proposed
        if proposed.error_code in {None, "INSUFFICIENT_TOPIC_EVIDENCE"}:
            verified = verify_window_topics(
                window, proposed, candidates, generator=generator, token=token,
                checkpoint=verifier_attempt_checkpoint, on_verifier_progress=on_verifier_progress,
            )
            token.raise_if_cancelled()
            if verified.error_code is not None:
                result = proposed.model_copy(update={
                    "payload": None, "category_citations": (), "error_code": verified.error_code,
                })
            elif verified.primary_category is None:
                result = proposed.model_copy(update={
                    "payload": None, "category_citations": (),
                    "error_code": "INSUFFICIENT_TOPIC_EVIDENCE",
                    "verification_policy_version": VERIFIER_POLICY,
                })
            else:
                if proposed.payload is None:
                    result = classify_locked_payload(
                        window, candidates, verified.primary_category,
                        generator=generator, token=token, prior_attempts=proposed.attempts,
                        checkpoint=classification_attempt_checkpoint,
                    )
                if result.payload is not None:
                    payload = result.payload.model_copy(update={
                        "primary_category": verified.primary_category,
                        "secondary_categories": verified.secondary_categories,
                    })
                    result = result.model_copy(update={
                        "payload": payload, "error_code": None,
                        "category_citations": verified.citations,
                        "verification_policy_version": VERIFIER_POLICY,
                    })
        token.raise_if_cancelled()
        result_checkpoint(result)
        token.raise_if_cancelled()
        results.append(result)
    return tuple(results)
