"""Model-free integration checks for promotion of verified transcript topics."""
import json

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange, TranscriptSegment
from engine.signals.taxonomy import TaxonomyCandidate
from engine.signals.text_types import CategoryCitation, GenerationResult
from engine.transcribe.types import TranscriptionRun, WindowResult


CANDIDATES = tuple(TaxonomyCandidate(id, id, (id,), 0.5) for id in ("633", "514", "255", "1", "2"))


def primary(id="514"):
    return json.dumps({"primary_category": id})


def payload(id="514", secondary=()):
    return json.dumps(dict(primary_category=id, secondary_categories=list(secondary),
        keywords=[dict(term="camera", confidence=0.8)],
        entities=[dict(text="Canon", type="brand", mention_count=1)],
        sentiment=dict(label="positive", score=0.7),
        category_evidence=[dict(category_id=value, segment_indices=[0]) for value in (id, *secondary)]))


def verdict(id, status="supported", indices=(1,)):
    return json.dumps(dict(category_id=id, status=status, segment_indices=list(indices) if status == "supported" else []))


class Generator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate_json(self, prompt, schema, *, max_tokens, token):
        self.prompts.append(prompt)
        value = next(self.responses)
        if callable(value):
            value = value(token)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, GenerationResult) else GenerationResult(raw=value, finish_reason="stop", seconds=0.1, token_count=4)


def execute(responses, *, token=None, verifier_checkpoint=None, result_checkpoint=None, on_progress=None):
    from engine.signals.verified_classification import classify_and_verify_transcription_run

    class Index:
        def shortlist(self, text, *, top_k, token):
            assert text == "camera settings Canon camera"
            return CANDIDATES

    segments = tuple(TranscriptSegment(time_range=TimeRange(start_seconds=i, end_seconds=i+1), text=text, confidence=0.9, language="en") for i, text in enumerate(("camera settings", "Canon camera")))
    run = TranscriptionRun(windows=(WindowResult(time_range=TimeRange(start_seconds=0, end_seconds=2), segments=segments, language="en"),), status="ready")
    saved, attempts, verified, results, progress = [], [], [], [], []
    fake = Generator(responses)
    output = classify_and_verify_transcription_run(run, taxonomy_index=Index(), generator=fake, token=token or CancellationToken(), candidate_checkpoint=saved.append, classification_attempt_checkpoint=attempts.append, verifier_attempt_checkpoint=verifier_checkpoint or verified.append, result_checkpoint=result_checkpoint or results.append, on_verifier_progress=on_progress or (lambda span, done, limit, state: progress.append((span, done, limit, state))))
    return output[0], fake, attempts, verified, results, progress


def test_verifier_owns_roles_and_citations_preserving_payload_values():
    result, _, attempts, verified, saved, progress = execute([primary(), payload(secondary=("633", "255")), verdict("514"), verdict("633"), verdict("255", "unsupported")])
    assert result.payload.primary_category == "514"
    assert result.payload.secondary_categories == ("633",)
    assert result.category_citations == (CategoryCitation(category_id="514", segment_indices=(1,)), CategoryCitation(category_id="633", segment_indices=(1,)))
    assert result.payload.keywords[0].term == "camera"
    assert result.payload.entities[0].text == "Canon"
    assert result.payload.sentiment.score == 0.7
    assert result.verification_policy_version == "english-transcript-verifier-v2"
    assert len(attempts) == 2 and len(verified) == 3 and saved == [result]
    assert [(done, limit, state) for _, done, limit, state in progress] == [(0,5,"starting"),(1,5,"completed"),(1,5,"starting"),(2,5,"completed"),(2,5,"starting"),(3,5,"completed")]


def test_rejected_and_uncertain_proposals_use_ranked_alternative_and_five_call_cap():
    result, _, _, verified, _, _ = execute([primary(), payload(secondary=("1", "2")), verdict("514", "unsupported"), verdict("633", "uncertain"), verdict("255"), verdict("1"), verdict("2")])
    assert result.payload.primary_category == "255"
    assert result.payload.secondary_categories == ("1", "2")
    assert len(verified) == 5


def test_all_rejected_becomes_semantic_abstention_without_proposed_citations():
    result, _, _, verified, _, _ = execute([primary(), payload(), verdict("514", "unsupported"), verdict("633", "uncertain"), verdict("255", "unsupported")])
    assert result.payload is None and result.category_citations == ()
    assert result.error_code == "INSUFFICIENT_TOPIC_EVIDENCE"
    assert result.verification_policy_version == "english-transcript-verifier-v2"
    assert len(verified) == 3


def test_classifier_abstention_recovers_with_real_locked_payload_attempt():
    result, _, attempts, verified, _, _ = execute([primary("__INSUFFICIENT_TOPIC_EVIDENCE__"), verdict("633", indices=(0,1)), payload("633")])
    assert result.payload.primary_category == "633"
    assert result.payload.keywords[0].term == "camera"
    assert result.category_citations == (CategoryCitation(category_id="633", segment_indices=(0,1)),)
    assert [attempt.stage for attempt in attempts] == ["primary", "payload"]
    assert attempts[-1].raw == payload("633") and len(verified) == 1


@pytest.mark.parametrize("response,code", [("broken","INVALID_VERIFIER_OUTPUT"), (GenerationResult(raw="partial", finish_reason="length", seconds=0.1, token_count=4),"INCOMPLETE_GENERATION"), (RuntimeError("broken"),"GENERATION_ERROR")])
def test_verifier_technical_failures_never_promote(response, code):
    result, _, _, verified, _, progress = execute([primary(), payload(), response])
    assert result.error_code == code and result.payload is None and result.category_citations == ()
    assert result.verification_policy_version is None
    assert len(verified) == 1 and progress[-1][1:] == (1,5,"completed")


@pytest.mark.parametrize("response,code", [("broken","INVALID_CLASSIFICATION_OUTPUT"), (RuntimeError("broken"),"GENERATION_ERROR"), (payload("514"),"INVALID_CLASSIFICATION_OUTPUT")])
def test_abstention_recovery_payload_failures_remain_technical(response, code):
    result, _, attempts, _, _, _ = execute([primary("__INSUFFICIENT_TOPIC_EVIDENCE__"), verdict("633"), response])
    assert result.error_code == code and result.payload is None
    assert len(attempts) == 2 and result.category_citations == ()


def test_classifier_technical_error_skips_verifier():
    result, _, _, verified, _, progress = execute(["broken"])
    assert result.error_code == "INVALID_CLASSIFICATION_OUTPUT"
    assert verified == progress == []


@pytest.mark.parametrize("when", ["before", "during", "after", "checkpoint", "result"])
def test_cancellation_prevents_promotion(when):
    token = CancellationToken()
    saved, progress, results = [], [], []
    if when == "before":
        token.cancel()
    def cancel_during(token):
        token.cancel()
        if when == "during":
            raise JobCancelled("cancelled")
        return verdict("514")
    def checkpoint(attempt):
        saved.append(attempt)
        if when == "checkpoint":
            token.cancel()
    def save_result(result):
        results.append(result)
        token.cancel()
    with pytest.raises(JobCancelled):
        execute([primary(), payload(), cancel_during if when in {"during", "after"} else verdict("514")], token=token, verifier_checkpoint=checkpoint, result_checkpoint=save_result if when == "result" else results.append, on_progress=lambda span, done, limit, state: progress.append((done,state)))
    assert len(saved) == (0 if when in {"before", "during"} else 1)
    assert progress == ([] if when == "before" else [(0,"starting")] if when == "during" else [(0,"starting"),(1,"completed")])
    assert len(results) == (1 if when == "result" else 0)
