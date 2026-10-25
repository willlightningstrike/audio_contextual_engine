import json

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange, TranscriptSegment
from engine.signals.taxonomy import TaxonomyCandidate
from engine.signals.text_types import ClassificationPayload, ClassificationResult, GenerationAttempt, GenerationResult, SentimentValue
from engine.signals.topic_verify import verifier_max_tokens
from engine.signals.verified_topics import transcript_candidate, verify_window_topics
from engine.standards import load_taxonomy
from engine.transcribe.types import AnalysisWindow


def candidate(id, path=None):
    return TaxonomyCandidate(id, id, path or (id,), 0.5)


@pytest.fixture
def window():
    return AnalysisWindow(time_range=TimeRange(start_seconds=0, end_seconds=4), segments=tuple(
        TranscriptSegment(time_range=TimeRange(start_seconds=i, end_seconds=i+1), text="camera settings", confidence=0.9, language="en") for i in range(2)
    ))


def proposal(window, primary="514", secondaries=(), error=None):
    payload = None if error else ClassificationPayload(primary_category=primary, secondary_categories=secondaries, keywords=(), entities=(), sentiment=SentimentValue(label="neutral", score=0.0))
    return ClassificationResult(time_range=window.time_range, payload=payload, error_code=error, attempts=(GenerationAttempt(attempt=1, prompt_sha256="0"*64),))


class Fake:
    def __init__(self, responses, cancel=False):
        self.responses = iter(responses)
        self.prompts = []
        self.cancel = cancel

    def generate_json(self, prompt, schema, *, max_tokens, token):
        segment_count = len(schema["oneOf"][0]["properties"]["segment_indices"]["items"]["enum"])
        assert max_tokens == verifier_max_tokens(segment_count)
        self.prompts.append(prompt)
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        if self.cancel:
            token.cancel()
        return item if isinstance(item, GenerationResult) else GenerationResult(raw=item, finish_reason="stop", seconds=0.1, token_count=8)


def verdict(id, status="supported", indices=(0,)):
    return json.dumps(dict(status=status, category_id=id, segment_indices=list(indices) if status == "supported" else []))


def run(window, proposed, responses, ranked=None, token=None, checkpoint=None):
    fake = Fake(responses)
    saved = []
    result = verify_window_topics(window, proposed, ranked or [candidate("633"), candidate("514"), candidate("255"), candidate("363", ("Entertainment", "Music", "Rock Music"))], generator=fake, token=token or CancellationToken(), checkpoint=checkpoint or saved.append)
    return result, fake, saved


def test_recover_ranked_camera(window):
    result, fake, saved = run(window, proposal(window), [verdict("514", "unsupported"), verdict("633")])
    assert result.primary_category == "633"
    assert result.citations[0].segment_indices == (0,)
    assert len(saved) == len(fake.prompts) == 2
    assert all('"category_id": "363"' not in p for p in fake.prompts)


def test_secondary_independent_citations(window):
    result, _, _ = run(window, proposal(window, secondaries=("633", "255")), [verdict("514"), verdict("633", indices=(1,)), verdict("255")])
    assert result.secondary_categories == ("633", "255")
    assert [c.segment_indices for c in result.citations] == [(0,), (1,), (0,)]


def test_unsupported_secondary(window):
    result, _, _ = run(window, proposal(window, secondaries=("633",)), [verdict("514"), verdict("633", "unsupported")])
    assert result.secondary_categories == ()


@pytest.mark.parametrize("raw", ["broken", '{"status":"supported","category_id":"514","segment_indices":[' + '9'*5000 + ']}'])
def test_invalid_stops_without_recovery(window, raw):
    result, fake, saved = run(window, proposal(window), [raw])
    assert result.error_code == "INVALID_VERIFIER_OUTPUT"
    assert result.primary_category is None and result.citations == ()
    assert len(saved) == len(fake.prompts) == 1
    assert saved[0].raw == raw


@pytest.mark.parametrize("error", ["INSUFFICIENT_TOPIC_EVIDENCE", None])
def test_abstention_and_music_use_three_candidates(window, error):
    result, fake, _ = run(window, proposal(window, primary="363", error=error), [verdict("633", "uncertain"), verdict("514", "unsupported"), verdict("255", "uncertain")])
    assert result.error_code is None and result.primary_category is None
    assert len(fake.prompts) == 3


@pytest.mark.parametrize("primary,error", [("999", None), ("514", "GENERATION_ERROR")])
def test_technical_proposal_no_recovery(window, primary, error):
    result, fake, _ = run(window, proposal(window, primary=primary, error=error), [])
    assert result.error_code is not None and result.primary_category is None
    assert fake.prompts == []


def test_hierarchy_removal(window):
    ranked = [candidate("514", ("Tech", "Camera")), candidate("633", ("Tech",)), candidate("255", ("Hobby",))]
    result, fake, _ = run(window, proposal(window, secondaries=("633", "255")), [verdict("514"), verdict("255")], ranked)
    assert result.secondary_categories == ("255",) and len(fake.prompts) == 2


def test_five_call_cap(window):
    ranked = [candidate(id) for id in ("633", "514", "255", "1", "2")]
    result, fake, _ = run(window, proposal(window, secondaries=("1", "2")), [verdict("514", "unsupported"), verdict("633", "unsupported"), verdict("255"), verdict("1"), verdict("2")], ranked)
    assert result.primary_category == "255" and len(fake.prompts) == 5


def test_required_call_exceeding_budget_fails_closed(window, monkeypatch):
    monkeypatch.setattr("engine.signals.verified_topics.MAX_VERIFIER_CALLS", 1)
    result, fake, _ = run(window, proposal(window, secondaries=("633",)), [verdict("514")])
    assert result.error_code == "VERIFIER_CALL_BUDGET_EXCEEDED"
    assert result.primary_category is None and len(fake.prompts) == 1


def test_3_1_music_genres_are_not_transcript_topics():
    music = [row for row in load_taxonomy() if row.path[:2] == ("Entertainment", "Music")]
    assert len(music) == 34   # 338 itself plus the 33 moved genre rows
    assert not any(transcript_candidate(candidate(row.id, row.path)) for row in music)


def test_non_music_entertainment_and_genres_stay_eligible():
    assert transcript_candidate(candidate("324", ("Entertainment", "Movies")))
    assert transcript_candidate(candidate("700", ("Genres", "Thriller")))
    assert not transcript_candidate(candidate("650", ("Genres", "Music Video")))


def test_music_video_and_music_parent_excluded(window):
    ranked = [candidate("338"), candidate("650"), candidate("514"), candidate("633")]
    result, fake, _ = run(window, proposal(window, primary="338", secondaries=("650",)), [verdict("514")], ranked)
    assert result.primary_category == "514" and result.secondary_categories == ()
    assert all('"category_id": "338"' not in p and '"category_id": "650"' not in p for p in fake.prompts)


def test_secondary_hierarchy_with_accepted_secondary(window):
    ranked = [candidate("514"), candidate("633", ("Tech",)), candidate("255", ("Tech", "Camera"))]
    result, fake, _ = run(window, proposal(window, secondaries=("633", "255")), [verdict("514"), verdict("633")], ranked)
    assert result.secondary_categories == ("633",) and len(fake.prompts) == 2


def test_raw_checkpoint_precedes_parser(window, monkeypatch):
    saved = []
    def parser(raw, id, window):
        assert saved[0].raw == raw
        raise ValueError("decoder failure")
    monkeypatch.setattr("engine.signals.verified_topics.parse_verifier_judgment", parser)
    result, _, _ = run(window, proposal(window), [verdict("514")], checkpoint=saved.append)
    assert result.error_code == "INVALID_VERIFIER_OUTPUT"


@pytest.mark.parametrize("during", [False, True])
def test_cancellation_persists_returned_call(window, during):
    token = CancellationToken()
    saved = []
    fake = Fake([verdict("514", "unsupported")], cancel=during)
    def checkpoint(attempt):
        saved.append(attempt)
        token.cancel()
    with pytest.raises(JobCancelled):
        verify_window_topics(window, proposal(window), [candidate("514"), candidate("633")], generator=fake, token=token, checkpoint=checkpoint)
    assert len(saved) == len(fake.prompts) == 1


def test_precall_cancel(window):
    token = CancellationToken()
    token.cancel()
    with pytest.raises(JobCancelled):
        run(window, proposal(window), [], token=token)


def test_progress_completed_follows_durable_checkpoint_even_for_invalid_output(window):
    events = []
    result = verify_window_topics(window, proposal(window), [candidate("514")], generator=Fake(["broken"]), token=CancellationToken(), checkpoint=lambda attempt: events.append(("checkpoint", attempt.raw)), on_verifier_progress=lambda span, done, limit, state: events.append((state, done, limit)))
    assert result.error_code == "INVALID_VERIFIER_OUTPUT"
    assert events == [("starting", 0, 5), ("checkpoint", "broken"), ("completed", 1, 5)]


@pytest.mark.parametrize("response,code", [(RuntimeError("broken"), "GENERATION_ERROR"), (GenerationResult(raw="partial", finish_reason="length", seconds=0.1, token_count=8), "INCOMPLETE_GENERATION")])
def test_technical_call_clears_categories(window, response, code):
    result, fake, saved = run(window, proposal(window, secondaries=("633",)), [verdict("514"), response])
    assert result.error_code == code and result.primary_category is None and result.citations == ()
    assert len(saved) == len(fake.prompts) == 2
    if isinstance(response, GenerationResult):
        assert saved[-1].raw == "partial" and saved[-1].finish_reason == "length"


def test_deep_ranking_caps_verifier_pool_at_fifteen_eligible(window):
    music = [candidate(f"m{i}", ("Entertainment", "Music", f"m{i}")) for i in range(20)]
    eligible = [candidate(f"e{i}") for i in range(20)]
    result, fake, _ = run(window, proposal(window, primary="m0"), [verdict("e0")], music + eligible)
    assert result.primary_category == "e0"
    competitors = json.loads(fake.prompts[0][fake.prompts[0].index("{"):])["competitors"]
    assert [item["category_id"] for item in competitors] == [f"e{i}" for i in range(1, 15)]
