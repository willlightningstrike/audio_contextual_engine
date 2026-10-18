from __future__ import annotations

from engine.domain import TimeRange, TranscriptSegment
from engine.signals.text_types import (
    ClassificationPayload,
    ClassificationResult,
    GenerationAttempt,
    SentimentValue,
)
from engine.transcribe.types import TranscriptionRun, WindowResult


def _run(status: str = "ready") -> TranscriptionRun:
    return TranscriptionRun(
        status=status,
        windows=(
            WindowResult(
                time_range=TimeRange(start_seconds=0, end_seconds=10),
                segments=(
                    TranscriptSegment(
                        time_range=TimeRange(start_seconds=1, end_seconds=3),
                        text="English speech",
                        confidence=0.9,
                        language="en",
                    ),
                ),
                language="en",
            ),
        ),
    )


def _classification(start: float = 1, end: float = 3) -> ClassificationResult:
    return ClassificationResult(
        time_range=TimeRange(start_seconds=start, end_seconds=end),
        payload=ClassificationPayload(
            primary_category="A",
            secondary_categories=(),
            keywords=(),
            entities=(),
            sentiment=SentimentValue(label="neutral", score=0.0),
        ),
        attempts=(
            GenerationAttempt(
                attempt=1,
                stage="primary",
                prompt_sha256="0" * 64,
                raw='{"primary_category":"A"}',
                finish_reason="stop",
                seconds=0.1,
                token_count=5,
            ),
        ),
    )


def _uncertain_vocal_window(start: float, end: float) -> WindowResult:
    return WindowResult(
        time_range=TimeRange(start_seconds=start, end_seconds=end),
        error_code="VOCAL_PRESENCE_UNCERTAIN",
        failure_stage="vocal_screening",
        failure_reason="vocal_presence_uncertain",
    )


def test_partial_run_cannot_complete():
    from engine.completion import completion_errors

    errors = completion_errors(_run("partial"), (_classification(),))

    assert errors
    assert any("partial" in error for error in errors)


def test_classification_failure_cannot_complete():
    from engine.completion import completion_errors

    failed = ClassificationResult(
        time_range=TimeRange(start_seconds=1, end_seconds=3),
        payload=None,
        attempts=(
            GenerationAttempt(
                attempt=1,
                stage="primary",
                prompt_sha256="0" * 64,
                error_code="INVALID_CLASSIFICATION_OUTPUT",
            ),
        ),
        error_code="INVALID_CLASSIFICATION_OUTPUT",
    )

    assert "classification_failed:INVALID_CLASSIFICATION_OUTPUT" in completion_errors(_run(), (failed,))


def test_primary_semantic_abstention_covers_ready_english_run():
    from engine.completion import completion_errors

    abstained = ClassificationResult(
        time_range=TimeRange(start_seconds=1, end_seconds=3),
        payload=None,
        attempts=(GenerationAttempt(
            attempt=1, stage="primary", prompt_sha256="0" * 64,
            raw='{"primary_category":"__INSUFFICIENT_TOPIC_EVIDENCE__"}',
            finish_reason="stop",
        ),),
        error_code="INSUFFICIENT_TOPIC_EVIDENCE",
        policy_version="english-audio-topic-sufficiency-v1",
    )

    assert completion_errors(_run(), (abstained,)) == ()
    assert "transcript_segment_classification_coverage_incomplete" in completion_errors(
        _run(), (abstained, abstained)
    )


def test_unresolved_screening_diagnostic_cannot_complete_even_without_error_code():
    from engine.completion import completion_errors

    run = _run()
    uncertain = TranscriptionRun(
        status="ready",
        windows=(
            WindowResult(
                **{
                    **run.windows[0].model_dump(mode="python"),
                    "failure_stage": "vocal_screening",
                }
            ),
        ),
    )

    assert any(
        error.startswith("transcription_window_uncertain")
        for error in completion_errors(uncertain, (_classification(),))
    )


def test_every_transcript_segment_requires_classification_coverage():
    from engine.completion import completion_errors

    run = _run()
    assert completion_errors(run, ())


def test_ready_fully_classified_run_has_no_completion_errors():
    from engine.completion import completion_errors

    assert completion_errors(_run(), (_classification(),)) == ()


def test_one_terminal_uncertain_vocal_window_is_excluded_after_successful_english_windows():
    from engine.completion import completion_errors

    first = _run().windows[0].model_copy(
        update={"time_range": TimeRange(start_seconds=0, end_seconds=30)}
    )
    run = TranscriptionRun(
        status="partial",
        windows=(first, _uncertain_vocal_window(30, 45)),
    )

    assert completion_errors(run, (_classification(),)) == ()


def test_interior_uncertain_vocal_window_still_blocks_completion():
    from engine.completion import completion_errors

    first = _run().windows[0]
    last = first.model_copy(
        update={
            "time_range": TimeRange(start_seconds=20, end_seconds=30),
            "segments": (
                first.segments[0].model_copy(
                    update={"time_range": TimeRange(start_seconds=21, end_seconds=23)}
                ),
            ),
        }
    )
    run = TranscriptionRun(
        status="partial",
        windows=(first, _uncertain_vocal_window(10, 20), last),
    )

    errors = completion_errors(
        run,
        (_classification(), _classification(start=21, end=23)),
    )

    assert any("VOCAL_PRESENCE_UNCERTAIN" in error for error in errors)


def test_multiple_uncertain_tail_windows_still_block_completion():
    from engine.completion import completion_errors

    first = _run().windows[0].model_copy(
        update={"time_range": TimeRange(start_seconds=0, end_seconds=10)}
    )
    run = TranscriptionRun(
        status="partial",
        windows=(
            first,
            _uncertain_vocal_window(10, 20),
            _uncertain_vocal_window(20, 25),
        ),
    )

    assert completion_errors(run, (_classification(),))


def test_uncertain_terminal_window_over_30_seconds_still_blocks_completion():
    from engine.completion import completion_errors

    first = _run().windows[0].model_copy(
        update={"time_range": TimeRange(start_seconds=0, end_seconds=10)}
    )
    run = TranscriptionRun(
        status="partial",
        windows=(first, _uncertain_vocal_window(10, 41)),
    )

    assert completion_errors(run, (_classification(),))


def test_noncontiguous_history_with_uncertain_tail_still_blocks_completion():
    from engine.completion import completion_errors

    first = _run().windows[0].model_copy(
        update={"time_range": TimeRange(start_seconds=0, end_seconds=10)}
    )
    run = TranscriptionRun(
        status="partial",
        windows=(first, _uncertain_vocal_window(11, 20)),
    )

    assert completion_errors(run, (_classification(),))


def test_uncertain_only_clip_still_fails_as_out_of_scope():
    from engine.completion import completion_errors

    run = TranscriptionRun(
        status="failed",
        windows=(_uncertain_vocal_window(0, 10),),
    )

    errors = completion_errors(run, ())

    assert "no_usable_transcript" in errors
    assert any("VOCAL_PRESENCE_UNCERTAIN" in error for error in errors)
