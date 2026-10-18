from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest
from pydantic import ValidationError

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange, TranscriptSegment
from engine.models.asr import MlxAsrAdapter
from engine.models.asr import NoUsableTranscriptError
from engine.transcribe.service import (
    MODEL_FAILURE,
    NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE,
    _rebase_segments_with_corrections,
    transcribe_media,
)
from engine.transcribe.types import ScreenResult, WindowResult


def _segment(start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(
        time_range=TimeRange(start_seconds=0, end_seconds=1),
        text="example",
        confidence=0.8,
        language="en",
    ).model_copy(
        update={"time_range": SimpleNamespace(start_seconds=start, end_seconds=end)}
    )


class _Screener:
    def screen(self, _audio: Path, token: CancellationToken) -> ScreenResult:
        token.raise_if_cancelled()
        return ScreenResult(speech_max=1, singing_max=0, admitted=True)


class _Transcriber:
    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self.segments = segments

    def detect_language_probabilities(self, _audio, _window, token):
        token.raise_if_cancelled()
        return {"en": 1.0}

    def transcribe_window(self, _audio, _window, _language, token):
        token.raise_if_cancelled()
        return self.segments


def _wav(tmp_path: Path, seconds: float = 30.0) -> Path:
    path = tmp_path / "window.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * int(16_000 * seconds))
    return path.resolve()


def _raw_segment(
    start: float,
    end: float,
    text: str,
    *,
    probability: float,
) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "text": text,
        "confidence": 0.8,
        "words": [
            {
                "word": f" {text}",
                "start": start,
                "end": min(start + 0.2, end),
                "probability": probability,
            }
        ],
    }


def test_word_timestamp_adapter_filters_only_contiguous_anomalous_tail(
    tmp_path: Path,
):
    audio = _wav(tmp_path)
    generated: list[dict[str, object]] = []

    class Model:
        def generate(self, _path: str, **kwargs: object):
            generated.append(kwargs)
            return {
                "segments": [
                    _raw_segment(0.0, 5.0, "opening", probability=0.99),
                    _raw_segment(27.9, 28.0, "interior", probability=0.01),
                    _raw_segment(28.1, 29.0, "supported", probability=0.99),
                    _raw_segment(29.2, 29.5, "hallucination-one", probability=0.01),
                    _raw_segment(29.5, 29.98, "hallucination-two", probability=0.01),
                ]
            }

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())

    segments = adapter.transcribe_window(
        audio,
        TimeRange(start_seconds=0, end_seconds=30),
        "en",
        CancellationToken(),
    )

    assert [segment.text for segment in segments] == [
        "opening",
        "interior",
        "supported",
    ]
    assert generated == [
        {
            "language": "en",
            "task": "transcribe",
            "return_timestamps": True,
            "word_timestamps": True,
            "verbose": False,
        }
    ]


def test_tail_filter_stops_at_supported_terminal_segment(tmp_path: Path):
    audio = _wav(tmp_path)

    class Model:
        def generate(self, _path: str, **_kwargs: object):
            return {
                "segments": [
                    _raw_segment(0.0, 5.0, "opening", probability=0.99),
                    _raw_segment(29.0, 29.4, "low-confidence", probability=0.01),
                    _raw_segment(29.4, 29.9, "supported-ending", probability=0.99),
                ]
            }

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())

    segments = adapter.transcribe_window(
        audio,
        TimeRange(start_seconds=0, end_seconds=30),
        "en",
        CancellationToken(),
    )

    assert [segment.text for segment in segments] == [
        "opening",
        "low-confidence",
        "supported-ending",
    ]


def test_wordless_out_of_bounds_last_nonblank_tail_does_not_block_valid_speech(
    tmp_path: Path,
):
    audio = _wav(tmp_path)

    class Model:
        def generate(self, _path: str, **_kwargs: object):
            return {
                "segments": [
                    _raw_segment(0.0, 5.0, "supported speech", probability=0.99),
                    _raw_segment(28.72, 29.9, "low-probability tail", probability=0.001),
                    {
                        "start": 48.48,
                        "end": 49.4,
                        "text": "To consist",
                        "words": [],
                        "avg_logprob": -5.18,
                    },
                    {"start": 29.96, "end": 29.96, "text": "", "words": []},
                ]
            }

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())
    segments = adapter.transcribe_window(
        audio,
        TimeRange(start_seconds=0, end_seconds=30),
        "en",
        CancellationToken(),
    )
    rebased, _ = _rebase_segments_with_corrections(
        segments, TimeRange(start_seconds=0, end_seconds=30)
    )

    assert [segment.text for segment in rebased] == ["supported speech"]


@pytest.mark.parametrize(
    "tail",
    [
        [_raw_segment(48.48, 49.4, "word-supported outside audio", probability=0.99)],
        [{"start": 29.7, "end": 30.4, "text": "straddles end", "words": []}],
        [
            {"start": 48.48, "end": 49.4, "text": "outside audio", "words": []},
            _raw_segment(29.7, 29.9, "supported ending", probability=0.99),
        ],
        [
            {"start": 48.48, "end": 49.4, "text": "first outside", "words": []},
            {"start": 49.5, "end": 49.9, "text": "second outside", "words": []},
        ],
    ],
)
def test_other_invalid_timestamps_are_not_quarantined(
    tmp_path: Path, tail: list[dict[str, object]]
):
    audio = _wav(tmp_path)

    class Model:
        def generate(self, _path: str, **_kwargs: object):
            return {
                "segments": [
                    _raw_segment(0.0, 5.0, "supported speech", probability=0.99),
                    *tail,
                ]
            }

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())
    segments = adapter.transcribe_window(
        audio,
        TimeRange(start_seconds=0, end_seconds=30),
        "en",
        CancellationToken(),
    )

    with pytest.raises(ValueError, match="ASR segment timestamps are invalid"):
        _rebase_segments_with_corrections(
            segments, TimeRange(start_seconds=0, end_seconds=30)
        )


def test_lone_wordless_out_of_bounds_segment_cannot_become_usable_transcript(
    tmp_path: Path,
):
    audio = _wav(tmp_path)

    class Model:
        def generate(self, _path: str, **_kwargs: object):
            return {
                "segments": [
                    {"start": 48.48, "end": 49.4, "text": "outside audio", "words": []}
                ]
            }

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())
    segments = adapter.transcribe_window(
        audio,
        TimeRange(start_seconds=0, end_seconds=30),
        "en",
        CancellationToken(),
    )
    with pytest.raises(ValueError, match="ASR segment timestamps are invalid"):
        _rebase_segments_with_corrections(
            segments, TimeRange(start_seconds=0, end_seconds=30)
        )


def test_terminal_segment_ending_exactly_two_seconds_before_boundary_is_retained(
    tmp_path: Path,
):
    audio = _wav(tmp_path)

    class Model:
        def generate(self, _path: str, **_kwargs: object):
            return {
                "segments": [
                    _raw_segment(27.0, 28.0, "exact-boundary", probability=0.01)
                ]
            }

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())

    segments = adapter.transcribe_window(
        audio,
        TimeRange(start_seconds=0, end_seconds=30),
        "en",
        CancellationToken(),
    )

    assert [segment.text for segment in segments] == ["exact-boundary"]


def test_punctuation_only_output_does_not_become_transcript_evidence(
    tmp_path: Path,
):
    audio = _wav(tmp_path)

    class Model:
        def generate(self, _path: str, **_kwargs: object):
            return {
                "segments": [
                    {
                        "start": 29.0,
                        "end": 29.5,
                        "text": "!!!",
                        "confidence": 1.0,
                        "words": [
                            {
                                "word": "!",
                                "start": 29.0,
                                "end": 29.2,
                                "probability": 0.99,
                            }
                        ],
                    }
                ]
            }

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())

    with pytest.raises(NoUsableTranscriptError):
        adapter.transcribe_window(
            audio,
            TimeRange(start_seconds=0, end_seconds=30),
            "en",
            CancellationToken(),
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [(-0.1, 1.0), (0.0, math.inf), (2.0, 1.0)],
)
def test_raw_invalid_adapter_timestamps_are_reported_at_timestamp_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    start: float,
    end: float,
):
    audio = _wav(tmp_path)

    class Model:
        dims = SimpleNamespace(n_mels=80)

        def detect_language(self, _features):
            return None, {"en": 1.0}

        def generate(self, _path: str, **_kwargs: object):
            return {
                "segments": [
                    {
                        "start": start,
                        "end": end,
                        "text": "example",
                        "confidence": 0.9,
                        "words": [],
                    }
                ]
            }

    adapter = MlxAsrAdapter(
        tmp_path.resolve(),
        load_fn=lambda _path: Model(),
        language_inputs_fn=lambda _path, _n_mels: object(),
    )
    monkeypatch.setattr(
        "engine.transcribe.service.slice_audio_window",
        lambda *_args, **_kwargs: audio,
    )
    saved: list[WindowResult] = []

    transcribe_media(
        SimpleNamespace(audio_path=audio, duration_seconds=30),
        transcriber=adapter,
        screener=_Screener(),
        work_dir=tmp_path,
        token=CancellationToken(),
        checkpoint=saved.append,
        on_progress=lambda _completed, _total: None,
    )

    assert saved[0].error_code == MODEL_FAILURE
    assert saved[0].failure_stage == "timestamp_validation"
    assert saved[0].failure_reason == "invalid_timestamps"


def test_malformed_word_metadata_fails_closed(tmp_path: Path):
    audio = _wav(tmp_path)

    class Model:
        def generate(self, _path: str, **_kwargs: object):
            segment = _raw_segment(29.0, 29.5, "malformed", probability=0.01)
            segment["words"] = [
                {
                    "word": " malformed",
                    "start": math.nan,
                    "end": 29.2,
                    "probability": 0.01,
                }
            ]
            return {"segments": [segment]}

    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _path: Model())

    with pytest.raises(ValueError, match="invalid ASR word timestamps"):
        adapter.transcribe_window(
            audio,
            TimeRange(start_seconds=0, end_seconds=30),
            "en",
            CancellationToken(),
        )


def _run_invalid_timestamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, segments: list[TranscriptSegment]
) -> WindowResult:
    monkeypatch.setattr(
        "engine.transcribe.service.slice_audio_window",
        lambda *_args, **_kwargs: tmp_path / "window.wav",
    )
    saved: list[WindowResult] = []
    transcribe_media(
        SimpleNamespace(audio_path=tmp_path / "input.wav", duration_seconds=30),
        transcriber=_Transcriber(segments),
        screener=_Screener(),
        work_dir=tmp_path,
        token=CancellationToken(),
        checkpoint=saved.append,
        on_progress=lambda _completed, _total: None,
    )
    return saved[0]


@pytest.mark.parametrize(
    "segments",
    [
        [_segment(27.2, 32)],
        [_segment(-0.1, 1)],
        [_segment(0, math.inf)],
        [_segment(0, 2), _segment(1, 3)],
    ],
)
def test_invalid_timestamp_failure_has_safe_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    segments: list[TranscriptSegment],
):
    result = _run_invalid_timestamp(monkeypatch, tmp_path, segments)

    assert result.error_code == MODEL_FAILURE
    assert result.failure_stage == "timestamp_validation"
    assert result.failure_reason == "invalid_timestamps"
    assert result.attempt_count == 1


def test_window_result_attempt_count_must_be_a_strict_positive_integer():
    window = TimeRange(start_seconds=0, end_seconds=1)

    for invalid in (0, -1, 1.0, True, "1"):
        with pytest.raises(ValidationError):
            WindowResult(time_range=window, attempt_count=invalid)

    assert WindowResult(time_range=window, attempt_count=2).attempt_count == 2


def test_window_result_old_payload_receives_backward_compatible_defaults():
    result = WindowResult.model_validate(
        {
            "time_range": {"start_seconds": 0, "end_seconds": 1},
            "segments": [],
            "language": "en",
            "error_code": "MODEL_FAILURE",
        }
    )

    assert result.failure_stage is None
    assert result.failure_reason is None
    assert result.attempt_count == 1


def test_model_exception_text_and_path_are_not_exposed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class FailingTranscriber(_Transcriber):
        def transcribe_window(self, _audio, _window, _language, _token):
            raise RuntimeError("secret at /Users/example/private.wav")

    monkeypatch.setattr(
        "engine.transcribe.service.slice_audio_window",
        lambda *_args, **_kwargs: tmp_path / "window.wav",
    )
    saved: list[WindowResult] = []
    transcribe_media(
        SimpleNamespace(audio_path=tmp_path / "input.wav", duration_seconds=30),
        transcriber=FailingTranscriber([]),
        screener=_Screener(),
        work_dir=tmp_path,
        token=CancellationToken(),
        checkpoint=saved.append,
        on_progress=lambda _completed, _total: None,
    )

    assert saved[0].failure_stage == "transcription"
    assert saved[0].failure_reason == "model_call_failed"
    assert "secret" not in saved[0].model_dump_json()
    assert "/Users" not in saved[0].model_dump_json()


@pytest.mark.parametrize(
    ("failing_stage", "expected_stage", "expected_reason"),
    [
        ("slice", "audio_slicing", "audio_slice_failed"),
        ("screen", "vocal_screening", "model_call_failed"),
        ("language", "language_detection", "model_call_failed"),
    ],
)
def test_stage_failures_record_only_safe_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing_stage: str,
    expected_stage: str,
    expected_reason: str,
):
    private_message = "secret at /Users/example/private.wav"

    def slice_audio(*_args, **_kwargs):
        if failing_stage == "slice":
            raise RuntimeError(private_message)
        return tmp_path / "window.wav"

    class StageScreener(_Screener):
        def screen(self, audio: Path, token: CancellationToken) -> ScreenResult:
            if failing_stage == "screen":
                raise RuntimeError(private_message)
            return super().screen(audio, token)

    class StageTranscriber(_Transcriber):
        def detect_language_probabilities(self, audio, window, token):
            if failing_stage == "language":
                raise RuntimeError(private_message)
            return super().detect_language_probabilities(audio, window, token)

    monkeypatch.setattr("engine.transcribe.service.slice_audio_window", slice_audio)
    saved: list[WindowResult] = []

    transcribe_media(
        SimpleNamespace(audio_path=tmp_path / "input.wav", duration_seconds=30),
        transcriber=StageTranscriber([]),
        screener=StageScreener(),
        work_dir=tmp_path,
        token=CancellationToken(),
        checkpoint=saved.append,
        on_progress=lambda _completed, _total: None,
    )

    result = saved[0]
    assert result.error_code == MODEL_FAILURE
    assert result.failure_stage == expected_stage
    assert result.failure_reason == expected_reason
    assert private_message not in result.model_dump_json()


@pytest.mark.parametrize(
    ("callback_name", "failure"),
    [
        ("checkpoint", OSError("checkpoint failed")),
        ("progress", OSError("progress failed")),
        ("checkpoint", JobCancelled()),
        ("progress", JobCancelled()),
    ],
)
def test_callback_failures_and_cancellation_propagate_without_conversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    callback_name: str,
    failure: BaseException,
):
    monkeypatch.setattr(
        "engine.transcribe.service.slice_audio_window",
        lambda *_args, **_kwargs: tmp_path / "window.wav",
    )
    saved: list[WindowResult] = []

    def fail(*_args):
        raise failure

    with pytest.raises(type(failure), match=str(failure) or None):
        transcribe_media(
            SimpleNamespace(audio_path=tmp_path / "input.wav", duration_seconds=30),
            transcriber=_Transcriber([_segment(0, 1)]),
            screener=_Screener(),
            work_dir=tmp_path,
            token=CancellationToken(),
            checkpoint=fail if callback_name == "checkpoint" else saved.append,
            on_progress=fail if callback_name == "progress" else lambda *_args: None,
        )

    assert len(saved) == (0 if callback_name == "checkpoint" else 1)
    if callback_name == "progress":
        assert saved[0].error_code is None
        assert saved[0].failure_stage is None
        assert saved[0].failure_reason is None


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"en": True},
        {"en": math.nan},
        {"en": math.inf},
        {"en": -0.1},
        {"en": 1.1},
        {"": 1.0},
        {1: 1.0},
        {"en": "1.0"},
    ],
)
def test_malformed_returned_language_maps_have_invalid_probability_diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probabilities: object
):
    class LanguageTranscriber(_Transcriber):
        def detect_language_probabilities(self, _audio, _window, _token):
            return probabilities

    monkeypatch.setattr(
        "engine.transcribe.service.slice_audio_window",
        lambda *_args, **_kwargs: tmp_path / "window.wav",
    )
    saved: list[WindowResult] = []

    transcribe_media(
        SimpleNamespace(audio_path=tmp_path / "input.wav", duration_seconds=30),
        transcriber=LanguageTranscriber([]),
        screener=_Screener(),
        work_dir=tmp_path,
        token=CancellationToken(),
        checkpoint=saved.append,
        on_progress=lambda *_args: None,
    )

    assert saved[0].error_code == NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE
    assert saved[0].failure_stage == "language_detection"
    assert saved[0].failure_reason == "invalid_language_probabilities"


@pytest.mark.parametrize(
    "probabilities",
    [{"en": 0.5, "fr": 0.5}, {"fr": 0.8, "en": 0.2}],
)
def test_valid_unconfirmed_language_maps_are_not_diagnosed_as_malformed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    probabilities: dict[str, float],
):
    class LanguageTranscriber(_Transcriber):
        def detect_language_probabilities(self, _audio, _window, _token):
            return probabilities

    monkeypatch.setattr(
        "engine.transcribe.service.slice_audio_window",
        lambda *_args, **_kwargs: tmp_path / "window.wav",
    )
    saved: list[WindowResult] = []

    transcribe_media(
        SimpleNamespace(audio_path=tmp_path / "input.wav", duration_seconds=30),
        transcriber=LanguageTranscriber([]),
        screener=_Screener(),
        work_dir=tmp_path,
        token=CancellationToken(),
        checkpoint=saved.append,
        on_progress=lambda *_args: None,
    )

    assert saved[0].error_code == NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE
    assert saved[0].failure_reason == "non_english_or_unconfirmed_language"
