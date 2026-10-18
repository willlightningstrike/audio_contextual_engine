from __future__ import annotations

import math
import warnings
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange, TranscriptSegment
from engine.models.asr import MlxAsrAdapter
from engine.transcribe.service import (
    MODEL_FAILURE,
    NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE,
    NO_USABLE_TRANSCRIPT,
    VOCAL_PRESENCE_UNCERTAIN,
    transcribe_media,
)
from engine.transcribe.types import ScreenResult, WindowResult


def _segment(start: float = 0, end: float = 1, text: str = "hello") -> TranscriptSegment:
    return TranscriptSegment(
        time_range=TimeRange(start_seconds=start, end_seconds=end),
        text=text,
        confidence=0.9,
        language="en",
    )


class FakeScreener:
    def __init__(self, results: list[ScreenResult] | None = None) -> None:
        self.results = iter(results or [])
        self.calls: list[Path] = []

    def screen(self, audio: Path, token: CancellationToken) -> ScreenResult:
        token.raise_if_cancelled()
        self.calls.append(audio)
        return next(self.results, ScreenResult(speech_max=0.9, singing_max=0, admitted=True))


class FakeTranscriber:
    def __init__(self) -> None:
        self.language_result: object = {"en": 0.9, "fr": 0.1}
        self.transcript_results: list[object] = []
        self.language_calls: list[tuple[Path, TimeRange]] = []
        self.asr_calls: list[tuple[Path, TimeRange, str]] = []

    def detect_language_probabilities(
        self, audio: Path, window: TimeRange, token: CancellationToken
    ) -> object:
        token.raise_if_cancelled()
        self.language_calls.append((audio, window))
        if isinstance(self.language_result, BaseException):
            raise self.language_result
        return self.language_result

    def transcribe_window(
        self, audio: Path, window: TimeRange, language: str, token: CancellationToken
    ) -> list[TranscriptSegment]:
        token.raise_if_cancelled()
        self.asr_calls.append((audio, window, language))
        if self.transcript_results:
            result = self.transcript_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result  # type: ignore[return-value]
        return [_segment()]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    duration: float = 1,
    transcriber: FakeTranscriber | None = None,
    screener: FakeScreener | None = None,
    token: CancellationToken | None = None,
    checkpoint=None,
    on_progress=None,
    **options,
):
    slices: list[TimeRange] = []

    def slice_spy(media, window: TimeRange, work_dir: Path, active_token: CancellationToken):
        active_token.raise_if_cancelled()
        slices.append(window)
        return work_dir / f"window-{len(slices)}.wav"

    monkeypatch.setattr("engine.transcribe.service.slice_audio_window", slice_spy)
    saved: list[WindowResult] = []
    progress: list[tuple[int, int]] = []
    media = SimpleNamespace(
        audio_path=tmp_path / "prepared.wav",
        duration_seconds=duration,
        requested_language="en",
    )
    run = transcribe_media(
        media,
        transcriber=transcriber or FakeTranscriber(),
        screener=screener or FakeScreener(),
        work_dir=tmp_path / "work",
        token=token or CancellationToken(),
        checkpoint=checkpoint or saved.append,
        on_progress=on_progress or (lambda completed, total: progress.append((completed, total))),
        **options,
    )
    return run, slices, saved, progress


def test_every_bounded_work_unit_screens_before_language_and_uses_local_asr_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    events: list[str] = []

    class OrderedScreener(FakeScreener):
        def screen(self, audio: Path, token: CancellationToken) -> ScreenResult:
            events.append(f"screen:{audio.name}")
            return super().screen(audio, token)

    class OrderedTranscriber(FakeTranscriber):
        def detect_language_probabilities(self, audio, window, token):
            events.append(f"language:{audio.name}")
            return super().detect_language_probabilities(audio, window, token)

    transcriber = OrderedTranscriber()
    screener = OrderedScreener()

    run, slices, saved, progress = _run(
        monkeypatch, tmp_path, duration=95, transcriber=transcriber, screener=screener
    )

    assert [(item.start_seconds, item.end_seconds) for item in slices] == [
        (0, 30),
        (30, 60),
        (60, 90),
        (90, 95),
    ]
    assert events == [
        "screen:window-1.wav",
        "language:window-1.wav",
        "screen:window-2.wav",
        "language:window-2.wav",
        "screen:window-3.wav",
        "language:window-3.wav",
        "screen:window-4.wav",
        "language:window-4.wav",
    ]
    assert [(call[1].start_seconds, call[1].end_seconds) for call in transcriber.asr_calls] == [
        (0, 30),
        (0, 30),
        (0, 30),
        (0, 5),
    ]
    assert all(call[2] == "en" for call in transcriber.asr_calls)
    assert run.status == "ready"
    assert tuple(saved) == run.windows
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_below_threshold_records_uncertainty_and_never_calls_asr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    screener = FakeScreener([ScreenResult(speech_max=0, singing_max=0, admitted=False)])

    run, _, saved, _ = _run(
        monkeypatch, tmp_path, transcriber=transcriber, screener=screener
    )

    assert run.status == "failed"
    assert saved[0].error_code == VOCAL_PRESENCE_UNCERTAIN
    assert saved[0].segments == ()
    assert transcriber.language_calls == []
    assert transcriber.asr_calls == []


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"en": math.nan},
        {"en": 1.1},
        {"en": True},
        {"en": 0.4, "fr": 0.6},
        {"en": 0.5, "fr": 0.5},
        ValueError("invalid ASR language probabilities"),
    ],
)
def test_only_a_finite_unique_top_english_language_is_admitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, probabilities: object
):
    transcriber = FakeTranscriber()
    transcriber.language_result = probabilities

    run, _, saved, _ = _run(monkeypatch, tmp_path, transcriber=transcriber)

    assert run.status == "failed"
    assert saved[0].error_code == NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE
    assert transcriber.asr_calls == []


@pytest.mark.parametrize(
    "raw_probabilities",
    [
        {"en": True},
        {"en": 0.1, "fr": "0.9"},
    ],
)
def test_malformed_adapter_language_map_cannot_bypass_service_english_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_probabilities: dict[str, object],
):
    audio = tmp_path / "bounded.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16_000)
    monkeypatch.setattr(
        "engine.transcribe.service.slice_audio_window",
        lambda *_args, **_kwargs: audio,
    )
    generate_calls = 0

    class Model:
        dims = SimpleNamespace(n_mels=80)

        def detect_language(self, _mel):
            return raw_probabilities

        def generate(self, *_args, **_kwargs):
            nonlocal generate_calls
            generate_calls += 1
            return {"segments": [{"start": 0, "end": 1, "text": "wrong branch"}]}

    transcriber = MlxAsrAdapter(
        tmp_path.resolve(),
        load_fn=lambda _path: Model(),
        language_inputs_fn=lambda _path, _n_mels: object(),
    )
    saved: list[WindowResult] = []
    run = transcribe_media(
        SimpleNamespace(audio_path=audio, duration_seconds=1),
        transcriber=transcriber,
        screener=FakeScreener(),
        work_dir=tmp_path / "work",
        token=CancellationToken(),
        checkpoint=saved.append,
        on_progress=lambda _completed, _total: None,
    )

    assert run.status == "failed"
    assert saved[0].error_code == NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE
    assert generate_calls == 0


def test_user_english_hint_cannot_bypass_detected_non_english(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.language_result = {"fr": 0.9, "en": 0.1}

    run, _, saved, _ = _run(monkeypatch, tmp_path, transcriber=transcriber)

    assert run.status == "failed"
    assert saved[0].error_code == NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE


def test_local_segment_timestamps_are_rebased_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [[_segment(1, 2)], [_segment(1, 2)], [_segment(1, 2)]]

    run, _, _, _ = _run(monkeypatch, tmp_path, duration=65, transcriber=transcriber)

    assert [
        (result.segments[0].time_range.start_seconds, result.segments[0].time_range.end_seconds)
        for result in run.windows
    ] == [(1, 2), (31, 32), (61, 62)]


def test_one_ulp_adjacent_overlap_is_corrected_and_records_original_local_timestamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [
        [_segment(0, 12.200000000000001, "first"), _segment(12.2, 13, "second")]
    ]

    run, _, saved, _ = _run(
        monkeypatch, tmp_path, duration=15, transcriber=transcriber
    )

    assert run.status == "ready"
    assert saved[0].segments[1].time_range.start_seconds == 12.200000000000001
    assert len(saved[0].timestamp_corrections) == 1
    correction = saved[0].timestamp_corrections[0]
    assert correction.segment_index == 1
    assert correction.original_start_seconds == 12.2
    assert correction.corrected_start_seconds == 12.200000000000001
    assert correction.local_end_seconds == 13


def test_two_representable_steps_below_binade_boundary_are_not_corrected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    first_end = 2.0
    second_start = math.nextafter(math.nextafter(first_end, -math.inf), -math.inf)
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [
        [_segment(0, first_end), _segment(second_start, 3.0)]
    ]

    run, _, saved, _ = _run(
        monkeypatch, tmp_path, duration=4, transcriber=transcriber
    )

    assert run.status == "failed"
    assert saved[0].error_code == MODEL_FAILURE
    assert saved[0].timestamp_corrections == ()


@pytest.mark.parametrize("overlap", [1e-9, 0.001, 0.02])
def test_genuine_adjacent_overlap_remains_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overlap: float
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [
        [_segment(0, 12.2 + overlap), _segment(12.2, 13)]
    ]

    run, _, saved, _ = _run(
        monkeypatch, tmp_path, duration=15, transcriber=transcriber
    )

    assert run.status == "failed"
    assert saved[0].error_code == MODEL_FAILURE
    assert saved[0].timestamp_corrections == ()


def test_one_ulp_audio_endpoint_overrun_remains_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [[_segment(0, math.nextafter(15, math.inf))]]

    run, _, saved, _ = _run(
        monkeypatch, tmp_path, duration=15, transcriber=transcriber
    )

    assert run.status == "failed"
    assert saved[0].error_code == MODEL_FAILURE
    assert saved[0].timestamp_corrections == ()


@pytest.mark.parametrize(
    "second_range",
    [
        SimpleNamespace(start_seconds=0.0, end_seconds=1.0),
        SimpleNamespace(start_seconds=-math.ulp(0.0), end_seconds=1.0),
        SimpleNamespace(start_seconds=math.nan, end_seconds=1.0),
    ],
)
def test_zero_negative_or_nonfinite_adjacent_start_is_not_repaired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, second_range: object
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [
        [
            _segment(0, math.ulp(0.0)),
            SimpleNamespace(
                time_range=second_range,
                text="bad",
                confidence=0.9,
                language="en",
            ),
        ]
    ]

    run, _, saved, _ = _run(monkeypatch, tmp_path, transcriber=transcriber)

    assert run.status == "failed"
    assert saved[0].error_code == MODEL_FAILURE
    assert saved[0].timestamp_corrections == ()


def test_one_ulp_snap_cannot_create_zero_length_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    previous_end = math.nextafter(1.0, math.inf)
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [
        [_segment(0, previous_end), _segment(1.0, previous_end)]
    ]

    run, _, saved, _ = _run(monkeypatch, tmp_path, transcriber=transcriber)

    assert run.status == "failed"
    assert saved[0].error_code == MODEL_FAILURE
    assert saved[0].timestamp_corrections == ()


def test_real_gap_is_not_corrected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [[_segment(0, 12.2), _segment(12.3, 13)]]

    run, _, saved, _ = _run(
        monkeypatch, tmp_path, duration=15, transcriber=transcriber
    )

    assert run.status == "ready"
    assert saved[0].segments[1].time_range.start_seconds == 12.3
    assert saved[0].timestamp_corrections == ()


def test_correction_uses_local_values_and_rebases_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [
        [_segment()],
        [_segment(0, 12.200000000000001), _segment(12.2, 13)],
    ]

    run, _, saved, _ = _run(
        monkeypatch, tmp_path, duration=45, transcriber=transcriber
    )

    assert run.status == "ready"
    assert saved[1].segments[1].time_range.start_seconds == 42.2
    assert saved[1].timestamp_corrections[0].original_start_seconds == 12.2
    assert saved[1].timestamp_corrections[0].corrected_start_seconds == 12.200000000000001


@pytest.mark.parametrize(
    "time_range",
    [
        SimpleNamespace(start_seconds=-1, end_seconds=1),
        SimpleNamespace(start_seconds=2, end_seconds=1),
        SimpleNamespace(start_seconds=0, end_seconds=31),
    ],
)
def test_invalid_local_timestamps_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, time_range: object
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [
        [SimpleNamespace(time_range=time_range, text="bad", confidence=0.9, language="en")]
    ]

    run, _, saved, _ = _run(monkeypatch, tmp_path, duration=30, transcriber=transcriber)

    assert run.status == "failed"
    assert saved[0].error_code == MODEL_FAILURE
    assert saved[0].segments == ()


@pytest.mark.parametrize(
    "segments",
    [
        [_segment(2, 3), _segment(1, 2)],
        [_segment(0, 2), _segment(1, 3)],
    ],
)
def test_unordered_or_overlapping_local_segments_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, segments: list[TranscriptSegment]
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [segments]

    run, _, saved, _ = _run(monkeypatch, tmp_path, duration=30, transcriber=transcriber)

    assert run.status == "failed"
    assert saved[0].error_code == MODEL_FAILURE


def test_empty_asr_has_no_usable_transcript_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [[]]

    run, _, saved, _ = _run(monkeypatch, tmp_path, transcriber=transcriber)

    assert run.status == "failed"
    assert saved[0].error_code == NO_USABLE_TRANSCRIPT


def test_blank_segment_warning_is_not_suppressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class WarningTranscriber(FakeTranscriber):
        def transcribe_window(self, audio, window, language, token):
            warnings.warn("blank segment skipped", UserWarning)
            return [_segment()]

    with pytest.warns(UserWarning, match="blank segment skipped"):
        run, _, _, _ = _run(monkeypatch, tmp_path, transcriber=WarningTranscriber())

    assert run.status == "ready"


def test_ordinary_model_failure_is_checkpointed_and_later_windows_continue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [RuntimeError("model failed"), [_segment()]]

    run, _, saved, progress = _run(
        monkeypatch, tmp_path, duration=31, transcriber=transcriber
    )

    assert run.status == "partial"
    assert [result.error_code for result in saved] == [MODEL_FAILURE, None]
    assert progress == [(1, 2), (2, 2)]


def test_non_english_window_stops_after_checkpointing_partial_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    language_results = iter(({"en": 0.9}, {"fr": 0.9}, {"en": 0.9}))
    transcriber.detect_language_probabilities = lambda audio, window, token: next(language_results)

    run, slices, saved, progress = _run(
        monkeypatch, tmp_path, duration=61, transcriber=transcriber
    )

    assert run.status == "partial"
    assert len(slices) == 2
    assert [result.error_code for result in saved] == [None, NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE]
    assert progress == [(1, 3), (2, 3)]


def test_non_english_windows_within_allowance_do_not_stop_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    language_results = iter(({"en": 0.9}, {"fr": 0.9}, {"en": 0.9}))
    transcriber.detect_language_probabilities = lambda audio, window, token: next(language_results)

    run, slices, saved, _ = _run(
        monkeypatch, tmp_path, duration=90, transcriber=transcriber,
        non_english_stop_seconds=30,
    )

    assert len(slices) == 3
    assert [result.error_code for result in saved] == [
        None, NON_ENGLISH_OR_UNCONFIRMED_LANGUAGE, None,
    ]
    assert run.status == "partial"


def test_non_english_windows_beyond_allowance_stop_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    transcriber = FakeTranscriber()
    transcriber.language_result = {"fr": 0.9}

    _, slices, saved, _ = _run(
        monkeypatch, tmp_path, duration=150, transcriber=transcriber,
        non_english_stop_seconds=45,
    )

    assert len(slices) == len(saved) == 2


@pytest.mark.parametrize("value", [-1, float("nan"), True, "30"])
def test_non_english_allowance_must_be_finite_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value
):
    with pytest.raises(ValueError):
        _run(monkeypatch, tmp_path, non_english_stop_seconds=value)


def test_checkpoint_failure_propagates_before_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    progress: list[tuple[int, int]] = []

    with pytest.raises(OSError, match="checkpoint failed"):
        _run(
            monkeypatch,
            tmp_path,
            checkpoint=lambda _result: (_ for _ in ()).throw(OSError("checkpoint failed")),
            on_progress=lambda completed, total: progress.append((completed, total)),
        )

    assert progress == []


def test_progress_failure_propagates_after_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    saved: list[WindowResult] = []

    with pytest.raises(OSError, match="progress failed"):
        _run(
            monkeypatch,
            tmp_path,
            checkpoint=saved.append,
            on_progress=lambda completed, total: (_ for _ in ()).throw(OSError("progress failed")),
        )

    assert len(saved) == 1


@pytest.mark.parametrize("failure", [JobCancelled(), KeyboardInterrupt()])
def test_cancellation_and_interrupts_propagate_without_error_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: BaseException
):
    transcriber = FakeTranscriber()
    transcriber.transcript_results = [failure]
    saved: list[WindowResult] = []

    with pytest.raises(type(failure)):
        _run(monkeypatch, tmp_path, transcriber=transcriber, checkpoint=saved.append)

    assert saved == []
