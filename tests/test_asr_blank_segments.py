import wave
from types import SimpleNamespace

import pytest

from engine.cancel import CancellationToken
from engine.domain import TimeRange
from engine.models.asr import MlxAsrAdapter


def transcribe(tmp_path, segments):
    audio = tmp_path / "audio.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000)
    adapter = MlxAsrAdapter(tmp_path.resolve(), load_fn=lambda _: SimpleNamespace(
        generate=lambda *args, **kwargs: {"segments": segments}))
    return adapter.transcribe_window(audio, TimeRange(start_seconds=0, end_seconds=1), "en", CancellationToken())


def test_blank_segment_warns_and_preserves_usable_text(tmp_path):
    with pytest.warns(UserWarning, match="blank ASR segment"):
        result = transcribe(tmp_path, [{"start": 0, "end": .5, "text": "  "},
                                       {"start": .5, "end": 1, "text": "hello"}])
    assert [s.text for s in result] == ["hello"]


def test_no_segments_is_an_error_not_no_vocals(tmp_path):
    with pytest.raises(ValueError, match="no usable transcript"):
        transcribe(tmp_path, [])


def test_all_blank_segments_warn_then_error(tmp_path):
    from engine.models.asr import NoUsableTranscriptError
    with pytest.warns(UserWarning, match="blank ASR segment"):
        with pytest.raises(NoUsableTranscriptError):
            transcribe(tmp_path, [{"start": 0, "end": 1, "text": "\t "}])


@pytest.mark.parametrize("segment", [{"start": 1, "end": 0, "text": " "},
                                     {"start": 0, "end": 1, "text": None}])
def test_malformed_segments_are_not_skipped_or_coerced(tmp_path, segment):
    with pytest.raises(ValueError, match="invalid ASR segment"):
        transcribe(tmp_path, [segment])
