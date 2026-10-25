from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from engine.cancel import CancellationToken
from engine.domain import ContainerKind
from engine.ingest.probe import (
    InvalidProbe,
    NoAudioStream,
    RequestedStreamInvalid,
    choose_audio_stream,
    parse_probe,
    probe_source,
)


FIXTURES = Path(__file__).parent / "fixtures" / "ffprobe"


def fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def test_parse_video_selects_default_audio_and_retains_audio_metadata():
    probe = parse_probe(fixture("video_two_audio.json"))

    assert probe.container_kind is ContainerKind.VIDEO
    assert probe.selected_audio_stream == 2
    assert [stream.index for stream in probe.audio_streams] == [1, 2]
    assert probe.audio_streams[1].language == "es"
    assert probe.duration_seconds == 12.5


def test_parse_audio_only_container():
    probe = parse_probe(fixture("audio_only.json"))

    assert probe.container_kind is ContainerKind.AUDIO
    assert probe.selected_audio_stream == 0


def test_attached_cover_art_does_not_make_audio_a_video_container():
    data = fixture("audio_only.json")
    data["streams"].append({
        "index": 1, "codec_type": "video", "codec_name": "mjpeg",
        "disposition": {"attached_pic": 1},
    })

    probe = parse_probe(data)

    assert probe.container_kind is ContainerKind.AUDIO
    assert probe.selected_audio_stream == 0


@pytest.mark.parametrize("artwork_first", [False, True])
def test_real_video_with_cover_art_remains_video(artwork_first: bool):
    data = fixture("video_two_audio.json")
    artwork = {
        "index": 3, "codec_type": "video", "codec_name": "mjpeg",
        "disposition": {"attached_pic": 1},
    }
    data["streams"].insert(0 if artwork_first else len(data["streams"]), artwork)

    assert parse_probe(data).container_kind is ContainerKind.VIDEO


def test_no_audio_is_rejected_when_selecting():
    with pytest.raises(NoAudioStream):
        choose_audio_stream(parse_probe(fixture("silent_video.json")), None)


@pytest.mark.parametrize("requested", [0, 99, True, False, -1, 1.0])
def test_requested_video_or_missing_stream_is_rejected(requested: int):
    probe = parse_probe(fixture("video_two_audio.json"))

    with pytest.raises(RequestedStreamInvalid):
        choose_audio_stream(probe, requested)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format.duration", True),
        ("stream.channels", False),
        ("stream.sample_rate", True),
        ("stream.channels", 1.5),
        ("stream.sample_rate", "16000.5"),
    ],
)
def test_boolean_and_lossy_integer_probe_metadata_is_rejected(field: str, value: object):
    data = fixture("audio_only.json")
    if field == "format.duration":
        data["format"]["duration"] = value  # type: ignore[index]
    else:
        data["streams"][0][field.removeprefix("stream.")] = value  # type: ignore[index]

    with pytest.raises(InvalidProbe):
        parse_probe(data)


def test_first_audio_is_used_without_default():
    data = fixture("video_two_audio.json")
    for stream in data["streams"]:  # type: ignore[index]
        stream.get("disposition", {})["default"] = 0  # type: ignore[union-attr]

    assert choose_audio_stream(parse_probe(data), None).index == 1


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"streams": []},
        {"format": {"duration": "not-a-duration"}, "streams": []},
    ],
)
def test_malformed_probe_or_duration_is_rejected(data: dict[str, object]):
    with pytest.raises(InvalidProbe):
        parse_probe(data)


def test_probe_command_is_argv_and_preserves_spaces():
    calls: list[tuple[list[str], bool]] = []

    def runner(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs["shell"]))
        return type("Result", (), {"stdout": json.dumps(fixture("audio_only.json"))})()

    result = probe_source(Path("input with spaces.mp3"), runner=runner, ffprobe="probe-bin")

    assert result.selected_audio_stream == 0
    assert calls == [(
        ["probe-bin", "-v", "error", "-show_streams", "-show_format", "-of", "json", "input with spaces.mp3"],
        False,
    )]


@pytest.mark.parametrize("index", [True, False])
def test_boolean_audio_stream_index_is_rejected(index: bool):
    data = fixture("audio_only.json")
    data["streams"][0]["index"] = index  # type: ignore[index]

    with pytest.raises(InvalidProbe):
        parse_probe(data)


def test_duplicate_audio_stream_indexes_are_rejected():
    data = fixture("video_two_audio.json")
    data["streams"][2]["index"] = 1  # type: ignore[index]

    with pytest.raises(InvalidProbe, match="unique"):
        parse_probe(data)


@pytest.mark.parametrize("stdout", [None, b"\xff", object()])
def test_missing_or_undecodable_probe_stdout_is_invalid(stdout: object):
    def runner(command: list[str], **kwargs: object) -> object:
        return type("Result", (), {"stdout": stdout})()

    with pytest.raises(InvalidProbe):
        probe_source(Path("input.mp3"), runner=runner)


def test_cancellable_probe_rejects_valid_json_when_ffprobe_exits_nonzero():
    class Process:
        returncode = 1

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return (json.dumps(fixture("audio_only.json")), "diagnostic")

    with pytest.raises(InvalidProbe, match="exit"):
        probe_source(Path("input.mp3"), token=CancellationToken(), popen_factory=lambda *args, **kwargs: Process())


@pytest.mark.parametrize("failure", [OSError("read error"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")])
def test_cancellable_probe_failure_cleans_up_then_normalizes_to_invalid_probe(failure: BaseException):
    class Process:
        returncode: int | None = None

        def __init__(self) -> None:
            self.communicates = 0
            self.terminated = False
            self.killed = False

        def poll(self) -> None:
            return None

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.communicates += 1
            if self.communicates == 1:
                raise failure
            return ("", "")

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    process = Process()
    with pytest.raises(InvalidProbe):
        probe_source(Path("input.mp3"), token=CancellationToken(), popen_factory=lambda *args, **kwargs: process)
    assert process.terminated
    assert process.communicates == 2


def test_probe_cleanup_reaps_after_a_second_communicate_error():
    class Process:
        returncode: int | None = None

        def __init__(self) -> None:
            self.communicates = 0
            self.terminated = False
            self.killed = False
            self.waits = 0

        def poll(self) -> None:
            return None

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.communicates += 1
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("ffprobe", timeout)
            return 0

    process = Process()
    with pytest.raises(InvalidProbe):
        probe_source(Path("input.mp3"), token=CancellationToken(), popen_factory=lambda *args, **kwargs: process)
    assert process.terminated
    assert process.killed
    assert process.waits == 2


def test_missing_container_duration_is_unknown_not_invalid():
    data = fixture("audio_only.json")
    del data["format"]["duration"]  # type: ignore[union-attr]

    probe = parse_probe(data)

    assert probe.duration_seconds is None
    assert probe.source_metadata.duration_seconds is None
