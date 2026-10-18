import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from engine.domain import TimeRange, TranscriptSegment


def test_cancellation_probe_measures_request_to_stop_without_retry():
    from scripts.verify_asr_recovery import cancellation_probe

    class Timer:
        def __init__(self, _delay, callback):
            self.callback = callback

        def start(self):
            self.callback()

        def cancel(self):
            pass

        def join(self):
            pass

    class Asr:
        calls = 0

        def transcribe_window(self, audio, window, language, token):
            self.calls += 1
            token.raise_if_cancelled()

    timestamps = iter((10.0, 12.5))
    asr = Asr()

    result = cancellation_probe(
        asr,
        Path("unused.wav"),
        30.0,
        timer_factory=Timer,
        monotonic=lambda: next(timestamps),
    )

    assert result == {
        "outcome": "cancelled",
        "request_to_stop_seconds": 2.5,
        "target_seconds": 35.0,
        "passed": True,
        "attempt_count": 1,
    }
    assert asr.calls == 1


def test_probe_preserves_overrun_as_failed_evidence():
    from scripts.verify_asr_recovery import probe_window

    class Asr:
        def detect_language_probabilities(self, audio, window, token):
            return {"en": 0.99}

        def transcribe_window(self, audio, window, language, token):
            return [TranscriptSegment(time_range=TimeRange(start_seconds=27.2, end_seconds=32),
                                      text="example", confidence=1, language="en")]

    result = probe_window(Path("unused.wav"), 30.0, Asr())
    assert result["status"] == "failed"
    assert result["failure_stage"] == "timestamp_validation"
    assert result["segments"][0]["time_range"]["end_seconds"] == 32.0


def test_probe_never_transcribes_non_english():
    from scripts.verify_asr_recovery import probe_window

    class Asr:
        def detect_language_probabilities(self, audio, window, token):
            return {"zh": 0.99}

        def transcribe_window(self, *args):
            raise AssertionError("must not transcribe")

    result = probe_window(Path("unused.wav"), 30.0, Asr())
    assert result["status"] == "ineligible"
    assert "segments" not in result


def test_runtime_identity_uses_metadata_without_importing_model_packages(
    tmp_path, monkeypatch
):
    from scripts import verify_asr_recovery

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "models": {
                    "asr": {"repository": "example/asr", "revision": "asr-rev"},
                    "embeddings": {
                        "repository": "example/embed",
                        "revision": "embed-rev",
                    },
                }
            }
        )
    )
    code = tmp_path / "probe.py"
    code.write_text("print('probe')\n")
    adapter = tmp_path / "asr.py"
    adapter.write_text("# adapter\n")
    requested_packages = []

    def version(name):
        requested_packages.append(name)
        return f"{name}-version"

    def run(command, **_kwargs):
        if command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        if command[-2:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout=" M file.py\n")
        raise AssertionError(command)

    identity = verify_asr_recovery.runtime_identity(
        tmp_path,
        manifest,
        code_path=code,
        adapter_path=adapter,
        version_fn=version,
        run_fn=run,
    )

    assert identity["python"]["implementation"]
    assert identity["python"]["version"]
    assert identity["platform"]
    assert identity["packages"] == {
        name: f"{name}-version" for name in requested_packages
    }
    assert identity["git"] == {"head": "abc123", "dirty": True}
    assert identity["code_sha256"] == hashlib.sha256(code.read_bytes()).hexdigest()
    assert identity["adapter_sha256"] == hashlib.sha256(
        adapter.read_bytes()
    ).hexdigest()
    assert identity["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert identity["model_revisions"] == {
        "asr": {"repository": "example/asr", "revision": "asr-rev"},
        "embeddings": {"repository": "example/embed", "revision": "embed-rev"},
    }


def test_protocol_is_written_with_runtime_identity_before_inference(tmp_path):
    from scripts.verify_asr_recovery import initialize_protocol

    output = tmp_path / "run"
    identity = {"git": {"head": "abc123", "dirty": False}}
    protocol = initialize_protocol(output, identity)

    assert output.is_dir()
    assert json.loads((output / "protocol.json").read_text()) == protocol
    assert protocol["runtime_identity"] == identity
    assert protocol["status"] == "initialized_before_inference"
    assert protocol["mode"] == "production_word_timestamps_tail_filter"


def test_final_summary_repeats_protocol_identity_and_hashes(tmp_path):
    from scripts.verify_asr_recovery import build_summary

    source_protocol = tmp_path / "source-protocol.json"
    source_protocol.write_text('{"source": true}\n')
    run_protocol = tmp_path / "run-protocol.json"
    run_protocol.write_text('{"runtime_identity": {"git": "abc123"}}\n')
    identity = {"git": {"head": "abc123", "dirty": False}}

    summary = build_summary(
        [{"status": "failed"}],
        {"peak_rss_bytes": 123},
        source_protocol,
        run_protocol,
        identity,
    )

    assert summary["runtime_identity"] is identity
    assert summary["mode"] == "production_word_timestamps_tail_filter"
    assert summary["protocol_sha256"] == hashlib.sha256(
        source_protocol.read_bytes()
    ).hexdigest()
    assert summary["run_protocol_sha256"] == hashlib.sha256(
        run_protocol.read_bytes()
    ).hexdigest()
