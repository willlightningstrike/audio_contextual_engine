import hashlib
import io
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from engine.cancel import CancellationToken, JobCancelled
from scripts import probe_yamnet_music as probe


def wav(tmp_path, samples=80000, *, channels=1, rate=16000, width=2):
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * samples * channels * width)
    return path


@pytest.fixture(autouse=True)
def assets(monkeypatch, tmp_path):
    for name in ("weights.npz", "frontend.npz", "LICENSE"):
        (tmp_path / name).write_bytes(b"explicit fake asset")
    monkeypatch.setattr(probe, "_snapshot_path", lambda: tmp_path)
    monkeypatch.setattr(probe, "_verify_snapshot", lambda path, hashes: path)
    monkeypatch.setattr(probe, "class_map_metadata", lambda: {
        "verified": False, "error": "offline", "names": None,
    })


def run(path, tmp_path, score_fn=lambda model, audio: np.zeros((1, 521)), token=None):
    return probe.probe_yamnet(path, out=tmp_path / "scores.json",
                              token=token or CancellationToken(),
                              load_fn=lambda root: object(), score_fn=score_fn)


def test_complete_track_scores_include_exact_tail_and_bounded_reads(tmp_path):
    lengths = []
    def score(model, audio):
        assert audio.dtype == np.float32
        lengths.append(len(audio))
        return np.full((2, 521), .25)
    report = run(wav(tmp_path, 976000), tmp_path, score)
    assert lengths == [80000] * 12 + [16000]
    assert report["status"] == "complete"
    assert report["progress"] == {"completed_units": 13, "expected_units": 13}
    assert [(r["start_seconds"], r["end_seconds"]) for r in report["records"]] == [
        (0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
        (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60), (60, 61)]
    assert report["records"][-1]["end_sample"] == 976000
    assert report["records"][0]["scores"] == [[.25] * 521] * 2
    assert json.loads((tmp_path / "scores.json").read_text()) == report


def test_one_sample_tail_is_not_lost(tmp_path):
    report = run(wav(tmp_path, 480001), tmp_path)
    assert report["records"][-1]["start_sample"] == 480000
    assert report["records"][-1]["end_sample"] == 480001
    assert len(report["records"]) == 7


def test_cancelled_returned_unit_is_retained_and_no_next_call(tmp_path):
    token = CancellationToken()
    calls = []
    def score(model, audio):
        calls.append(len(audio))
        if len(calls) == 2:
            token.cancel()
        return np.zeros((1, 521))
    with pytest.raises(JobCancelled):
        run(wav(tmp_path, 240000), tmp_path, score, token)
    report = json.loads((tmp_path / "scores.json").read_text())
    assert calls == [80000, 80000]
    assert report["status"] == "cancelled"
    assert report["progress"]["completed_units"] == 2
    assert len(report["records"]) == 2


@pytest.mark.parametrize("scores, match", [
    (np.zeros((1, 520)), "521"), (np.zeros((0, 521)), "521"),
    (np.zeros(521), "521"), (np.full((1, 521), np.nan), "invalid"),
    (np.full((1, 521), np.inf), "invalid"),
    (np.full((1, 521), -.1), "invalid"),
    (np.full((1, 521), 1.1), "invalid"),
])
def test_invalid_score_is_technical_failure_with_evidence(tmp_path, scores, match):
    with pytest.raises(ValueError, match=match):
        run(wav(tmp_path), tmp_path, lambda model, audio: scores)
    report = json.loads((tmp_path / "scores.json").read_text())
    assert report["status"] == "technical_error"
    assert report["progress"]["completed_units"] == 0
    assert report["errors"][0]["raw_scores"] is not None


def test_error_retains_prior_completed_scores(tmp_path):
    calls = []
    def score(model, audio):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("inference broke")
        return np.zeros((1, 521))
    with pytest.raises(RuntimeError, match="inference broke"):
        run(wav(tmp_path, 240000), tmp_path, score)
    report = json.loads((tmp_path / "scores.json").read_text())
    assert len(report["records"]) == 1
    assert report["status"] == "technical_error"


def test_existing_output_is_reserved_before_loading(tmp_path):
    path = wav(tmp_path)
    out = tmp_path / "scores.json"
    out.write_text("keep")
    def forbidden(root):
        pytest.fail("model loading before output reservation")
    with pytest.raises(FileExistsError):
        probe.probe_yamnet(path, out=out, token=CancellationToken(), load_fn=forbidden)
    assert out.read_text() == "keep"


def test_missing_audio_fails_before_loading(tmp_path):
    def forbidden(root):
        pytest.fail("model loading before audio validation")
    with pytest.raises(FileNotFoundError):
        probe.probe_yamnet(tmp_path / "absent.wav", out=tmp_path / "scores.json",
                           token=CancellationToken(), load_fn=forbidden)


@pytest.mark.parametrize("kwargs", [{"channels": 2}, {"rate": 8000}, {"width": 1}, {"samples": 0}])
def test_non_normalized_audio_rejected_before_load(tmp_path, kwargs):
    def forbidden(root):
        pytest.fail("model loading before format validation")
    with pytest.raises(ValueError, match="PCM16|nonempty"):
        probe.probe_yamnet(wav(tmp_path, **kwargs), out=tmp_path / "scores.json",
                           token=CancellationToken(), load_fn=forbidden)


def test_truncated_pcm_is_failure_with_interval_evidence(tmp_path):
    path = wav(tmp_path)
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="short PCM read"):
        run(path, tmp_path)
    report = json.loads((tmp_path / "scores.json").read_text())
    assert report["status"] == "technical_error"
    assert report["errors"][0]["interval"]["end_sample"] == 80000
    assert report["records"] == []


def test_exact_pcm_seek_and_normalization(tmp_path):
    path = wav(tmp_path, 80001)
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        samples = np.zeros(80001, dtype="<i2")
        samples[0], samples[80000] = -32768, 16384
        stream.writeframes(samples.tobytes())
    starts = []
    def score(model, audio):
        starts.append(float(audio[0]))
        return np.zeros((1, 521))
    report = run(path, tmp_path, score)
    assert starts == [-1.0, .5]
    assert report["records"][-1]["end_sample"] == 80001


def test_asset_mismatch_is_preserved_before_model_load(tmp_path, monkeypatch):
    from engine.models.vocals import _verify_snapshot
    monkeypatch.setattr(probe, "_verify_snapshot", _verify_snapshot)
    def forbidden(root):
        pytest.fail("loaded mismatched assets")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        probe.probe_yamnet(wav(tmp_path), out=tmp_path / "scores.json",
                           token=CancellationToken(), load_fn=forbidden)
    assert json.loads((tmp_path / "scores.json").read_text())["status"] == "technical_error"


def test_metadata_fetch_preserves_pinned_native_class_map(tmp_path, monkeypatch):
    # Restore the real metadata function; replace only its network boundary.
    monkeypatch.undo()
    monkeypatch.setattr(probe, "CLASS_MAP_DIR", tmp_path / "metadata")
    raw = ("index,mid,display_name\n" + "".join(
        f"{index},/m/{index},Native {index}\n" for index in range(521))).encode()
    urls = []
    def fetch(url, timeout):
        urls.append(url)
        return io.BytesIO(raw)
    monkeypatch.setattr(probe, "urlopen", fetch)
    metadata = probe.class_map_metadata()
    assert metadata["verified"] is True
    assert metadata["names"][24] == "Native 24"
    assert probe.YAMNET_REVISION in urls[0]
    assert metadata["license"] == "Apache-2.0"
    assert Path(metadata["local_path"]).read_bytes() == raw
    assert metadata["sha256"] == hashlib.sha256(raw).hexdigest()


def test_unavailable_class_names_preserve_numeric_diagnostics(tmp_path, monkeypatch):
    monkeypatch.undo()
    def unavailable(url, timeout):
        raise OSError("offline")
    monkeypatch.setattr(probe, "urlopen", unavailable)
    metadata = probe.class_map_metadata()
    assert metadata["verified"] is False
    assert metadata["names"] is None
    assert "offline" in metadata["error"]
