from __future__ import annotations

import hashlib
import math
import sys
from types import ModuleType, SimpleNamespace
import wave
from pathlib import Path

import numpy as np
import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.models.vocals import MlxYamnetAdapter, admitted


def _write_pcm16_mono(path: Path, *, seconds: float) -> None:
    frames = round(seconds * 16_000)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\0\0" * frames)


def _snapshot(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    root = (tmp_path / "yamnet").resolve()
    root.mkdir()
    hashes: dict[str, str] = {}
    for name, contents in (("weights.npz", b"weights"), ("frontend.npz", b"frontend")):
        (root / name).write_bytes(contents)
        hashes[name] = hashlib.sha256(contents).hexdigest()
    return root, hashes


def test_admission_is_not_proof_of_absence():
    assert admitted(0.6, 0.0) is True
    assert admitted(0.0, 0.02) is True
    assert admitted(0.0, 0.0) is False


def test_admission_rejects_non_finite_or_out_of_range_scores():
    for speech, singing in (
        (math.nan, 0),
        (0, math.inf),
        (-0.1, 0),
        (0, 1.1),
        ("0.6", 0),
        (None, 0),
    ):
        with pytest.raises(ValueError, match="finite scores"):
            admitted(speech, singing)


def test_screening_checks_every_five_second_subwindow_and_uses_global_class_maxima(
    tmp_path: Path,
):
    snapshot, hashes = _snapshot(tmp_path)
    audio = tmp_path / "twelve-seconds.wav"
    _write_pcm16_mono(audio, seconds=12)
    sample_counts: list[int] = []
    scores = iter(
        (
            np.array([[0.1] + [0.0] * 23 + [0.001]], dtype=np.float32),
            np.array([[0.2] + [0.0] * 23 + [0.02]], dtype=np.float32),
            np.array([[0.6] + [0.0] * 23 + [0.003]], dtype=np.float32),
        )
    )

    def score(_model: object, waveform: np.ndarray) -> np.ndarray:
        sample_counts.append(len(waveform))
        return next(scores)

    adapter = MlxYamnetAdapter(
        snapshot,
        expected_hashes=hashes,
        load_fn=lambda _path: object(),
        score_fn=score,
    )

    result = adapter.screen(audio, CancellationToken())

    assert sample_counts == [80_000, 80_000, 32_000]
    assert result.speech_max == pytest.approx(0.6)
    assert result.singing_max == pytest.approx(0.02)
    assert result.admitted is True


def test_short_nonempty_input_is_screened_once(tmp_path: Path):
    snapshot, hashes = _snapshot(tmp_path)
    audio = tmp_path / "short.wav"
    _write_pcm16_mono(audio, seconds=0.25)
    lengths: list[int] = []

    def score(_model: object, waveform: np.ndarray) -> np.ndarray:
        lengths.append(len(waveform))
        return np.zeros((1, 25), dtype=np.float32)

    result = MlxYamnetAdapter(
        snapshot,
        expected_hashes=hashes,
        load_fn=lambda _path: object(),
        score_fn=score,
    ).screen(audio, CancellationToken())

    assert lengths == [4_000]
    assert result.admitted is False


@pytest.mark.parametrize("seconds", [0, 31])
def test_empty_or_unbounded_input_is_rejected_before_model_load(tmp_path: Path, seconds: float):
    snapshot, hashes = _snapshot(tmp_path)
    audio = tmp_path / "invalid-duration.wav"
    _write_pcm16_mono(audio, seconds=seconds)
    loads = 0

    def load(_path: Path) -> object:
        nonlocal loads
        loads += 1
        return object()

    adapter = MlxYamnetAdapter(snapshot, expected_hashes=hashes, load_fn=load)

    with pytest.raises(ValueError, match="nonempty|30 seconds"):
        adapter.screen(audio, CancellationToken())

    assert loads == 0


@pytest.mark.parametrize(
    "scores",
    [
        np.empty((0, 25), dtype=np.float32),
        np.zeros((1, 24), dtype=np.float32),
        np.array([[math.nan] + [0.0] * 24], dtype=np.float32),
    ],
)
def test_invalid_model_scores_are_rejected(tmp_path: Path, scores: np.ndarray):
    snapshot, hashes = _snapshot(tmp_path)
    audio = tmp_path / "audio.wav"
    _write_pcm16_mono(audio, seconds=1)
    adapter = MlxYamnetAdapter(
        snapshot,
        expected_hashes=hashes,
        load_fn=lambda _path: object(),
        score_fn=lambda _model, _waveform: scores,
    )

    with pytest.raises(ValueError, match="scores"):
        adapter.screen(audio, CancellationToken())


def test_cancellation_between_subwindows_stops_further_model_calls(tmp_path: Path):
    snapshot, hashes = _snapshot(tmp_path)
    audio = tmp_path / "audio.wav"
    _write_pcm16_mono(audio, seconds=12)
    token = CancellationToken()
    calls = 0

    def score(_model: object, _waveform: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        token.cancel()
        return np.zeros((1, 25), dtype=np.float32)

    adapter = MlxYamnetAdapter(
        snapshot,
        expected_hashes=hashes,
        load_fn=lambda _path: object(),
        score_fn=score,
    )

    with pytest.raises(JobCancelled):
        adapter.screen(audio, token)

    assert calls == 1


def test_snapshot_hash_mismatch_is_rejected_before_load(tmp_path: Path):
    snapshot, hashes = _snapshot(tmp_path)
    hashes["weights.npz"] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        MlxYamnetAdapter(snapshot, expected_hashes=hashes, load_fn=lambda _path: object())


def test_parity_evidence_records_process_rss_and_system_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from scripts import verify_vocals

    fixture = tmp_path / "only.npz"
    waveform = np.zeros(1, dtype=np.float32)
    expected = np.zeros((1, 1), dtype=np.float32)
    np.savez(
        fixture,
        waveform=waveform,
        scores=expected,
        embeddings=expected,
        logmel=expected,
    )
    fake_core = ModuleType("mlx.core")
    fake_core.array = np.asarray
    fake_core.eval = lambda *_values: None
    fake_core.get_peak_memory = lambda: 123
    fake_mlx = ModuleType("mlx")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setattr(verify_vocals, "EXPECTED_FIXTURES", 1)
    monkeypatch.setattr(
        verify_vocals,
        "MlxYamnetAdapter",
        lambda _model_dir: SimpleNamespace(
            model=lambda _waveform: (expected, expected, expected)
        ),
    )
    swaps = iter((1_000, 1_200))
    monkeypatch.setattr(verify_vocals, "_swap_bytes", lambda: next(swaps))
    monkeypatch.setattr(
        verify_vocals.resource,
        "getrusage",
        lambda _who: SimpleNamespace(ru_maxrss=456),
    )

    result = verify_vocals.verify(tmp_path, tmp_path)

    assert result["peak_mlx_bytes"] == 123
    assert result["peak_rss_bytes"] == 456
    assert result["system_swap_before_bytes"] == 1_000
    assert result["system_swap_after_bytes"] == 1_200
    assert result["system_swap_delta_bytes"] == 200
    assert result["rss_ceiling_bytes"] == 68_719_476_736
    assert result["process_rss_ceiling_margin_bytes"] == 68_719_476_280
    assert "rss_headroom_bytes" not in result
