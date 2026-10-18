"""Parity-preserving MLX YAMNet adapter for bounded vocal eligibility screening."""

from __future__ import annotations

import hashlib
import math
import wave
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from engine.cancel import CancellationToken
from engine.models import require_resolved_local_directory
from engine.transcribe.types import ScreenResult


SPEECH_THRESHOLD = 0.5102361757308245
SINGING_THRESHOLD = 0.011171962483786047
SPEECH_INDEX = 0
SINGING_INDEX = 24
SAMPLE_RATE = 16_000
SCREEN_SECONDS = 5
MAX_AUDIO_SECONDS = 30
EXPECTED_ASSET_HASHES = {
    "weights.npz": "8dacef0002f1b4eec29af9f40c0e2dc76fe76dde38e8572dab06d4ef94d9acdb",
    "frontend.npz": "f9bde8b7c1f30f0d733278aab88bda6431d3c610a423af8677a9c2ac4cb83601",
}


def _score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("vocal admission requires finite scores between 0 and 1")
    converted = float(value)
    if not math.isfinite(converted) or not 0 <= converted <= 1:
        raise ValueError("vocal admission requires finite scores between 0 and 1")
    return converted


def admitted(speech_max: float, singing_max: float) -> bool:
    """Apply the frozen eligibility rule; false means uncertain, not absent."""
    speech = _score(speech_max)
    singing = _score(singing_max)
    return speech >= SPEECH_THRESHOLD or singing >= SINGING_THRESHOLD


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_snapshot(path: Path, expected_hashes: Mapping[str, str]) -> Path:
    root = require_resolved_local_directory(path)
    for name, expected in expected_hashes.items():
        asset = root / name
        if asset.is_symlink() or not asset.is_file() or asset.resolve(strict=True).parent != root:
            raise ValueError(f"YAMNet asset is missing or unsafe: {name}")
        if _sha256(asset) != expected:
            raise ValueError(f"YAMNet asset SHA-256 mismatch: {name}")
    return root


class _MlxYamnet:
    """Reviewed port of official TensorFlow Models YAMNet (Apache-2.0)."""

    def __init__(self, root: Path, mx: Any, np: Any) -> None:
        self._mx = mx
        with np.load(root / "weights.npz", allow_pickle=False) as archive:
            self._weights = {key: mx.array(archive[key]) for key in archive.files}
        with np.load(root / "frontend.npz", allow_pickle=False) as archive:
            self._mel = mx.array(archive["mel"])
            self._hann = mx.array(archive["hann"])
        self._used: set[str] = set()

    def _weight(self, name: str) -> Any:
        self._used.add(name)
        return self._weights[name]

    def _conv(self, x: Any, name: str, stride: int = 1, depthwise: bool = False) -> Any:
        mx = self._mx
        key = "depthwise_kernel:0" if depthwise else "kernel:0"
        kernel = self._weight(f"{name}/{key}")
        kernel = mx.transpose(kernel, (2, 0, 1, 3) if depthwise else (3, 0, 1, 2))
        padding: list[tuple[int, int]] = []
        for size, width in zip(x.shape[1:3], kernel.shape[1:3], strict=True):
            total = max(0, ((size + stride - 1) // stride - 1) * stride + width - size)
            padding.append((total // 2, total - total // 2))
        x = mx.pad(x, [(0, 0), *padding, (0, 0)])
        x = mx.conv2d(x, kernel, stride=stride, groups=x.shape[-1] if depthwise else 1)
        beta = self._weight(f"{name}/bn/beta:0")
        mean = self._weight(f"{name}/bn/moving_mean:0")
        variance = self._weight(f"{name}/bn/moving_variance:0")
        return mx.maximum((x - mean) * mx.rsqrt(variance + 1e-4) + beta, 0)

    def __call__(self, waveform: Any) -> tuple[Any, Any, Any]:
        mx = self._mx
        if waveform.ndim != 1:
            raise ValueError("expected a 16 kHz mono waveform")
        sample_count = waveform.shape[0]
        padded = 15_600 + ((max(sample_count, 15_600) - 15_600 + 7_679) // 7_680) * 7_680
        waveform = mx.pad(waveform, [(0, padded - sample_count)])
        frame_indices = mx.arange((padded - 400) // 160 + 1)[:, None] * 160 + mx.arange(400)
        frames = waveform[frame_indices] * self._hann
        # The frozen parity gate requires CPU FFT. Convolution remains on Metal.
        with mx.stream(mx.cpu):
            magnitude = mx.abs(mx.fft.rfft(frames, n=512, axis=-1))
        logmel = mx.log(magnitude @ self._mel + 0.001)
        patch_indices = (
            mx.arange((logmel.shape[0] - 96) // 48 + 1)[:, None] * 48 + mx.arange(96)
        )
        x = logmel[patch_indices][..., None]
        x = self._conv(x, "layer1/conv", 2)
        for layer in range(2, 15):
            stride = 2 if layer in (3, 5, 7, 13) else 1
            x = self._conv(x, f"layer{layer}/depthwise_conv", stride, True)
            x = self._conv(x, f"layer{layer}/pointwise_conv")
        embedding = mx.mean(x, axis=(1, 2))
        scores = mx.sigmoid(
            embedding @ self._weight("dense/kernel:0") + self._weight("dense/bias:0")
        )
        if self._used != set(self._weights):
            raise ValueError("YAMNet checkpoint contains unused parameters")
        return scores, embedding, logmel


def _default_load(path: Path) -> _MlxYamnet:
    import mlx.core as mx
    import numpy as np

    return _MlxYamnet(path, mx, np)


def _default_score(model: _MlxYamnet, waveform: Any) -> Any:
    import mlx.core as mx
    import numpy as np

    scores, _, _ = model(mx.array(waveform))
    mx.eval(scores)
    return np.asarray(scores)


class MlxYamnetAdapter:
    """Screen every five-second subwindow and return global Speech/Singing maxima."""

    def __init__(
        self,
        snapshot_path: Path,
        *,
        expected_hashes: Mapping[str, str] = EXPECTED_ASSET_HASHES,
        load_fn: Callable[[Path], Any] = _default_load,
        score_fn: Callable[[Any, Any], Any] = _default_score,
    ) -> None:
        self._snapshot_path = _verify_snapshot(snapshot_path, expected_hashes)
        self._load_fn = load_fn
        self._score_fn = score_fn
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_fn(self._snapshot_path)
        return self._model

    def screen(self, audio: Path, token: CancellationToken) -> ScreenResult:
        import numpy as np

        token.raise_if_cancelled()
        path = Path(audio).resolve(strict=True)
        try:
            with wave.open(str(path), "rb") as stream:
                if (
                    stream.getnchannels() != 1
                    or stream.getsampwidth() != 2
                    or stream.getframerate() != SAMPLE_RATE
                    or stream.getcomptype() != "NONE"
                ):
                    raise ValueError("vocal screening requires a mono 16 kHz PCM16 WAV")
                frame_count = stream.getnframes()
                if frame_count == 0:
                    raise ValueError("vocal screening requires a nonempty WAV")
                if frame_count > MAX_AUDIO_SECONDS * SAMPLE_RATE:
                    raise ValueError("vocal screening input must not exceed 30 seconds")
                waveform = np.frombuffer(stream.readframes(frame_count), dtype="<i2").astype(
                    np.float32
                ) / 32768.0
        except wave.Error as exc:
            raise ValueError("vocal screening requires a readable WAV") from exc

        speech_max = 0.0
        singing_max = 0.0
        subwindow_samples = SCREEN_SECONDS * SAMPLE_RATE
        for start in range(0, frame_count, subwindow_samples):
            token.raise_if_cancelled()
            scores = np.asarray(
                self._score_fn(self.model, waveform[start : start + subwindow_samples])
            )
            token.raise_if_cancelled()
            if (
                scores.ndim != 2
                or scores.shape[0] == 0
                or scores.shape[1] <= SINGING_INDEX
                or not np.isfinite(scores).all()
                or (scores < 0).any()
                or (scores > 1).any()
            ):
                raise ValueError("YAMNet returned invalid scores")
            speech_max = max(speech_max, float(scores[:, SPEECH_INDEX].max()))
            singing_max = max(singing_max, float(scores[:, SINGING_INDEX].max()))

        return ScreenResult(
            speech_max=speech_max,
            singing_max=singing_max,
            admitted=admitted(speech_max, singing_max),
        )

    def release(self) -> None:
        loaded = self._model is not None
        self._model = None
        if loaded:
            from mlx import core as mx

            mx.clear_cache()
