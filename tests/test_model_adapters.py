from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange


def _load_script(name: str):
    import importlib.util

    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True)
    return path.resolve()


def _model_manifest(tmp_path: Path) -> Path:
    from engine.models.registry import MODEL_SPECS

    models: dict[str, dict[str, object]] = {}
    for role, spec in MODEL_SPECS.items():
        directory = _snapshot(tmp_path, role)
        payload = f"{role}-weights".encode()
        (directory / "model.safetensors").write_bytes(payload)
        models[role] = {
            **spec,
            "local_path": str(directory),
            "files": [{"path": "model.safetensors", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}],
        }
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": models}))
    return manifest


def _wav(tmp_path: Path, seconds: float = 1.0) -> Path:
    path = tmp_path / "window.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path.resolve()


def test_registry_loads_each_adapter_once_and_shares_text_audio_model(tmp_path: Path):
    from engine.models.registry import ModelPaths, ModelRegistry

    calls = {"asr": 0, "embedder": 0, "text_audio": 0}
    paths = ModelPaths(
        asr=_snapshot(tmp_path, "asr"),
        embeddings=_snapshot(tmp_path, "embeddings"),
        text_audio=_snapshot(tmp_path, "text_audio"),
    )

    def fake_asr(path: Path):
        assert path == paths.asr
        calls["asr"] += 1
        return object()

    def fake_embedder(path: Path):
        assert path == paths.embeddings
        calls["embedder"] += 1
        return object()

    def fake_text_audio(path: Path):
        assert path == paths.text_audio
        calls["text_audio"] += 1
        return object()

    registry = ModelRegistry(
        paths,
        load_asr=fake_asr,
        load_embedder=fake_embedder,
        load_text_audio=fake_text_audio,
    )

    assert registry.asr is registry.asr
    assert registry.embedder is registry.embedder
    assert registry.text_generator is registry.audio_understanding
    assert calls == {"asr": 1, "embedder": 1, "text_audio": 1}


def test_registry_rejects_repo_ids_relative_paths_and_symlink_dirs(tmp_path: Path):
    from engine.models.registry import ModelPaths

    snapshot = _snapshot(tmp_path, "snapshot")
    link = tmp_path / "link"
    link.symlink_to(snapshot, target_is_directory=True)

    with pytest.raises(ValueError, match="resolved local directory"):
        ModelPaths(asr=Path("mlx-community/model"), embeddings=snapshot, text_audio=snapshot)
    with pytest.raises(ValueError, match="resolved local directory"):
        ModelPaths(asr=link.absolute(), embeddings=snapshot, text_audio=snapshot)


def test_fetch_models_downloads_exact_qwen2_audio_snapshot_under_generic_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fetch_models = _load_script("fetch_models")
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"]))
        destination.mkdir(parents=True)
        (destination / "config.json").write_text("{}")
        (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    manifest = fetch_models.fetch(tmp_path / "models", candidate="qwen2_audio")

    assert set(manifest["models"]) == {"asr", "embeddings", "text_audio"}
    assert calls[-1] == {
        "repo_id": "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
        "revision": "c65570002626f41b4dc08b7b54f42f99f3e82e7f",
        "local_dir": str((tmp_path / "models" / "text_audio").resolve()),
    }


def test_text_audio_default_loader_uses_public_mlx_audio_local_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from engine.models.text_audio import _default_load

    snapshot = _snapshot(tmp_path, "text_audio")
    loaded = object()
    calls: list[tuple[str, bool, bool]] = []

    def fake_load(path: str, *, lazy: bool, strict: bool):
        calls.append((path, lazy, strict))
        return loaded

    monkeypatch.setitem(sys.modules, "mlx_audio.stt", SimpleNamespace(load=fake_load))

    assert _default_load(snapshot) is loaded
    assert calls == [(str(snapshot), False, True)]


def test_asr_converts_validated_segments_and_checks_cancellation(tmp_path: Path):
    from engine.models.asr import MlxAsrAdapter

    snapshot = _snapshot(tmp_path, "asr")
    wav = _wav(tmp_path)
    generated = []

    class FakeModel:
        def generate(self, path: str, **kwargs: object):
            generated.append((path, kwargs))
            return {"segments": [{"start": 0.5, "end": 2.0, "text": "hello", "confidence": 0.8}]}

    adapter = MlxAsrAdapter(snapshot, load_fn=lambda path: FakeModel())
    result = adapter.transcribe_window(wav, TimeRange(start_seconds=10, end_seconds=20), "en", CancellationToken())

    assert [(item.time_range.start_seconds, item.time_range.end_seconds, item.text, item.language) for item in result] == [
        (0.5, 2.0, "hello", "en")
    ]
    assert generated == [(str(wav), {"language": "en", "task": "transcribe", "return_timestamps": True, "word_timestamps": True, "verbose": False})]

    token = CancellationToken()
    token.cancel()
    with pytest.raises(JobCancelled):
        adapter.transcribe_window(wav, TimeRange(start_seconds=0, end_seconds=1), "en", token)


def test_asr_rejects_unbounded_windows_and_malformed_upstream_output(tmp_path: Path):
    from engine.models.asr import MlxAsrAdapter

    snapshot = _snapshot(tmp_path, "asr")
    wav = _wav(tmp_path)
    adapter = MlxAsrAdapter(snapshot, load_fn=lambda path: type("Model", (), {"generate": lambda self, *_args, **_kwargs: {"segments": [{"start": 3, "end": 1, "text": "bad"}]}})())

    with pytest.raises(ValueError, match="30 seconds"):
        adapter.transcribe_window(wav, TimeRange(start_seconds=0, end_seconds=30.1), "en", CancellationToken())
    with pytest.raises(ValueError, match="invalid ASR segment"):
        adapter.transcribe_window(wav, TimeRange(start_seconds=0, end_seconds=30), "en", CancellationToken())


def test_asr_converts_language_probabilities_from_the_resident_model(tmp_path: Path):
    from engine.models.asr import MlxAsrAdapter

    snapshot = _snapshot(tmp_path, "asr")
    wav = _wav(tmp_path)
    seen: list[tuple[Path, int]] = []

    class FakeModel:
        dims = type("Dims", (), {"n_mels": 128})()

        def detect_language(self, mel):
            assert mel == "mel"
            return object(), {"en": 0.8, "ja": 0.2}

    adapter = MlxAsrAdapter(
        snapshot,
        load_fn=lambda path: FakeModel(),
        language_inputs_fn=lambda path, n_mels: seen.append((path, n_mels)) or "mel",
    )

    assert adapter.detect_language_probabilities(wav, TimeRange(start_seconds=0, end_seconds=30), CancellationToken()) == {
        "en": 0.8,
        "ja": 0.2,
    }
    assert seen == [(wav, 128)]


def test_embedder_returns_normalized_float32_vectors_from_mlx_tensor(tmp_path: Path):
    from engine.models.embeddings import MlxEmbeddingAdapter

    snapshot = _snapshot(tmp_path, "embeddings")

    class FakeMlxTensor:
        __module__ = "mlx.core"

        def __array__(self, dtype=None):
            return np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=dtype)

    class FakeEmbedder:
        def encode(self, texts):
            assert texts == ["first", "second"]
            return FakeMlxTensor()

    adapter = MlxEmbeddingAdapter(snapshot, load_fn=lambda path: FakeEmbedder())
    result = adapter.encode(["first", "second"])

    assert result.dtype == np.float32
    assert result.tolist() == [[0.6000000238418579, 0.800000011920929], [0.0, 1.0]]


def test_embedder_casts_mlx_bfloat16_before_numpy_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sys
    from types import SimpleNamespace

    from engine.models.embeddings import MlxEmbeddingAdapter

    casts: list[object] = []

    class FloatTensor:
        __module__ = "mlx.core"

        def __array__(self, dtype=None):
            return np.asarray([[3.0, 4.0]], dtype=dtype)

    class BfloatTensor:
        __module__ = "mlx.core"

        def astype(self, dtype):
            casts.append(dtype)
            return FloatTensor()

        def __array__(self, dtype=None):
            raise RuntimeError("NumPy cannot consume MLX bfloat16 directly")

    float32 = object()
    fake_core = SimpleNamespace(array=lambda value: value, eval=lambda value: None, float32=float32)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_core))
    model = lambda **tensors: SimpleNamespace(text_embeds=BfloatTensor())
    tokenizer = lambda texts, **kwargs: {"input_ids": np.asarray([[1]])}
    adapter = MlxEmbeddingAdapter(_snapshot(tmp_path, "bfloat-embeddings"), load_fn=lambda path: (model, tokenizer))

    result = adapter.encode(["context"])

    assert casts == [float32]
    assert result.dtype == np.float32
    assert result.tolist() == [[0.6000000238418579, 0.800000011920929]]


def test_text_audio_adapter_uses_resident_language_model_and_one_bounded_wav(tmp_path: Path):
    from engine.models.text_audio import MlxTextAudioAdapter

    snapshot = _snapshot(tmp_path, "text_audio")
    wav = _wav(tmp_path)
    text_calls: list[dict[str, object]] = []
    audio_calls: list[dict[str, object]] = []

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "first"}]
            assert tokenize is False
            assert add_generation_prompt is True
            return "formatted:first"

    class Model:
        language_model = object()
        _processor = type("Processor", (), {"tokenizer": Tokenizer()})()

        def generate(self, audio, **kwargs):
            audio_calls.append({"audio": audio, **kwargs})
            return type("Output", (), {"text": "audio-answer"})()

    model = Model()
    adapter = MlxTextAudioAdapter(
        snapshot,
        load_fn=lambda path: model,
        text_generate_fn=lambda language_model, tokenizer, prompt, **kwargs: text_calls.append(
            {"language_model": language_model, "tokenizer": tokenizer, "prompt": prompt, **kwargs}
        ) or "text-answer",
    )

    assert adapter.generate_text("first", max_tokens=16, token=CancellationToken()) == "text-answer"
    assert adapter.generate_audio(wav, TimeRange(start_seconds=0, end_seconds=30), "second", max_tokens=8, token=CancellationToken()) == "audio-answer"
    assert text_calls == [
        {
            "language_model": model.language_model,
            "tokenizer": model._processor.tokenizer,
            "prompt": "formatted:first",
            "max_tokens": 16,
            "temperature": 0.0,
            "verbose": False,
        }
    ]
    assert audio_calls == [
        {
            "audio": str(wav),
            "prompt": "second",
            "max_tokens": 8,
            "temperature": 0.0,
            "verbose": False,
        }
    ]
    with pytest.raises(ValueError, match="30 seconds"):
        adapter.generate_audio(wav, TimeRange(start_seconds=0, end_seconds=31), "too long", max_tokens=8, token=CancellationToken())


def test_audio_adapters_reject_a_wav_longer_than_the_bounded_window(tmp_path: Path):
    from engine.models.asr import MlxAsrAdapter
    from engine.models.text_audio import MlxTextAudioAdapter

    snapshot = _snapshot(tmp_path, "snapshot")
    long_wav = _wav(tmp_path, seconds=30.1)
    asr = MlxAsrAdapter(snapshot, load_fn=lambda path: object())
    text_audio = MlxTextAudioAdapter(snapshot, load_fn=lambda path: object())

    with pytest.raises(ValueError, match="30 seconds"):
        asr.transcribe_window(long_wav, TimeRange(start_seconds=0, end_seconds=30), "en", CancellationToken())
    with pytest.raises(ValueError, match="30 seconds"):
        text_audio.generate_audio(long_wav, TimeRange(start_seconds=0, end_seconds=30), "prompt", max_tokens=1, token=CancellationToken())


def test_verified_manifest_rejects_tampered_or_uninventoried_snapshot_files(tmp_path: Path):
    from engine.models.registry import load_verified_model_paths

    manifest = _model_manifest(tmp_path)
    paths = load_verified_model_paths(manifest)
    assert paths.asr.name == "asr"

    (paths.asr / "model.safetensors").write_bytes(b"xxxxxxxxxxx")
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_model_paths(manifest)

    manifest = _model_manifest(tmp_path / "fresh")
    asr = Path(json.loads(manifest.read_text())["models"]["asr"]["local_path"])
    (asr / "untracked.bin").write_bytes(b"extra")
    with pytest.raises(ValueError, match="unexpected"):
        load_verified_model_paths(manifest)


def test_verified_manifest_rejects_a_symlinked_manifest(tmp_path: Path):
    from engine.models.registry import load_verified_model_paths

    manifest = _model_manifest(tmp_path)
    linked = tmp_path / "linked-manifest.json"
    linked.symlink_to(manifest)

    with pytest.raises(ValueError, match="must not be a symlink"):
        load_verified_model_paths(linked)


def test_model_provenance_evidence_requires_mlx_parameter_and_tensor_types():
    from engine.models import collect_mlx_evidence

    class FakeMlxArray:
        __module__ = "mlx.core"

    class Model:
        def parameters(self):
            return {"layer": [FakeMlxArray()]}

    evidence = collect_mlx_evidence(Model(), {"embedding": FakeMlxArray(), "invalid": object()})

    assert evidence["parameter_leaf_count"] == 1
    assert evidence["parameter_leaves_are_mlx"] is True
    assert evidence["tensor_types"]["embedding"].startswith("mlx")
    assert evidence["tensor_types"]["invalid"] == "builtins.object"


def test_adapters_record_mlx_parameter_and_inference_tensor_provenance(tmp_path: Path):
    from engine.models.asr import MlxAsrAdapter
    from engine.models.embeddings import MlxEmbeddingAdapter

    class FakeMlxArray:
        __module__ = "mlx.core"

        def __array__(self, dtype=None):
            return np.asarray([[1.0, 0.0]], dtype=dtype)

    class AsrModel:
        dims = type("Dims", (), {"n_mels": 128})()

        def parameters(self):
            return {"weight": FakeMlxArray()}

        def detect_language(self, mel):
            return object(), {"en": 1.0}

    class EmbedderModel:
        def parameters(self):
            return {"weight": FakeMlxArray()}

        def encode(self, texts):
            return FakeMlxArray()

    wav = _wav(tmp_path)
    asr = MlxAsrAdapter(_snapshot(tmp_path, "asr-provenance"), load_fn=lambda path: AsrModel(), language_inputs_fn=lambda path, n_mels: FakeMlxArray())
    embedder = MlxEmbeddingAdapter(_snapshot(tmp_path, "embedder-provenance"), load_fn=lambda path: EmbedderModel())
    asr.detect_language_probabilities(wav, TimeRange(start_seconds=0, end_seconds=1), CancellationToken())
    embedder.encode(["context"])

    assert asr.runtime_evidence()["tensors_are_mlx"] is True
    assert embedder.runtime_evidence()["tensors_are_mlx"] is True


def test_text_audio_adapter_records_both_resident_towers_as_mlx_tensors(tmp_path: Path):
    from engine.models.text_audio import MlxTextAudioAdapter

    class FakeMlxArray:
        __module__ = "mlx.core"

    weight = FakeMlxArray()
    model = SimpleNamespace(
        parameters=lambda: {"weight": weight},
        audio_tower=SimpleNamespace(conv1=SimpleNamespace(weight=weight)),
        language_model=SimpleNamespace(
            model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight))
        ),
    )
    adapter = MlxTextAudioAdapter(
        _snapshot(tmp_path, "text-audio-provenance"), load_fn=lambda _path: model
    )

    evidence = adapter.runtime_evidence()

    assert evidence["parameter_leaves_are_mlx"] is True
    assert set(evidence["tensor_types"]) == {
        "audio_tower_input_weight",
        "language_model_embedding_weight",
    }
    assert all(value.startswith("mlx.core") for value in evidence["tensor_types"].values())
    assert evidence["tensors_are_mlx"] is True


def test_model_response_decoder_canonicalizes_json_and_literal_containers_only():
    from engine.models.response import canonical_json_container

    assert canonical_json_container('{"request":"json"}') == '{"request":"json"}'
    assert canonical_json_container("{'request': 'literal'}") == '{"request":"literal"}'
    assert canonical_json_container("['first', {'second': False}]") == '["first",{"second":false}]'

    for invalid in (
        "plain prose",
        "'scalar'",
        "(1, 2)",
        "__import__('os').system('echo unsafe')",
        "{1: 'value'}",
        "{False: 'value'}",
        "{None: 'value'}",
        "{'value': (1, 2)}",
        "{'value': {1, 2}}",
        "{'value': 1e400}",
    ):
        with pytest.raises(ValueError, match="JSON container"):
            canonical_json_container(invalid)

    with pytest.raises(ValueError, match="JSON container"):
        canonical_json_container("[" + "0," * 8192 + "0]")
