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


def _ground_truth_samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for language in ("en", "zh", "ja"):
        for number in range(5):
            samples.append({"id": f"{language}-speech-{number}", "audio_path": "/private/speech.wav", "language": language, "content_types": ["speech"], "taxonomy_ids": ["150"], "is_music": False, "vocal_presence": "unknown", "genres": [], "moods": []})
    for content_type in ("vocal_music", "speech_sparse_music", "instrumental_music"):
        for number in range(5):
            samples.append({"id": f"{content_type}-{number}", "audio_path": "/private/music.wav", "language": "none", "content_types": [content_type], "taxonomy_ids": ["150"], "is_music": True, "vocal_presence": "instrumental", "genres": ["ambient"], "moods": ["calm"]})
    return samples


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


def test_accuracy_harness_counts_ancestor_or_descendant_taxonomy_matches():
    harness = _load_script("accuracy_harness")

    assert harness.hierarchical_match("150", "151") is True
    assert harness.hierarchical_match("151", "150") is True
    assert harness.hierarchical_match("151", "179") is False


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


def test_accuracy_harness_generates_observations_from_an_injected_runner_not_labels():
    harness = _load_script("accuracy_harness")
    samples = _ground_truth_samples()

    observations = harness.generate_observations(
        samples,
        lambda sample: {
            "taxonomy_ids": ["150"],
            "language": sample["language"],
            "is_music": sample["is_music"],
            "vocal_presence": sample["vocal_presence"],
            "genres": sample["genres"],
            "moods": sample["moods"],
            "transcript": "derived audio text",
        },
    )

    assert set(observations) == {sample["id"] for sample in samples}
    assert "observed" not in samples[0]
    assert observations["en-speech-0"]["transcript"] == "derived audio text"


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


def test_accuracy_observer_derives_predictions_from_audio_adapters(tmp_path: Path):
    harness = _load_script("accuracy_harness")
    wav = _wav(tmp_path)
    calls: list[str] = []

    class Asr:
        def detect_language_probabilities(self, audio, window, token):
            calls.append("language")
            return {"en": 0.9, "ja": 0.1}

        def transcribe_window(self, audio, window, language, token):
            calls.append("transcribe")
            from engine.domain import TranscriptSegment

            return [TranscriptSegment(time_range=TimeRange(start_seconds=0, end_seconds=1), text="museum attractions", confidence=0.8, language=language)]

    class Embedder:
        def encode(self, texts):
            calls.append(f"embed:{len(texts)}")
            if len(texts) > 1:
                return np.asarray(
                    [
                        [1.0, 0.0]
                        if text.startswith("Category: Attractions\n")
                        else [0.0, 1.0]
                        for text in texts
                    ],
                    dtype=np.float32,
                )
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    class TextAudio:
        def generate_audio(self, audio, window, prompt, *, max_tokens, token):
            assert "first response character must be {" in prompt
            assert "never backslash-escape quotation marks" in prompt
            assert "never wrap the object in a string" in prompt
            assert '"is_music": false' in prompt
            assert "unquoted JSON boolean true or the unquoted JSON boolean false" in prompt
            assert "True" not in prompt and "False" not in prompt
            assert "exactly these four keys and no others" in prompt
            calls.append("audio")
            return '{"is_music": false, "vocal_presence": "unknown", "genres": [], "moods": []}'

        def generate_text(self, prompt, *, max_tokens, token):
            assert "museum attractions" in prompt
            calls.append("text")
            return json.dumps(
                {
                    "taxonomy_ids": ["150"],
                    "keywords": [],
                    "entities": [],
                    "sentiment": {"label": "neutral", "score": 0.0},
                }
            )

    text_audio = TextAudio()
    registry = type(
        "Registry",
        (),
        {
            "asr": Asr(),
            "embedder": Embedder(),
            "audio_understanding": text_audio,
            "text_generator": text_audio,
        },
    )()
    observer = harness.MlxObservationRunner(registry)
    observed = observer({"id": "sample", "audio_path": str(wav)})

    assert calls == [
        "language",
        "transcribe",
        "audio",
        "embed:703",
        "embed:1",
        "text",
    ]
    assert observed["language"] == "en"
    assert len(observed["retrieval_shortlist"]) == 15
    assert observed["taxonomy_ids"] == ["150"]
    assert observed["transcript"] == "museum attractions"


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


def test_model_smoke_selects_a_speech_labelled_sample_not_language_none(tmp_path: Path):
    smoke = _load_script("model_smoke")
    wav = _wav(tmp_path)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"samples": [
        {"audio_path": str(wav), "language": "none", "content_types": ["instrumental_music"]},
        {"audio_path": str(wav), "language": "ja", "content_types": ["speech"]},
    ]}))

    audio, language = smoke.sample_speech_audio(labels)

    assert audio == wav
    assert language == "ja"


def test_committed_label_schema_and_example_forbid_observed_predictions():
    schema = json.loads((Path(__file__).parents[1] / "benchmarks" / "manifest.schema.json").read_text())
    example = json.loads((Path(__file__).parents[1] / "benchmarks" / "labels.example.json").read_text())

    required = schema["properties"]["samples"]["items"]["required"]
    assert "observed" not in required
    assert "observed" not in schema["properties"]["samples"]["items"]["properties"]
    assert "observed" not in example["samples"][0]


def test_model_smoke_alternates_distinct_text_and_audio_isolation_requests(tmp_path: Path):
    smoke = _load_script("model_smoke")
    wav = _wav(tmp_path)
    calls: list[tuple[str, str]] = []

    class TextAudioModel:
        def generate_text(self, prompt, *, max_tokens, token):
            request = "text-first" if "text-first" in prompt else "text-second"
            calls.append(("text", request))
            return json.dumps({"request": request})

        def generate_audio(self, audio, window, prompt, *, max_tokens, token):
            request = "audio-first" if "audio-first" in prompt else "audio-second"
            calls.append(("audio", request))
            return json.dumps({"request": request})

    results = smoke.exercise_prompt_isolation(TextAudioModel(), wav, TimeRange(start_seconds=0, end_seconds=1))

    assert calls == [("text", "text-first"), ("audio", "audio-first"), ("text", "text-second"), ("audio", "audio-second")]
    assert results == {"text": [{"request": "text-first"}, {"request": "text-second"}], "audio": [{"request": "audio-first"}, {"request": "audio-second"}]}


def test_model_smoke_records_call_throughput_from_cold_and_warm_latencies():
    smoke = _load_script("model_smoke")

    assert smoke._calls_per_second([2.0, 0.5]) == [0.5, 2.0]
    with pytest.raises(ValueError, match="positive"):
        smoke._calls_per_second([0.0])


def test_model_gate_errors_use_generic_text_audio_contract():
    smoke = _load_script("model_smoke")
    harness = _load_script("accuracy_harness")

    with pytest.raises(RuntimeError, match="text/audio model"):
        smoke._expect_isolated_json("not-json", "request")
    with pytest.raises(ValueError, match="text/audio model"):
        harness._parse_acoustic("not-json")


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


def test_model_boundaries_accept_canonicalized_literal_mappings_without_weakening_schemas():
    smoke = _load_script("model_smoke")
    harness = _load_script("accuracy_harness")

    assert smoke._expect_isolated_json("{'request': 'audio-first'}", "audio-first") == {
        "request": "audio-first"
    }
    assert harness._parse_acoustic(
        "{'is_music': False, 'vocal_presence': 'unknown', 'genres': [], 'moods': []}"
    ) == {
        "is_music": False,
        "vocal_presence": "unknown",
        "genres": [],
        "moods": [],
    }
    with pytest.raises(ValueError, match="invalid schema"):
        harness._parse_acoustic("{'is_music': False}")
    for invalid in (
        '{"is_music":false,"vocal_presence":"unknown","genres":"rock","moods":[]}',
        '{"is_music":false,"vocal_presence":"unknown","genres":[],"moods":"calm"}',
    ):
        with pytest.raises(ValueError, match="invalid labels"):
            harness._parse_acoustic(invalid)


def test_accuracy_harness_classifies_missing_prerequisites_as_blocked_and_runtime_as_failed():
    harness = _load_script("accuracy_harness")

    assert harness.error_result(FileNotFoundError("labels missing"))["status"] == "blocked"
    assert harness.error_result(RuntimeError("model malformed output"))["status"] == "failed"
    assert harness.error_result(OSError("disk read failed"))["status"] == "failed"


def test_model_smoke_classifies_only_missing_prerequisites_as_blocked():
    smoke = _load_script("model_smoke")

    assert smoke.error_result(FileNotFoundError("labels missing"))["status"] == "blocked"
    assert smoke.error_result(ValueError("tampered manifest"))["status"] == "failed"
    assert smoke.error_result(RuntimeError("MLX provenance failed"))["status"] == "failed"
    assert smoke.error_result(OSError("model execution failed"))["status"] == "failed"
    assert smoke.error_result(ImportError("mlx API missing"))["status"] == "failed"


def test_accuracy_mood_metric_excludes_non_music_empty_labels_and_requires_eligible_music():
    harness = _load_script("accuracy_harness")
    samples = _ground_truth_samples()
    observations = {
        sample["id"]: {
            "taxonomy_ids": ["150"],
            "language": sample["language"],
            "is_music": sample["is_music"],
            "vocal_presence": sample["vocal_presence"],
            "genres": [],
            "moods": [],
        }
        for sample in samples
    }
    thresholds = {name: 0.0 for name in json.loads((Path(__file__).parents[1] / "benchmarks" / "acceptance.json").read_text())["thresholds"] if name not in {"maximum_cancellation_latency_seconds", "maximum_ffmpeg_cancellation_latency_seconds", "maximum_peak_rss_bytes_on_current_64gb_host"}}

    result = harness.score(samples, observations, thresholds)

    assert result["metrics"]["mood_multilabel_f1"] == 0.0
    for sample in samples:
        if sample["is_music"]:
            sample["moods"] = []
    with pytest.raises(ValueError, match="mood"):
        harness.score(samples, observations, thresholds)


def test_acceptance_records_current_64gb_host_without_changing_memory_limit():
    thresholds = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "acceptance.json").read_text()
    )["thresholds"]

    assert "maximum_peak_rss_bytes_on_16gb" not in thresholds
    assert thresholds["maximum_peak_rss_bytes_on_current_64gb_host"] == 14_495_514_624


def test_gemma_gate_failure_evidence_records_raw_once_templated_cli_controls():
    evidence = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "gemma-gate-failure.json").read_text()
    )

    assert evidence["model"] == {
        "repository": "mlx-community/gemma-3n-E2B-it-4bit",
        "revision": "66e7276cfc589073b4f92ebed58c05301f990222",
    }
    assert evidence["runtime"] == {
        "mlx": "0.32.0",
        "mlx_vlm": "0.6.10",
        "python": "3.13.2",
        "transformers": "5.14.0",
    }
    assert evidence["environment"] == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert evidence["cli_template_flow"]["chat_template_applications"] == 1
    assert evidence["cli_template_flow"]["raw_prompt_assignment"] == {
        "path": "mlx_vlm/generate/dispatch.py",
        "line": 1352,
        "statement": "prompt = args.prompt",
    }
    assert evidence["cli_template_flow"]["template_call"]["line"] == 1428
    assert evidence["cli_template_flow"]["generation_call"]["line"] == 1562
    assert evidence["cli_template_flow"]["tokenization_without_retemplating"]["line"] == 816
    assert evidence["cli_template_flow"]["installed_source_sha256"] == {
        "mlx_vlm/generate/dispatch.py": "c6f62853331dd00a3247fd339c157301d7917e167fa812dfdbb2573cc8d8e41e",
        "mlx_vlm/prompt_utils.py": "a4a07a7c7f459f7ed357584a6b66eec0fe42aa45e5dbe9676240439be5d1a205",
    }

    expected = {
        "isolation": {
            "raw_prompt": 'Reply with exactly {"request":"text-first"}',
            "max_tokens": "24",
            "formatted_prompt_utf8_bytes": 103,
            "formatted_prompt_sha256": "7c3a50e16b0c1e84c142d72c8cad72514bf0693aaefdb077db24307f02ae0f8d",
            "response_sha256": "1461e356563b486c7ffb625128c133528578bd7230fe8bfe9a2782aa124304eb",
            "stdout_sha256": "8833e930f64f69de78eed391fb097d36584825f4f0436ddef9c1afdba4a32146",
        },
        "arithmetic_control": {
            "raw_prompt": "What is 2 plus 2? Answer with one number.",
            "max_tokens": "32",
            "formatted_prompt_utf8_bytes": 101,
            "formatted_prompt_sha256": "fadb8ebcb0bd159ff239e7a89659368dbc72e3d6a737f495a2c8c1cf9923e4ad",
            "response_sha256": "f5effce46b15593172337a46fae9b0f629a65da40a4fa4199e27e11b128c1955",
            "stdout_sha256": "f469b204d306c98a86e23da6f8947890b31059f81522beb7ebeb05f6e958a09c",
        },
    }
    for control in evidence["controls"]:
        wanted = expected[control["name"]]
        assert control["argv"] == [
            "uv", "run", "python", "-m", "mlx_vlm.generate",
            "--model", "data/models/gemma",
            "--max-tokens", wanted["max_tokens"],
            "--temperature", "0.0",
            "--no-verbose",
            "--prompt", wanted["raw_prompt"],
        ]
        assert "<bos>" not in control["raw_prompt"]
        assert control["raw_prompt"] == wanted["raw_prompt"]
        formatted_bytes = control["formatted_prompt"].encode()
        assert len(formatted_bytes) == wanted["formatted_prompt_utf8_bytes"]
        assert control["formatted_prompt_utf8_bytes"] == wanted["formatted_prompt_utf8_bytes"]
        assert hashlib.sha256(formatted_bytes).hexdigest() == wanted["formatted_prompt_sha256"]
        assert control["formatted_prompt_sha256"] == wanted["formatted_prompt_sha256"]
        response_bytes = control["response_text"].encode()
        assert len(response_bytes) == control["response_utf8_bytes"]
        assert hashlib.sha256(response_bytes).hexdigest() == wanted["response_sha256"]
        assert control["response_sha256"] == wanted["response_sha256"]
        assert control["stdout_utf8_bytes"] == control["response_utf8_bytes"] + 1
        assert hashlib.sha256(response_bytes + b"\n").hexdigest() == wanted["stdout_sha256"]
        assert control["stdout_sha256"] == wanted["stdout_sha256"]
        assert control["exit_code"] == 0


def test_qwen2_audio_failure_evidence_is_auditable_without_private_raw_outputs():
    evidence = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "qwen2-audio-gate-failure.json").read_text()
    )

    assert evidence["gate_status"] == "failed"
    assert evidence["contains_private_media_labels_or_transcripts"] is False
    assert evidence["model"] == {
        "repository": "mlx-community/Qwen2-Audio-7B-Instruct-4bit",
        "revision": "c65570002626f41b4dc08b7b54f42f99f3e82e7f",
    }
    assert evidence["snapshot"] == {
        "manifest_sha256": "afd293c91f76541431135038f168ec3c45df090202ebc175d76e3895f0b12e00",
        "file_count": 30,
        "total_bytes": 6574662274,
        "weights_sha256": "0967cde270ad62aa4824f0bbce283d0a2a6da2825ccf587640753c527d4174da",
    }
    assert evidence["normalization"]["fallback"] == "ast.literal_eval top-level dict/list only"
    assert evidence["normalization"]["canonicalizer"] == "json.dumps allow_nan=False"
    assert evidence["normalized_gate"]["model_smoke"]["status"] == "passed"
    assert evidence["normalized_gate"]["model_smoke"]["peak_rss_bytes"] == 9000845312
    assert evidence["normalized_gate"]["model_smoke"]["swap_delta_bytes"] == 0
    assert evidence["normalized_gate"]["accuracy"]["status"] == "failed"
    assert evidence["normalized_gate"]["accuracy"]["reason"] == "text/audio model acoustic response has invalid scalar values"
    assert evidence["normalized_gate"]["accuracy"]["failing_stratum"] == "English speech"
    rerun = evidence["fix_round_1"]
    assert rerun["prompt_contract"] == "RFC 8259 JSON with lowercase true/false"
    assert rerun["decoder_rejects_non_json_literal_contents"] is True
    assert rerun["acoustic_label_arrays_required"] is True
    assert rerun["model_smoke"]["status"] == "passed"
    assert rerun["model_smoke"]["peak_rss_bytes"] == 9005449216
    assert rerun["accuracy"] == {
        "command": "uv run python scripts/accuracy_harness.py --manifest benchmarks/labels.local.json",
        "exit_code": 2,
        "status": "failed",
        "artifact_sha256": "c16dd8010df93492d0ad5a63dd73a94e15bb9368c2f1f0d84cd4f9420226ef74",
        "reason": "text/audio model acoustic response has invalid scalar values",
        "sample_index": 0,
        "failing_stratum": "English speech",
        "invalid_value_type": "quoted boolean-like string",
        "raw_utf8_bytes": 77,
        "raw_sha256": "fdc31a24c1d5bd4bea64afcd707b89a48d31c6677d57b377b01ea320962cf97e",
        "metrics_available": False,
        "assessment": "Corrected strict-JSON prompt still produced a quoted boolean-like string; no coercion or threshold change was applied.",
    }
    assert rerun["thresholds_unchanged"] is True
    assert rerun["task_7_started"] is False
    assert "response_text" not in json.dumps(evidence)


def test_accuracy_manifest_rejects_coercible_music_and_malformed_acoustic_labels():
    harness = _load_script("accuracy_harness")
    samples = _ground_truth_samples()
    samples[0]["is_music"] = "false"
    with pytest.raises(ValueError, match="is_music"):
        harness.validate_manifest({"schema_version": 1, "samples": samples})

    samples = _ground_truth_samples()
    samples[-1]["genres"] = "ambient"
    with pytest.raises(ValueError, match="genres"):
        harness.validate_manifest({"schema_version": 1, "samples": samples})


@pytest.mark.parametrize("script_name", ["model_smoke", "accuracy_harness"])
def test_gate_clis_write_failed_evidence_for_mlx_import_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script_name: str):
    script = _load_script(script_name)
    manifest = tmp_path / "labels.json"
    manifest.write_text("{}")
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", [f"{script_name}.py", "--manifest", str(manifest), "--output", str(output)])
    if script_name == "model_smoke":
        monkeypatch.setattr(script, "run", lambda *_args: (_ for _ in ()).throw(ImportError("mlx API missing")))
    else:
        monkeypatch.setattr(script, "_load_json", lambda _path: (_ for _ in ()).throw(ImportError("mlx API missing")))

    assert script.main() == 2
    assert json.loads(output.read_text())["status"] == "failed"
