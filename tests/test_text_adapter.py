from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import importlib.util

import pytest

from engine.cancel import CancellationToken, JobCancelled


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Chunk:
    def __init__(self, text: str, tokens: int, finish_reason: str | None = None):
        self.text = text
        self.generation_tokens = tokens
        self.finish_reason = finish_reason


class Stream:
    def __init__(self, chunks, *, failure: BaseException | None = None):
        self.chunks = iter(chunks)
        self.failure = failure
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.failure:
            raise self.failure

    def close(self):
        self.closed = True


def _runtime(streams, events):
    model = object()

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            return "formatted"

    class Processor:
        def __call__(self, tokens, logits):
            events.append(("processor", tokens, logits))
            return logits

    def stream_generate(*args, **kwargs):
        events.append(("stream_generate", args, kwargs))
        return streams.pop(0)

    return SimpleNamespace(
        load=lambda path, lazy: (model, Tokenizer()),
        wrap=lambda loaded_model, tokenizer: (loaded_model, tokenizer),
        get_processor=lambda backend, wrapped, schema: Processor(),
        make_sampler=lambda temperature: ("sampler", temperature),
        stream_generate=stream_generate,
        eval=lambda tokens: events.append(("eval", tokens)),
        clear_cache=lambda: events.append(("clear_cache",)),
    )


def test_adapter_builds_a_fresh_schema_processor_uses_greedy_streaming_and_materializes_tokens(tmp_path: Path):
    from engine.models.text import MlxConstrainedTextAdapter

    streams = [
        Stream([Chunk('{"ok":', 1), Chunk("true}", 2, "stop")]),
        Stream([Chunk('{"ok":false}', 1, "stop")]),
    ]
    events: list[tuple] = []
    adapter = MlxConstrainedTextAdapter(
        tmp_path.resolve(), runtime=_runtime(streams, events), timeout_seconds=30
    )
    schema = {"type": "object"}

    first = adapter.generate_json(
        "prompt", schema, max_tokens=1024, token=CancellationToken()
    )
    second = adapter.generate_json(
        "prompt", schema, max_tokens=16, token=CancellationToken()
    )

    assert first.raw == '{"ok":true}'
    assert first.finish_reason == "stop"
    assert first.token_count == 2
    assert second.raw == '{"ok":false}'
    calls = [event for event in events if event[0] == "stream_generate"]
    assert calls[0][2]["max_tokens"] == 1024
    assert calls[0][2]["sampler"] == ("sampler", 0.0)
    processors = calls[0][2]["logits_processors"] + calls[1][2]["logits_processors"]
    assert processors[0] is not processors[1]
    processors[0]([1], [2])
    assert events[-2:] == [("eval", [1]), ("processor", [1], [2])]


def test_adapter_closes_stream_on_cancel_and_generation_error(tmp_path: Path):
    from engine.models.text import MlxConstrainedTextAdapter

    token = CancellationToken()

    class CancellingStream(Stream):
        def __iter__(self):
            yield Chunk("first", 1)
            token.cancel()
            yield Chunk("second", 2)

    cancelled = CancellingStream([])
    failed = Stream([], failure=RuntimeError("boom"))
    events: list[tuple] = []
    adapter = MlxConstrainedTextAdapter(
        tmp_path.resolve(), runtime=_runtime([cancelled, failed], events)
    )

    with pytest.raises(JobCancelled):
        adapter.generate_json("prompt", {"type": "object"}, max_tokens=8, token=token)
    assert cancelled.closed is True

    with pytest.raises(RuntimeError, match="boom"):
        adapter.generate_json(
            "prompt", {"type": "object"}, max_tokens=8, token=CancellationToken()
        )
    assert failed.closed is True


@pytest.mark.parametrize("max_tokens", [0, 1025, True, 1.5])
def test_adapter_rejects_unbounded_or_noninteger_token_limits_before_streaming(
    tmp_path: Path, max_tokens
):
    from engine.models.text import MlxConstrainedTextAdapter

    events: list[tuple] = []
    adapter = MlxConstrainedTextAdapter(
        tmp_path.resolve(), runtime=_runtime([], events)
    )
    with pytest.raises(ValueError, match="1..1024"):
        adapter.generate_json(
            "prompt",
            {"type": "object"},
            max_tokens=max_tokens,
            token=CancellationToken(),
        )
    assert not [event for event in events if event[0] == "stream_generate"]


def test_model_module_import_does_not_eagerly_import_or_initialize_mlx(monkeypatch):
    for name in ("engine.models.text", "mlx", "mlx_lm", "outlines"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    __import__("engine.models.text")

    assert "mlx" not in sys.modules
    assert "mlx_lm" not in sys.modules
    assert "outlines" not in sys.modules


def _manifest(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    import engine.models.registry as registry

    snapshot = (tmp_path / "qwen36_text").resolve()
    snapshot.mkdir()
    expected = {}
    inventory = []
    for number in range(1, 5):
        name = f"model-{number:05d}-of-00004.safetensors"
        payload = f"weight-{number}".encode()
        (snapshot / name).write_bytes(payload)
        record = {
            "path": name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        expected[name] = {"bytes": record["bytes"], "sha256": record["sha256"]}
        inventory.append(record)
    monkeypatch.setattr(registry, "QWEN36_WEIGHT_FILES", expected)
    manifest = tmp_path / "qwen36-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {
                    **registry.QWEN36_TEXT_SPEC,
                    "local_path": str(snapshot),
                    "files": inventory,
                },
            }
        )
    )
    return manifest, snapshot


def test_verified_qwen36_manifest_checks_four_pinned_shards_before_load(
    tmp_path: Path, monkeypatch
):
    from engine.models.registry import load_verified_text_path

    manifest, snapshot = _manifest(tmp_path, monkeypatch)
    assert load_verified_text_path(manifest) == snapshot

    (snapshot / "model-00004-of-00004.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size|SHA-256"):
        load_verified_text_path(manifest)


def test_text_registry_loads_one_resident_adapter_and_releases_it(
    tmp_path: Path, monkeypatch
):
    from engine.models.registry import TextModelRegistry, load_verified_text_path

    manifest, snapshot = _manifest(tmp_path, monkeypatch)
    path = load_verified_text_path(manifest)
    loads: list[Path] = []
    released: list[int] = []

    class Adapter:
        def release(self):
            released.append(1)

    registry = TextModelRegistry(
        path, load_text=lambda selected: loads.append(selected) or Adapter()
    )

    assert registry.generator is registry.generator
    assert loads == [snapshot]
    assert registry.release() is True
    assert released == [1]
    assert registry.release() is False
    assert loads == [snapshot]


def test_setup_registers_existing_qwen36_snapshot_without_selecting_legacy_audio(
    tmp_path: Path, monkeypatch
):
    import engine.models.registry as registry

    fetch_models = _load_script("fetch_models")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    manifest, snapshot = _manifest(models_dir, monkeypatch)
    manifest.unlink()
    downloads: list[object] = []
    monkeypatch.setattr(
        fetch_models,
        "snapshot_download",
        lambda **kwargs: downloads.append(kwargs),
        raising=False,
    )

    payload = fetch_models.fetch_text_model(models_dir, download=False)

    expected_manifest = models_dir / "qwen36-manifest.json"
    assert payload["model"]["repository"] == registry.QWEN36_TEXT_SPEC["repository"]
    assert payload["model"]["revision"] == registry.QWEN36_TEXT_SPEC["revision"]
    assert Path(payload["model"]["local_path"]) == snapshot
    assert registry.load_verified_text_path(expected_manifest) == snapshot
    assert downloads == []


def test_asr_only_manifest_loader_ignores_deleted_legacy_model_roles(
    tmp_path: Path,
):
    from engine.models.registry import MODEL_SPECS, load_verified_asr_path

    snapshot = (tmp_path / "asr").resolve()
    snapshot.mkdir()
    weights = b"asr-weights"
    (snapshot / "model.safetensors").write_bytes(weights)
    manifest = tmp_path / "model-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "asr": {
                        **MODEL_SPECS["asr"],
                        "local_path": str(snapshot),
                        "files": [
                            {
                                "path": "model.safetensors",
                                "bytes": len(weights),
                                "sha256": hashlib.sha256(weights).hexdigest(),
                            }
                        ],
                    },
                    "text_audio": {
                        "local_path": str(tmp_path / "deleted"),
                        "files": [],
                    },
                },
            }
        )
    )

    assert load_verified_asr_path(manifest) == snapshot


@pytest.mark.parametrize(
    "loader_name",
    ["load_verified_embedding_path", "load_verified_text_path"],
)
def test_single_role_manifest_loaders_reject_a_directory_with_controlled_error(
    tmp_path: Path,
    loader_name: str,
):
    import engine.models.registry as registry

    loader = getattr(registry, loader_name)
    with pytest.raises(ValueError, match="invalid .*model manifest"):
        loader(tmp_path)
