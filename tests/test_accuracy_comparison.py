from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from engine.cancel import JobCancelled
from engine.domain import TimeRange, TranscriptSegment


def _load_harness():
    path = Path(__file__).parents[1] / "scripts" / "accuracy_harness.py"
    spec = importlib.util.spec_from_file_location("accuracy_harness_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _thresholds() -> dict[str, float]:
    path = Path(__file__).parents[1] / "benchmarks" / "acceptance.json"
    return {
        name: value
        for name, value in json.loads(path.read_text())["thresholds"].items()
        if not name.startswith("maximum_")
    }


def _samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for language in ("en", "zh", "ja"):
        for number in range(5):
            samples.append(
                {
                    "id": f"{language}-speech-{number}",
                    "audio_path": f"/private/{language}-{number}.wav",
                    "language": language,
                    "content_types": ["speech"],
                    "taxonomy_ids": ["150"],
                    "is_music": False,
                    "vocal_presence": "unknown",
                    "genres": [],
                    "moods": [],
                }
            )
    for content_type in ("vocal_music", "speech_sparse_music", "instrumental_music"):
        for number in range(5):
            samples.append(
                {
                    "id": f"{content_type}-{number}",
                    "audio_path": f"/private/{content_type}-{number}.wav",
                    "language": "none",
                    "content_types": [content_type],
                    "taxonomy_ids": ["338"],
                    "is_music": True,
                    "vocal_presence": "instrumental",
                    "genres": ["ambient"],
                    "moods": ["calm"],
                }
            )
    return samples


def _successful_observation(sample: dict[str, object]) -> dict[str, object]:
    taxonomy_id = str(sample["taxonomy_ids"][0])  # type: ignore[index]
    return {
        "status": "passed",
        "components": {
            name: {"status": "passed"}
            for name in ("asr", "acoustic", "retrieval", "text_rerank")
        },
        "language": sample["language"],
        "transcript": "audio-derived text",
        "is_music": sample["is_music"],
        "vocal_presence": sample["vocal_presence"],
        "genres": sample["genres"],
        "moods": sample["moods"],
        "retrieval_shortlist": [
            {
                "taxonomy_id": taxonomy_id,
                "name": "hand checked",
                "path": ["hand checked"],
                "similarity": 0.75,
                "provenance": ["transcript"],
            }
        ],
        "taxonomy_ids": [taxonomy_id],
        "text_schema": {
            "keywords": [],
            "entities": [],
            "sentiment": {"label": "neutral", "score": 0.0},
        },
    }


def test_failed_sample_does_not_stop_corpus():
    harness = _load_harness()
    seen: list[str] = []

    def observe(sample):
        seen.append(sample["id"])
        if sample["id"] == "first":
            raise ValueError("invalid model schema")
        return {"taxonomy_ids": []}

    observations = harness.generate_observations(
        [{"id": "first"}, {"id": "second"}], observe
    )

    assert seen == ["first", "second"]
    assert set(observations) == {"first", "second"}
    assert observations["first"]["status"] == "failed"
    assert observations["first"]["errors"] == [
        {"component": "observer", "code": "ValueError", "kind": "execution"}
    ]


def test_progress_is_safe_and_checkpoint_preserves_each_completed_sample():
    harness = _load_harness()
    progress: list[dict[str, object]] = []
    checkpoints: list[dict[str, dict[str, object]]] = []
    samples = [
        {"id": "private-id", "audio_path": "/private/secret.wav"},
        {"id": "second", "audio_path": "/private/second.wav"},
    ]

    harness.generate_observations(
        samples,
        lambda sample: {
            "status": "private transcript" if sample["id"] == "private-id" else "passed",
            "transcript": "private transcript",
        },
        on_progress=progress.append,
        checkpoint=lambda value: checkpoints.append(json.loads(json.dumps(value))),
    )

    assert progress == [
        {"sample_index": 0, "sample_count": 2, "status": "failed"},
        {"sample_index": 1, "sample_count": 2, "status": "passed"},
    ]
    assert list(checkpoints[0]) == ["private-id"]
    assert list(checkpoints[1]) == ["private-id", "second"]
    assert checkpoints[0]["private-id"]["transcript"] == "private transcript"
    assert "/private/secret.wav" not in json.dumps(progress)
    assert "private transcript" not in json.dumps(progress)
    assert "private-id" not in json.dumps(progress)


@pytest.mark.parametrize("error", [JobCancelled(), KeyboardInterrupt()])
def test_generate_observations_propagates_cancellation_and_interrupts(error):
    harness = _load_harness()

    def cancelled(_sample):
        raise error

    with pytest.raises(type(error)):
        harness.generate_observations([{"id": "first"}], cancelled)


@pytest.mark.parametrize(
    ("audio_result", "error_kind"),
    [
        (
            '{"is_music": "false", "vocal_presence": "unknown", "genres": [], "moods": []}',
            "schema",
        ),
        (ValueError("audio inference failed"), "execution"),
    ],
)
def test_runner_keeps_text_results_when_acoustic_component_fails(
    tmp_path: Path, audio_result: str | Exception, error_kind: str
):
    harness = _load_harness()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")
    text_prompts: list[str] = []

    class Asr:
        def detect_language_probabilities(self, *_args):
            return {"en": 1.0}

        def transcribe_window(self, *_args):
            return [
                TranscriptSegment(
                    time_range=TimeRange(start_seconds=0, end_seconds=1),
                    text="museum attractions",
                    confidence=0.9,
                    language="en",
                )
            ]

    class Embedder:
        def encode(self, texts):
            if len(texts) > 1:
                return np.asarray(
                    [[1.0, 0.0] if text.startswith("Category: Attractions\n") else [0.0, 1.0] for text in texts],
                    dtype=np.float32,
                )
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    class AudioUnderstanding:
        def generate_audio(self, *_args, **_kwargs):
            if isinstance(audio_result, Exception):
                raise audio_result
            return audio_result

    class TextGenerator:
        def generate_text(self, prompt, **_kwargs):
            text_prompts.append(prompt)
            return json.dumps(
                {
                    "taxonomy_ids": ["150"],
                    "keywords": [],
                    "entities": [],
                    "sentiment": {"label": "neutral", "score": 0.0},
                }
            )

    registry = type(
        "Registry",
        (),
        {
            "asr": Asr(),
            "embedder": Embedder(),
            "audio_understanding": AudioUnderstanding(),
            "text_generator": TextGenerator(),
        },
    )()
    sample = {
        "id": "private-id",
        "audio_path": str(audio),
        "language": "ja",
        "content_types": ["secret-source-tag"],
        "taxonomy_ids": ["do-not-leak-truth"],
        "genres": ["truth-only"],
    }

    observed = harness.MlxObservationRunner(registry)(sample)

    assert observed["components"]["acoustic"] == {
        "status": "failed",
        "error_code": "ValueError",
        "error_kind": error_kind,
    }
    assert observed["components"]["text_rerank"] == {"status": "passed"}
    assert observed["taxonomy_ids"] == ["150"]
    assert observed["transcript"] == "museum attractions"
    assert text_prompts
    for forbidden in (
        "private-id",
        str(audio),
        "secret-source-tag",
        "do-not-leak-truth",
        "truth-only",
    ):
        assert forbidden not in text_prompts[0]


def test_empty_transcript_uses_instrumental_evidence_for_music_only_retrieval(tmp_path: Path):
    harness = _load_harness()
    audio = tmp_path / "instrumental.wav"
    audio.write_bytes(b"RIFF")
    encoded: list[list[str]] = []
    prompts: list[str] = []

    class Asr:
        def detect_language_probabilities(self, *_args):
            return {"en": 1.0}

        def transcribe_window(self, *_args):
            return []

    class Embedder:
        def encode(self, texts):
            encoded.append(list(texts))
            if len(texts) > 1:
                return np.asarray(
                    [[1.0, 0.0] if text.startswith("Category: Music\n") else [0.0, 1.0] for text in texts],
                    dtype=np.float32,
                )
            assert texts == [
                "Music. Vocal presence: instrumental. Genres: ambient. Moods: calm."
            ]
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    class TextAudio:
        def generate_audio(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "is_music": True,
                    "vocal_presence": "instrumental",
                    "genres": ["Ambient"],
                    "moods": ["Calm"],
                }
            )

        def generate_text(self, prompt, **_kwargs):
            prompts.append(prompt)
            return json.dumps(
                {
                    "taxonomy_ids": ["338"],
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

    observed = harness.MlxObservationRunner(registry)(
        {"id": "sample", "audio_path": str(audio)}
    )

    assert observed["transcript"] == ""
    assert observed["taxonomy_ids"] == ["338"]
    assert observed["retrieval_shortlist"][0]["taxonomy_id"] == "338"
    assert observed["retrieval_shortlist"][0]["similarity"] == 1.0
    assert observed["retrieval_shortlist"][0]["provenance"] == ["acoustic"]
    assert all("Music" in item["path"] for item in observed["retrieval_shortlist"])
    assert observed["retrieval_scope"] == "feasibility_only_not_production_category_acceptance"
    assert "instrumental" in prompts[0]
    assert len(encoded) == 2


def test_taxonomy_embedding_failure_is_retained_as_a_per_sample_component_error(
    tmp_path: Path,
):
    harness = _load_harness()
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF")

    class Asr:
        def detect_language_probabilities(self, *_args):
            return {"en": 1.0}

        def transcribe_window(self, *_args):
            return [
                TranscriptSegment(
                    time_range=TimeRange(start_seconds=0, end_seconds=1),
                    text="museum attractions",
                    confidence=0.9,
                    language="en",
                )
            ]

    class Embedder:
        def encode(self, _texts):
            raise RuntimeError("embedding unavailable")

    class TextAudio:
        def generate_audio(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "is_music": False,
                    "vocal_presence": "unknown",
                    "genres": [],
                    "moods": [],
                }
            )

        def generate_text(self, *_args, **_kwargs):
            raise AssertionError("rerank must not run without retrieval candidates")

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

    runner = harness.MlxObservationRunner(registry)
    observed = runner({"id": "sample", "audio_path": str(audio)})

    assert observed["components"]["asr"] == {"status": "passed"}
    assert observed["components"]["acoustic"] == {"status": "passed"}
    assert observed["components"]["retrieval"] == {
        "status": "failed",
        "error_code": "RuntimeError",
        "error_kind": "execution",
    }
    assert observed["components"]["text_rerank"] == {
        "status": "failed",
        "error_code": "RetrievalUnavailable",
        "error_kind": "dependency",
    }
    assert observed["taxonomy_ids"] == []


def test_rerank_rejects_duplicates_invalid_ids_and_ids_outside_shortlist():
    harness = _load_harness()
    candidates = [
        {
            "taxonomy_id": "338",
            "name": "Music",
            "path": ["Music"],
            "similarity": 0.9,
            "provenance": ["acoustic"],
        }
    ]

    def response(ids):
        return json.dumps(
            {
                "taxonomy_ids": ids,
                "keywords": [],
                "entities": [],
                "sentiment": {"label": "neutral", "score": 0.0},
            }
        )

    assert harness._parse_text_rerank(response(["338"]), candidates)["taxonomy_ids"] == ["338"]
    with pytest.raises(ValueError, match="duplicates"):
        harness._parse_text_rerank(response(["338", "338"]), candidates)
    with pytest.raises(ValueError, match="official"):
        harness._parse_text_rerank(response(["not-official"]), candidates)
    with pytest.raises(ValueError, match="shortlist"):
        harness._parse_text_rerank(response(["339"]), candidates)


def test_frozen_vocabulary_accepts_generic_music_labels_without_collapsing_distinctions():
    harness = _load_harness()

    assert harness.normalize_music_labels("genres", [" Ambient ", "hip-hop"]) == [
        "ambient",
        "hip hop",
    ]
    assert harness.normalize_music_labels(
        "genres", ["cinematic", "soundtrack", "orchestral"]
    ) == ["cinematic", "soundtrack", "orchestral"]
    assert harness.normalize_music_labels(
        "moods", ["CALM", "dramatic", "emotional", "hopeful"]
    ) == ["calm", "dramatic", "emotional", "hopeful"]
    assert harness.normalize_music_labels(
        "moods", ["sad", "melancholic", "upbeat", "energetic"]
    ) == [
        "sad",
        "melancholic",
        "upbeat",
        "energetic",
    ]
    with pytest.raises(ValueError, match="unsupported genre label"):
        harness.normalize_music_labels("genres", ["truth-derived-new-label"])
    with pytest.raises(ValueError, match="unsupported mood label"):
        harness.normalize_music_labels("moods", ["custom mood"])

    evidence = harness.vocabulary_evidence()
    assert evidence["version"] == "2026-09-07.v1"
    assert evidence["sha256"] == "257c452a46c291676ee2942415539971a8d645ef6a5cb3e539dae6568a0c97f4"
    assert evidence["genres"] == [
        "acoustic",
        "ambient",
        "blues",
        "cinematic",
        "classical",
        "country",
        "dance",
        "electronic",
        "folk",
        "gospel",
        "hip hop",
        "indie",
        "jazz",
        "latin",
        "metal",
        "orchestral",
        "pop",
        "punk",
        "r&b",
        "reggae",
        "rock",
        "soundtrack",
        "world",
    ]


def test_score_uses_retrieval_for_recall_and_validated_rerank_for_top3_precision():
    harness = _load_harness()
    samples = _samples()
    observations = {sample["id"]: _successful_observation(sample) for sample in samples}
    for sample in samples:
        observation = observations[sample["id"]]
        correct = str(sample["taxonomy_ids"][0])  # type: ignore[index]
        observation["retrieval_shortlist"] = [
            {
                "taxonomy_id": correct,
                "name": "candidate",
                "path": ["candidate"],
                "similarity": 0.5,
                "provenance": ["transcript"],
            }
        ]
        observation["taxonomy_ids"] = ["17"]

    result = harness.score(samples, observations, _thresholds())

    assert result["metrics"]["taxonomy_labelled_category_recall_at_15"] == 1.0
    assert result["metrics"]["whole_window_top3_hierarchical_precision"] == 0.0
    assert result["status"] == "failed"


def test_acoustic_failure_counts_in_full_denominator_without_suppressing_text_score():
    harness = _load_harness()
    samples = _samples()
    observations = {sample["id"]: _successful_observation(sample) for sample in samples}
    failed = observations[samples[0]["id"]]
    failed["status"] = "failed"
    failed["components"]["acoustic"] = {
        "status": "failed",
        "error_code": "ValueError",
        "error_kind": "schema",
    }
    failed["is_music"] = None
    failed["vocal_presence"] = None
    failed["genres"] = []
    failed["moods"] = []

    result = harness.score(samples, observations, _thresholds())

    assert result["metrics"]["whole_window_top3_hierarchical_precision"] == 1.0
    assert result["metrics"]["music_non_music_accuracy"] == pytest.approx(29 / 30)
    assert result["component_coverage"]["acoustic"] == {
        "successful_samples": 29,
        "eligible_samples": 30,
        "rate": pytest.approx(29 / 30),
    }
    assert result["schema_compliance"]["acoustic"] == pytest.approx(29 / 30)
    assert result["error_counts"] == {"acoustic": {"schema:ValueError": 1}}
    assert result["checks"]["schema_compliance"] is False
    assert result["status"] == "failed"


def test_invalid_text_diagnostics_fail_schema_and_count_prediction_as_incorrect():
    harness = _load_harness()
    samples = _samples()
    observations = {sample["id"]: _successful_observation(sample) for sample in samples}
    observations["en-speech-0"]["text_schema"] = {
        "keywords": [],
        "entities": [],
        "sentiment": {"label": "confident", "score": 2.0},
    }

    result = harness.score(samples, observations, _thresholds())

    assert result["metrics"]["whole_window_top3_hierarchical_precision"] == pytest.approx(
        29 / 30
    )
    assert result["schema_compliance"]["text_rerank"] == pytest.approx(29 / 30)
    assert result["checks"]["schema_compliance"] is False
    assert result["status"] == "failed"


def test_all_failed_observations_produce_zero_metrics_instead_of_disappearing():
    harness = _load_harness()
    samples = _samples()
    observations = {
        sample["id"]: {
            "status": "failed",
            "errors": [
                {"component": "observer", "code": "RuntimeError", "kind": "execution"}
            ],
        }
        for sample in samples
    }

    result = harness.score(samples, observations, _thresholds())

    for name in (
        "taxonomy_labelled_category_recall_at_15",
        "whole_window_top3_hierarchical_precision",
        "language_detection_accuracy",
        "music_non_music_accuracy",
        "vocal_presence_macro_f1",
        "genre_top3_accuracy",
        "mood_multilabel_f1",
    ):
        assert result["metrics"][name] == 0.0
    assert result["coverage"] == {
        "observed_samples": 30,
        "eligible_samples": 30,
        "rate": 1.0,
    }
    assert all(item["eligible_samples"] == 30 for item in result["component_coverage"].values())
    assert result["status"] == "failed"


def test_score_reports_full_denominator_language_and_content_breakdowns():
    harness = _load_harness()
    samples = _samples()
    observations = {sample["id"]: _successful_observation(sample) for sample in samples}
    del observations["en-speech-0"]

    result = harness.score(samples, observations, _thresholds())

    assert result["coverage"] == {
        "observed_samples": 29,
        "eligible_samples": 30,
        "rate": pytest.approx(29 / 30),
    }
    assert result["breakdowns"]["language"]["en"] == {
        "eligible_samples": 5,
        "labelled_categories": 5,
        "retrieval_recall_at_15": 0.8,
        "reranked_top3_precision": 0.8,
        "schema_compliance_rate": 0.8,
    }
    assert result["breakdowns"]["content_type"]["speech"]["eligible_samples"] == 15
    assert result["breakdowns"]["content_type"]["instrumental_music"] == {
        "eligible_samples": 5,
        "labelled_categories": 5,
        "retrieval_recall_at_15": 1.0,
        "reranked_top3_precision": 1.0,
        "schema_compliance_rate": 1.0,
    }
    assert result["checks"]["complete_corpus_coverage"] is False
    assert result["error_counts"]["observation"] == {
        "missing:MissingObservation": 1
    }
    assert result["status"] == "failed"


def test_atomic_write_preserves_previous_checkpoint_when_serialization_fails(
    tmp_path: Path,
):
    harness = _load_harness()
    output = tmp_path / "result.json"
    prior = '{"status":"running","completed_samples":1}\n'
    output.write_text(prior)

    with pytest.raises(TypeError):
        harness._write(output, {"status": "running", "invalid": object()})

    assert output.read_text() == prior


def test_atomic_write_preserves_previous_checkpoint_when_write_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _load_harness()
    output = tmp_path / "result.json"
    prior = '{"status":"running","completed_samples":1}\n'
    output.write_text(prior)

    class InterruptingStream:
        def __init__(self, descriptor: int):
            self._descriptor = descriptor

        def __enter__(self):
            return self

        def write(self, _value):
            raise KeyboardInterrupt

        def __exit__(self, *_args):
            os.close(self._descriptor)

    monkeypatch.setattr(
        harness.os,
        "fdopen",
        lambda descriptor, *_args, **_kwargs: InterruptingStream(descriptor),
    )

    with pytest.raises(KeyboardInterrupt):
        harness._write(output, {"status": "running", "completed_samples": 2})

    assert output.read_text() == prior
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


def test_atomic_write_preserves_previous_checkpoint_when_replace_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    harness = _load_harness()
    output = tmp_path / "result.json"
    prior = '{"status":"running","completed_samples":1}\n'
    output.write_text(prior)
    monkeypatch.setattr(
        harness.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        harness._write(output, {"status": "running", "completed_samples": 2})

    assert output.read_text() == prior
    assert sorted(path.name for path in tmp_path.iterdir()) == ["result.json"]


def _run_model_free_cli(
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    score_result,
):
    manifest = tmp_path / "private-labels.json"
    manifest.write_text(json.dumps({"schema_version": 1, "samples": _samples()}))
    output = tmp_path / "raw-result.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "accuracy_harness.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(harness, "load_verified_model_paths", lambda _path: object())
    monkeypatch.setattr(harness, "ModelRegistry", lambda _paths: object())
    monkeypatch.setattr(
        harness,
        "MlxObservationRunner",
        lambda _registry: lambda _sample: {
            "status": "passed",
            "transcript": "private transcript must stay in raw output",
        },
    )
    monkeypatch.setattr(harness, "score", score_result)

    exit_code = harness.main()
    stdout = capsys.readouterr().out
    return exit_code, stdout, output


def test_cli_stdout_contains_only_safe_progress_and_aggregate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    harness = _load_harness()
    aggregate = {
        "status": "failed",
        "metrics": {"safe_metric": 0.5},
        "checks": {"safe_check": False},
        "coverage": {"observed_samples": 30, "eligible_samples": 30, "rate": 1.0},
        "component_coverage": {"asr": {"rate": 1.0}},
        "schema_compliance": {"asr": 1.0},
        "error_counts": {"text_rerank": {"schema:ValueError": 1}},
        "breakdowns": {"language": {"en": {"eligible_samples": 5}}},
        "retrieval_scope": "feasibility_only_not_production_category_acceptance",
        "vocabulary": {
            "version": "2026-09-07.v1",
            "sha256": "safe-vocabulary-hash",
            "genres": ["must not be copied wholesale"],
        },
        "compatibility": {
            "legacy_observations_using_taxonomy_ids_for_retrieval": 0
        },
    }

    exit_code, stdout, output = _run_model_free_cli(
        harness,
        tmp_path,
        monkeypatch,
        capsys,
        lambda *_args: aggregate,
    )

    lines = [json.loads(line) for line in stdout.splitlines()]
    assert exit_code == 2
    assert len(lines) == 31
    assert lines[0] == {
        "progress": {"sample_count": 30, "sample_index": 0, "status": "passed"}
    }
    assert lines[-1] == {
        "status": "failed",
        "metrics": {"safe_metric": 0.5},
        "checks": {"safe_check": False},
        "coverage": {"observed_samples": 30, "eligible_samples": 30, "rate": 1.0},
        "component_coverage": {"asr": {"rate": 1.0}},
        "schema_compliance": {"asr": 1.0},
        "error_counts": {"text_rerank": {"schema:ValueError": 1}},
        "breakdowns": {"language": {"en": {"eligible_samples": 5}}},
        "retrieval_scope": "feasibility_only_not_production_category_acceptance",
        "vocabulary": {
            "version": "2026-09-07.v1",
            "sha256": "safe-vocabulary-hash",
        },
        "compatibility": {
            "legacy_observations_using_taxonomy_ids_for_retrieval": 0
        },
    }
    assert "private" not in stdout
    assert "must not be copied wholesale" not in stdout
    assert "private transcript must stay in raw output" in output.read_text()


def test_cli_stdout_redacts_top_level_error_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    harness = _load_harness()
    manifest = tmp_path / "private-labels.json"
    manifest.write_text("{}")
    output = tmp_path / "raw-result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["accuracy_harness.py", "--manifest", str(manifest), "--output", str(output)],
    )

    assert harness.main() == 2

    stdout = capsys.readouterr().out
    assert json.loads(stdout) == {"status": "failed", "error_code": "ValueError"}
    assert str(manifest) not in stdout
    assert "schema_version=1 and samples are required" not in stdout
    assert "schema_version=1 and samples are required" in output.read_text()


def test_cli_top_level_failure_does_not_overwrite_completed_sample_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    harness = _load_harness()

    def scoring_failure(*_args):
        raise RuntimeError("private scoring detail")

    exit_code, stdout, output = _run_model_free_cli(
        harness, tmp_path, monkeypatch, capsys, scoring_failure
    )

    retained = json.loads(output.read_text())
    assert exit_code == 2
    assert json.loads(stdout.splitlines()[-1]) == {
        "status": "failed",
        "error_code": "RuntimeError",
    }
    assert retained["status"] == "running"
    assert retained["completed_samples"] == 30
    assert retained["total_samples"] == 30
    assert len(retained["observations"]) == 30
    assert "private transcript must stay in raw output" in output.read_text()
