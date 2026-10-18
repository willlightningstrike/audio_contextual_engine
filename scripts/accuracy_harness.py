"""Validate a private labelled corpus and score recorded real-model observations."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.models.registry import (
    CANDIDATES,
    ModelRegistry,
    candidate_manifest_path,
    load_verified_model_paths,
)
from engine.standards import load_taxonomy

try:
    from scripts.benchmark_contracts import (
        ACOUSTIC_JSON_PROMPT,
        acoustic_description as _acoustic_description,
        normalize_music_labels,
        parse_acoustic as _parse_acoustic,
        parse_text_rerank as _parse_text_rerank,
        text_rerank_prompt as _text_rerank_prompt,
        vocabulary_evidence,
    )
except ModuleNotFoundError:  # Direct ``python scripts/accuracy_harness.py`` execution.
    from benchmark_contracts import (  # type: ignore[no-redef]
        ACOUSTIC_JSON_PROMPT,
        acoustic_description as _acoustic_description,
        normalize_music_labels,
        parse_acoustic as _parse_acoustic,
        parse_text_rerank as _parse_text_rerank,
        text_rerank_prompt as _text_rerank_prompt,
        vocabulary_evidence,
    )


def _write(path: Path, result: dict[str, Any]) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    descriptor_owned = True
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor_owned = False
        with stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if descriptor_owned:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def error_result(error: BaseException) -> dict[str, str]:
    """Separate missing local prerequisites from a real execution failure."""
    return {
        "status": "blocked" if isinstance(error, FileNotFoundError) else "failed",
        "error_code": type(error).__name__,
        "reason": str(error),
    }


_SAFE_AGGREGATE_FIELDS = (
    "metrics",
    "checks",
    "coverage",
    "component_coverage",
    "schema_compliance",
    "error_counts",
    "breakdowns",
    "retrieval_scope",
    "compatibility",
)


def _safe_stdout_result(result: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {"status": result["status"]}
    if "error_code" in result:
        safe["error_code"] = result["error_code"]
    for name in _SAFE_AGGREGATE_FIELDS:
        if name in result:
            safe[name] = result[name]
    vocabulary = result.get("vocabulary")
    if isinstance(vocabulary, dict):
        safe["vocabulary"] = {
            name: vocabulary[name]
            for name in ("version", "sha256")
            if name in vocabulary
        }
    return safe


def _has_valid_checkpoint(path: Path) -> bool:
    try:
        checkpoint = json.loads(path.read_text())
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(checkpoint, dict):
        return False
    observations = checkpoint.get("observations")
    completed = checkpoint.get("completed_samples")
    total = checkpoint.get("total_samples")
    return (
        checkpoint.get("status") == "running"
        and isinstance(observations, dict)
        and isinstance(completed, int)
        and not isinstance(completed, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and completed > 0
        and completed == len(observations)
        and completed <= total
    )


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("manifest must be an object")
    return data


def _f1(predicted: set[str], actual: set[str]) -> float:
    if not predicted and not actual:
        return 1.0
    if not predicted or not actual:
        return 0.0
    precision = len(predicted & actual) / len(predicted)
    recall = len(predicted & actual) / len(actual)
    return 2 * precision * recall / (precision + recall)


def _top_ids(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("taxonomy_ids must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("taxonomy_ids must contain non-empty strings")
    values = list(value)
    if len(values) != len(set(values)):
        raise ValueError("taxonomy_ids must not contain duplicates")
    return values[:limit]


@lru_cache(maxsize=1)
def _parents() -> dict[str, str | None]:
    from engine.standards import load_taxonomy

    return {row.id: row.parent_id for row in load_taxonomy()}


def _ancestors(taxonomy_id: str) -> set[str]:
    parents = _parents()
    result = {taxonomy_id}
    current = taxonomy_id
    while current in parents and parents[current] is not None:
        current = parents[current]  # type: ignore[assignment]
        if current in result:
            raise ValueError("taxonomy parent graph contains a cycle")
        result.add(current)
    return result


def hierarchical_match(predicted: str, actual: str) -> bool:
    """Accept the same category or a direct hierarchy relation, never siblings."""
    if predicted not in _parents() or actual not in _parents():
        return False
    return predicted in _ancestors(actual) or actual in _ancestors(predicted)


def _safe_error(component: str, error: BaseException, *, kind: str = "execution") -> dict[str, str]:
    return {"component": component, "code": type(error).__name__, "kind": kind}


def generate_observations(
    samples: list[dict[str, Any]],
    observer: Any,
    *,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    checkpoint: Callable[[dict[str, dict[str, Any]]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Capture freshly generated predictions without mutating private ground truth."""
    observations: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        identifier = sample["id"]
        try:
            result = observer(sample)
            if not isinstance(result, dict):
                raise ValueError("observation runner must return an object")
        except JobCancelled:
            raise
        except Exception as exc:
            result = {
                "status": "failed",
                "errors": [_safe_error("observer", exc)],
            }
        observations[identifier] = result
        if checkpoint is not None:
            checkpoint(dict(observations))
        if on_progress is not None:
            status = result.get("status", "passed")
            if status not in {"passed", "failed", "blocked"}:
                status = "failed"
            on_progress(
                {
                    "sample_index": index,
                    "sample_count": len(samples),
                    "status": status,
                }
            )
    return observations


def _taxonomy_text(row: Any) -> str:
    return "\n".join((f"Category: {row.name}", f"Hierarchy: {' > '.join(row.path)}"))


def _model_call(token: CancellationToken, call: Callable[[], Any]) -> Any:
    token.raise_if_cancelled()
    result = call()
    token.raise_if_cancelled()
    return result


def _component_failure(error: Exception, *, kind: str = "execution") -> dict[str, str]:
    return {"status": "failed", "error_code": type(error).__name__, "error_kind": kind}


class MlxObservationRunner:
    """Generate feasibility observations only from local WAV evidence and resident adapters."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._taxonomy = load_taxonomy()
        self._taxonomy_vectors: Any | None = None
        self._music_indexes = [
            index for index, row in enumerate(self._taxonomy) if row.id == "338" or "Music" in row.path
        ]

    def retrieve(self, texts: list[tuple[str, str]], token: CancellationToken) -> list[dict[str, Any]]:
        """Build the shared benchmark shortlist from audio-derived text only."""
        if not texts:
            return []
        if self._taxonomy_vectors is None:
            self._taxonomy_vectors = _model_call(
                token,
                lambda: self._registry.embedder.encode(
                    [_taxonomy_text(row) for row in self._taxonomy]
                ),
            )
        query_vectors = _model_call(
            token, lambda: self._registry.embedder.encode([text for _, text in texts])
        )
        merged: dict[str, dict[str, Any]] = {}
        for query_index, (source, _text) in enumerate(texts):
            eligible = range(len(self._taxonomy)) if source == "transcript" else self._music_indexes
            similarities = self._taxonomy_vectors @ query_vectors[query_index]
            order = sorted(
                eligible,
                key=lambda index: (-float(similarities[index]), self._taxonomy[index].id),
            )[:15]
            for index in order:
                row = self._taxonomy[index]
                similarity = float(similarities[index])
                current = merged.get(row.id)
                if current is None:
                    merged[row.id] = {
                        "taxonomy_id": row.id,
                        "name": row.name,
                        "path": list(row.path),
                        "similarity": similarity,
                        "provenance": [source],
                    }
                else:
                    current["similarity"] = max(float(current["similarity"]), similarity)
                    if source not in current["provenance"]:
                        current["provenance"].append(source)
        return sorted(
            merged.values(),
            key=lambda item: (-float(item["similarity"]), str(item["taxonomy_id"])),
        )[:15]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        audio_path = Path(sample["audio_path"]).expanduser().resolve(strict=True)
        window = TimeRange(start_seconds=0, end_seconds=30)
        token = CancellationToken()
        components: dict[str, dict[str, str]] = {}
        errors: list[dict[str, str]] = []
        language: str | None = None
        transcript = ""
        acoustic: dict[str, Any] | None = None
        shortlist: list[dict[str, Any]] = []
        reranked: dict[str, Any] | None = None

        try:
            probabilities = _model_call(
                token,
                lambda: self._registry.asr.detect_language_probabilities(audio_path, window, token),
            )
            language = max(probabilities, key=probabilities.__getitem__)
            segments = _model_call(
                token,
                lambda: self._registry.asr.transcribe_window(audio_path, window, language, token),
            )
            transcript = " ".join(segment.text for segment in segments).strip()
            components["asr"] = {"status": "passed"}
        except JobCancelled:
            raise
        except Exception as exc:
            components["asr"] = _component_failure(exc)
            errors.append(_safe_error("asr", exc))

        try:
            raw_acoustic = _model_call(
                token,
                lambda: self._registry.audio_understanding.generate_audio(
                    audio_path,
                    window,
                    ACOUSTIC_JSON_PROMPT,
                    max_tokens=160,
                    token=token,
                ),
            )
        except JobCancelled:
            raise
        except Exception as exc:
            components["acoustic"] = _component_failure(exc)
            errors.append(_safe_error("acoustic", exc))
        else:
            try:
                acoustic = _parse_acoustic(raw_acoustic)
                components["acoustic"] = {"status": "passed"}
            except ValueError as exc:
                components["acoustic"] = _component_failure(exc, kind="schema")
                errors.append(_safe_error("acoustic", exc, kind="schema"))

        audio_texts: list[tuple[str, str]] = []
        if transcript:
            audio_texts.append(("transcript", transcript))
        if acoustic is not None and acoustic["is_music"]:
            audio_texts.append(("acoustic", _acoustic_description(acoustic)))
        try:
            shortlist = self.retrieve(audio_texts, token)
            components["retrieval"] = {"status": "passed"}
        except JobCancelled:
            raise
        except Exception as exc:
            components["retrieval"] = _component_failure(exc)
            errors.append(_safe_error("retrieval", exc))

        if components["retrieval"]["status"] != "passed":
            components["text_rerank"] = {
                "status": "failed",
                "error_code": "RetrievalUnavailable",
                "error_kind": "dependency",
            }
            errors.append(
                {
                    "component": "text_rerank",
                    "code": "RetrievalUnavailable",
                    "kind": "dependency",
                }
            )
        else:
            try:
                if shortlist and audio_texts:
                    prompt = _text_rerank_prompt([text for _, text in audio_texts], shortlist)
                    raw_rerank = _model_call(
                        token,
                        lambda: self._registry.text_generator.generate_text(
                            prompt, max_tokens=320, token=token
                        ),
                    )
                else:
                    reranked = {
                        "taxonomy_ids": [],
                        "keywords": [],
                        "entities": [],
                        "sentiment": {"label": "neutral", "score": 0.0},
                    }
            except JobCancelled:
                raise
            except Exception as exc:
                components["text_rerank"] = _component_failure(exc)
                errors.append(_safe_error("text_rerank", exc))
            else:
                try:
                    if shortlist and audio_texts:
                        reranked = _parse_text_rerank(raw_rerank, shortlist)
                    components["text_rerank"] = {"status": "passed"}
                except ValueError as exc:
                    components["text_rerank"] = _component_failure(
                        exc, kind="schema"
                    )
                    errors.append(_safe_error("text_rerank", exc, kind="schema"))

        return {
            "status": "passed" if all(item["status"] == "passed" for item in components.values()) else "failed",
            "components": components,
            "errors": errors,
            "language": language,
            "transcript": transcript,
            "is_music": acoustic["is_music"] if acoustic is not None else None,
            "vocal_presence": acoustic["vocal_presence"] if acoustic is not None else None,
            "genres": acoustic["genres"] if acoustic is not None else [],
            "moods": acoustic["moods"] if acoustic is not None else [],
            "retrieval_shortlist": shortlist,
            "retrieval_scope": "feasibility_only_not_production_category_acceptance",
            "taxonomy_ids": reranked["taxonomy_ids"] if reranked is not None else [],
            "text_schema": {
                "keywords": reranked["keywords"] if reranked is not None else [],
                "entities": reranked["entities"] if reranked is not None else [],
                "sentiment": reranked["sentiment"] if reranked is not None else None,
            },
        }


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    samples = manifest.get("samples")
    if manifest.get("schema_version") != 1 or not isinstance(samples, list):
        raise ValueError("schema_version=1 and samples are required")
    if len(samples) < 30:
        raise ValueError("labelled manifest must contain at least 30 samples")
    ids: set[str] = set()
    languages: Counter[str] = Counter()
    contents: Counter[str] = Counter()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each sample must be an object")
        identifier = sample.get("id")
        audio_path = sample.get("audio_path")
        language = sample.get("language")
        content_types = sample.get("content_types")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("sample IDs must be unique non-empty strings")
        if language not in {"en", "zh", "ja", "none"}:
            raise ValueError("sample language must be en, zh, ja, or none")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError("sample audio_path must be a non-empty string")
        if not isinstance(content_types, list) or not content_types or any(not isinstance(item, str) for item in content_types):
            raise ValueError("sample content_types must be a non-empty list of strings")
        taxonomy_ids = sample.get("taxonomy_ids")
        if not isinstance(taxonomy_ids, list) or not taxonomy_ids or any(not isinstance(item, str) or not item for item in taxonomy_ids):
            raise ValueError("sample taxonomy_ids must be a non-empty string list")
        if len(taxonomy_ids) != len(set(taxonomy_ids)):
            raise ValueError("sample taxonomy_ids must not contain duplicates")
        if "is_music" in sample and type(sample["is_music"]) is not bool:
            raise ValueError("sample is_music must be a boolean")
        if "vocal_presence" in sample and sample["vocal_presence"] not in {"vocal", "instrumental", "mixed", "unknown"}:
            raise ValueError("sample vocal_presence is invalid")
        for name in ("genres", "moods"):
            if name in sample and (not isinstance(sample[name], list) or any(not isinstance(item, str) or not item for item in sample[name])):
                raise ValueError(f"sample {name} must be a string list")
            if name in sample:
                normalize_music_labels(name, sample[name])
        if "observed" in sample:
            raise ValueError("ground-truth manifest must not contain observed predictions")
        ids.add(identifier)
        if language in {"en", "zh", "ja"} and "speech" in content_types:
            languages[language] += 1
        contents.update(content_types)
    missing = [label for label, count in {**{key: languages[key] for key in ("en", "zh", "ja")}, **{key: contents[key] for key in ("vocal_music", "speech_sparse_music", "instrumental_music")}}.items() if count < 5]
    if missing:
        raise ValueError("manifest lacks required five-sample strata: " + ", ".join(missing))
    return samples


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("metric has no eligible samples")
    return sum(items) / len(items)


def _required_mean(name: str, values: Iterable[float]) -> float:
    try:
        return _mean(values)
    except ValueError as exc:
        raise ValueError(f"no eligible labelled samples for {name}") from exc


def _observed_acoustic(observed: dict[str, Any]) -> tuple[bool, str, list[str], list[str]]:
    is_music = observed.get("is_music")
    vocal = observed.get("vocal_presence")
    genres = observed.get("genres")
    moods = observed.get("moods")
    if type(is_music) is not bool or vocal not in {"vocal", "instrumental", "mixed", "unknown"}:
        raise ValueError("observed acoustic scalar values are invalid")
    try:
        normalized_genres = normalize_music_labels("genres", genres)
        normalized_moods = normalize_music_labels("moods", moods)
    except ValueError as exc:
        raise ValueError("observed acoustic labels are invalid") from exc
    return is_music, vocal, normalized_genres, normalized_moods


_COMPONENTS = ("asr", "acoustic", "retrieval", "text_rerank")


def _retrieval_ids(observed: dict[str, Any]) -> tuple[list[str], bool]:
    value = observed.get("retrieval_shortlist")
    if value is None:
        return _top_ids(observed.get("taxonomy_ids"), 15), True
    if not isinstance(value, list) or len(value) > 15:
        raise ValueError("retrieval_shortlist must be a list of at most 15 candidates")
    ids: list[str] = []
    official = set(_parents())
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("retrieval candidate must be an object")
        if not {"taxonomy_id", "name", "path", "similarity", "provenance"} <= set(item):
            raise ValueError("retrieval candidate has an invalid schema")
        taxonomy_id = item["taxonomy_id"]
        if not isinstance(taxonomy_id, str) or taxonomy_id not in official:
            raise ValueError("retrieval candidate must use an official taxonomy ID")
        if not isinstance(item["name"], str) or not item["name"]:
            raise ValueError("retrieval candidate must have a name")
        if not isinstance(item["path"], list) or any(not isinstance(part, str) or not part for part in item["path"]):
            raise ValueError("retrieval candidate must have a taxonomy path")
        similarity = item["similarity"]
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)) or not math.isfinite(similarity):
            raise ValueError("retrieval candidate must retain a finite similarity")
        provenance = item["provenance"]
        if (
            not isinstance(provenance, list)
            or not provenance
            or any(source not in {"transcript", "acoustic"} for source in provenance)
            or len(provenance) != len(set(provenance))
        ):
            raise ValueError("retrieval candidate provenance is invalid")
        ids.append(taxonomy_id)
    if len(ids) != len(set(ids)):
        raise ValueError("retrieval shortlist must not contain duplicate taxonomy IDs")
    return ids, False


def _reranked_ids(observed: dict[str, Any], shortlist: list[str], *, legacy: bool) -> list[str]:
    ids = _top_ids(observed.get("taxonomy_ids"), 4)
    if len(ids) > 3:
        raise ValueError("reranked taxonomy_ids must contain at most three IDs")
    if any(item not in _parents() for item in ids):
        raise ValueError("reranked taxonomy_ids must be official")
    if not legacy and any(item not in shortlist for item in ids):
        raise ValueError("reranked taxonomy_ids must belong to retrieval shortlist")
    return ids


def _component_declared_passed(observed: dict[str, Any], component: str, *, legacy: bool) -> bool:
    if legacy:
        return True
    components = observed.get("components")
    return (
        isinstance(components, dict)
        and isinstance(components.get(component), dict)
        and components[component].get("status") == "passed"
    )


def _breakdown(
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [row for sample, row in zip(samples, rows) if selector(sample)]
    eligible = len(selected)
    labelled = sum(row["taxonomy_total"] for row in selected)
    return {
        "eligible_samples": eligible,
        "labelled_categories": labelled,
        "retrieval_recall_at_15": sum(row["taxonomy_hits"] for row in selected) / labelled if labelled else 0.0,
        "reranked_top3_precision": sum(row["top3_precision"] for row in selected) / eligible if eligible else 0.0,
        "schema_compliance_rate": sum(row["schema_compliant"] for row in selected) / eligible if eligible else 0.0,
    }


def score(samples: list[dict[str, Any]], observations: dict[str, dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    language_precision: dict[str, list[float]] = {"en": [], "zh": [], "ja": []}
    taxonomy_hits = 0
    taxonomy_total = 0
    top3_precision: list[float] = []
    language_hits: list[float] = []
    music_hits: list[float] = []
    vocal_actual: list[str] = []
    vocal_predicted: list[str] = []
    genre_hits: list[float] = []
    mood_scores: list[float] = []
    component_successes: Counter[str] = Counter()
    component_schema: Counter[str] = Counter()
    error_counts: dict[str, Counter[str]] = {}
    rows: list[dict[str, Any]] = []
    observed_count = 0
    legacy_count = 0
    for sample in samples:
        raw_observed = observations.get(sample["id"])
        observed = raw_observed if isinstance(raw_observed, dict) else {}
        if isinstance(raw_observed, dict):
            observed_count += 1
        legacy = "components" not in observed and observed.get("status") != "failed"
        legacy_count += int(legacy and bool(observed))

        seen_errors: set[tuple[str, str, str]] = set()
        if not isinstance(raw_observed, dict):
            seen_errors.add(("observation", "missing", "MissingObservation"))
        explicit_errors = observed.get("errors", [])
        if isinstance(explicit_errors, list):
            for error in explicit_errors:
                if not isinstance(error, dict):
                    continue
                component = str(error.get("component", "observation"))
                code = str(error.get("code", "UnknownError"))
                kind = str(error.get("kind", "execution"))
                seen_errors.add((component, kind, code))
        components = observed.get("components")
        if isinstance(components, dict):
            for component, outcome in components.items():
                if isinstance(outcome, dict) and outcome.get("status") == "failed":
                    seen_errors.add(
                        (
                            str(component),
                            str(outcome.get("error_kind", "execution")),
                            str(outcome.get("error_code", "UnknownError")),
                        )
                    )
        for component, kind, code in seen_errors:
            error_counts.setdefault(component, Counter())[f"{kind}:{code}"] += 1

        asr_valid = (
            observed.get("language") in {"en", "zh", "ja", "none"}
            and (legacy or isinstance(observed.get("transcript"), str))
        )
        try:
            observed_is_music, observed_vocal, observed_genres, observed_moods = _observed_acoustic(observed)
            acoustic_valid = True
        except ValueError:
            observed_is_music, observed_vocal, observed_genres, observed_moods = None, None, [], []
            acoustic_valid = False
        try:
            predicted15, retrieval_legacy = _retrieval_ids(observed)
            retrieval_valid = True
        except (KeyError, TypeError, ValueError):
            predicted15, retrieval_legacy, retrieval_valid = [], legacy, False
        try:
            predicted3 = _reranked_ids(observed, predicted15, legacy=retrieval_legacy)
            if not retrieval_legacy:
                text_schema = observed.get("text_schema")
                if not isinstance(text_schema, dict):
                    raise ValueError("text rerank diagnostics are missing")
                parsed_rerank = _parse_text_rerank(
                    json.dumps({"taxonomy_ids": predicted3, **text_schema}),
                    observed["retrieval_shortlist"],
                )
                predicted3 = parsed_rerank["taxonomy_ids"]
            rerank_valid = True
        except (KeyError, TypeError, ValueError):
            predicted3, rerank_valid = [], False

        validity = {
            "asr": asr_valid,
            "acoustic": acoustic_valid,
            "retrieval": retrieval_valid,
            "text_rerank": rerank_valid,
        }
        for component in _COMPONENTS:
            declared = _component_declared_passed(observed, component, legacy=legacy)
            component_successes[component] += int(declared and validity[component])
            component_schema[component] += int(validity[component])

        actual_ids = set(sample.get("taxonomy_ids", []))
        sample_taxonomy_hits = sum(any(hierarchical_match(predicted, actual) for predicted in predicted15) for actual in actual_ids)
        taxonomy_hits += sample_taxonomy_hits
        taxonomy_total += len(actual_ids)
        precision = 0.0
        if actual_ids:
            precision = (
                sum(any(hierarchical_match(predicted, actual) for actual in actual_ids) for predicted in predicted3) / len(predicted3)
                if predicted3
                else 0.0
            )
            top3_precision.append(precision)
            if sample["language"] in language_precision and "speech" in sample["content_types"]:
                language_precision[sample["language"]].append(precision)
        if sample["language"] != "none":
            language_hits.append(float(observed.get("language") == sample["language"]))
        if "is_music" in sample:
            music_hits.append(float(observed_is_music == sample["is_music"]))
        if sample.get("is_music") is True and sample.get("vocal_presence") in {"vocal", "instrumental", "mixed"}:
            vocal_actual.append(sample["vocal_presence"])
            vocal_predicted.append(str(observed_vocal) if observed_vocal is not None else "__missing__")
        if sample.get("is_music") is True and sample.get("genres"):
            actual_genres = normalize_music_labels("genres", sample["genres"])
            genre_hits.append(float(bool(set(actual_genres) & set(observed_genres[:3]))))
        if sample.get("is_music") is True and sample.get("moods"):
            actual_moods = normalize_music_labels("moods", sample["moods"])
            mood_scores.append(_f1(set(observed_moods), set(actual_moods)))
        rows.append(
            {
                "taxonomy_hits": sample_taxonomy_hits,
                "taxonomy_total": len(actual_ids),
                "top3_precision": precision,
                "schema_compliant": int(all(validity.values())),
            }
        )
    labels = sorted(set(vocal_actual) | {value for value in vocal_predicted if value != "__missing__"})
    vocal_f1 = _required_mean("vocal presence", (
        _f1(
            {str(index) for index, value in enumerate(vocal_predicted) if value == label},
            {str(index) for index, value in enumerate(vocal_actual) if value == label},
        )
        for label in labels
    ))
    metrics = {
        "taxonomy_labelled_category_recall_at_15": taxonomy_hits / taxonomy_total if taxonomy_total else 0.0,
        "whole_window_top3_hierarchical_precision": _required_mean("whole-window top-3 hierarchical precision", top3_precision),
        "whole_window_top3_hierarchical_precision_by_language": {key: _required_mean(f"{key} language top-3 precision", value) for key, value in language_precision.items()},
        "language_detection_accuracy": _required_mean("language detection", language_hits),
        "music_non_music_accuracy": _required_mean("music/non-music", music_hits),
        "vocal_presence_macro_f1": vocal_f1,
        "genre_top3_accuracy": _required_mean("genre top-3", genre_hits),
        "mood_multilabel_f1": _required_mean("mood", mood_scores),
    }
    checks = {
        "taxonomy_labelled_category_recall_at_15": metrics["taxonomy_labelled_category_recall_at_15"] >= thresholds["taxonomy_labelled_category_recall_at_15"],
        "whole_window_top3_hierarchical_precision": metrics["whole_window_top3_hierarchical_precision"] >= thresholds["whole_window_top3_hierarchical_precision"],
        "language_detection_accuracy": metrics["language_detection_accuracy"] >= thresholds["language_detection_accuracy"],
        "music_non_music_accuracy": metrics["music_non_music_accuracy"] >= thresholds["music_non_music_accuracy"],
        "vocal_presence_macro_f1": metrics["vocal_presence_macro_f1"] >= thresholds["vocal_presence_macro_f1"],
        "genre_top3_accuracy": metrics["genre_top3_accuracy"] >= thresholds["genre_top3_accuracy"],
        "mood_multilabel_f1": metrics["mood_multilabel_f1"] >= thresholds["mood_multilabel_f1"],
        "language_strata": all(value >= thresholds["whole_window_top3_hierarchical_precision_by_language"] for value in metrics["whole_window_top3_hierarchical_precision_by_language"].values()),
        "complete_corpus_coverage": observed_count == len(samples),
        "component_coverage": all(component_successes[name] == len(samples) for name in _COMPONENTS),
        "schema_compliance": all(component_schema[name] == len(samples) for name in _COMPONENTS),
    }
    sample_count = len(samples)
    coverage = {
        "observed_samples": observed_count,
        "eligible_samples": sample_count,
        "rate": observed_count / sample_count if sample_count else 0.0,
    }
    component_coverage = {
        name: {
            "successful_samples": component_successes[name],
            "eligible_samples": sample_count,
            "rate": component_successes[name] / sample_count if sample_count else 0.0,
        }
        for name in _COMPONENTS
    }
    schema_compliance = {
        name: component_schema[name] / sample_count if sample_count else 0.0
        for name in _COMPONENTS
    }
    languages = sorted({str(sample["language"]) for sample in samples})
    content_types = sorted({str(item) for sample in samples for item in sample["content_types"]})
    breakdowns = {
        "language": {
            language: _breakdown(samples, rows, lambda sample, language=language: sample["language"] == language)
            for language in languages
        },
        "content_type": {
            content_type: _breakdown(
                samples,
                rows,
                lambda sample, content_type=content_type: content_type in sample["content_types"],
            )
            for content_type in content_types
        },
    }
    return {
        "metrics": metrics,
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
        "coverage": coverage,
        "component_coverage": component_coverage,
        "schema_compliance": schema_compliance,
        "error_counts": {name: dict(sorted(counts.items())) for name, counts in sorted(error_counts.items())},
        "breakdowns": breakdowns,
        "retrieval_scope": "feasibility_only_not_production_category_acceptance",
        "vocabulary": vocabulary_evidence(),
        "compatibility": {
            "legacy_observations_using_taxonomy_ids_for_retrieval": legacy_count,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("data/models"))
    parser.add_argument("--candidate", choices=CANDIDATES, default="qwen2_audio")
    parser.add_argument("--acceptance", type=Path, default=Path("benchmarks/acceptance.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest-accuracy.json"))
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    persist_result = True
    try:
        if not args.manifest.is_file():
            raise FileNotFoundError(f"private labelled manifest is missing: {args.manifest}")
        thresholds = _load_json(args.acceptance)["thresholds"]
        samples = validate_manifest(_load_json(args.manifest))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        original_connect = socket.socket.connect
        socket.socket.connect = lambda _self, address: (_ for _ in ()).throw(RuntimeError(f"network access is forbidden during accuracy harness: {address!r}"))  # type: ignore[assignment]
        assert original_connect is not socket.socket.connect
        try:
            model_manifest = candidate_manifest_path(args.models_dir, args.candidate)
            if args.candidate == "qwen2_audio":
                registry = ModelRegistry(load_verified_model_paths(model_manifest))
            else:
                registry = ModelRegistry(
                    load_verified_model_paths(
                        model_manifest, candidate=args.candidate
                    ),
                    candidate=args.candidate,
                )

            def checkpoint(observations: dict[str, dict[str, Any]]) -> None:
                _write(
                    args.output,
                    {
                        "status": "running",
                        "candidate": args.candidate,
                        "completed_samples": len(observations),
                        "total_samples": len(samples),
                        "observations": observations,
                        "network_blocked": True,
                        "vocabulary": vocabulary_evidence(),
                    },
                )

            def progress(event: dict[str, Any]) -> None:
                print(json.dumps({"progress": event}, sort_keys=True), flush=True)

            observations = generate_observations(
                samples,
                MlxObservationRunner(registry),
                on_progress=progress,
                checkpoint=checkpoint,
            )
            result = {
                **score(samples, observations, thresholds),
                "candidate": args.candidate,
                "observations": observations,
                "network_blocked": True,
            }
        finally:
            socket.socket.connect = original_connect  # type: ignore[assignment]
    except (FileNotFoundError, ImportError, RuntimeError, OSError, subprocess.CalledProcessError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result = error_result(exc)
        result["candidate"] = args.candidate
        persist_result = not _has_valid_checkpoint(args.output)
    if persist_result:
        _write(args.output, result)
    print(json.dumps(_safe_stdout_result(result), sort_keys=True), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
