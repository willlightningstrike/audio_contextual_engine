"""Pure, model-free contracts shared by the audio feasibility benchmark."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from engine.models.response import canonical_json_container
from engine.standards import load_taxonomy


VOCABULARY_VERSION = "2026-09-07.v1"
GENRES = (
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
)
GENRE_ALIASES = {
    "hip-hop": "hip hop",
    "hiphop": "hip hop",
    "r and b": "r&b",
    "rnb": "r&b",
    "sound track": "soundtrack",
}
MOODS = (
    "calm",
    "dramatic",
    "emotional",
    "energetic",
    "happy",
    "hopeful",
    "melancholic",
    "romantic",
    "sad",
    "tense",
    "upbeat",
    "uplifting",
)
MOOD_ALIASES: dict[str, str] = {}


def _vocabulary_payload() -> dict[str, Any]:
    return {
        "genres": list(GENRES),
        "genre_aliases": GENRE_ALIASES,
        "moods": list(MOODS),
        "mood_aliases": MOOD_ALIASES,
    }


def vocabulary_evidence() -> dict[str, Any]:
    payload = _vocabulary_payload()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": VOCABULARY_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        **payload,
    }


def normalize_music_labels(kind: str, values: object) -> list[str]:
    if kind not in {"genres", "moods"}:
        raise ValueError("music label kind must be genres or moods")
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{kind} must be a non-empty string list or an empty list")
    vocabulary = set(GENRES if kind == "genres" else MOODS)
    aliases = GENRE_ALIASES if kind == "genres" else MOOD_ALIASES
    normalized: list[str] = []
    for value in values:
        label = " ".join(value.strip().lower().split())
        label = aliases.get(label, label)
        if label not in vocabulary:
            raise ValueError(f"unsupported {kind[:-1]} label: {value}")
        if label in normalized:
            raise ValueError(f"{kind} must not contain duplicates after normalization")
        normalized.append(label)
    return normalized


ACOUSTIC_JSON_PROMPT = (
    "Analyse only this audio. Output one JSON object and nothing else, with exactly "
    "these four keys and no others: {\"is_music\": false, "
    "\"vocal_presence\": \"unknown\", \"genres\": [], \"moods\": []}. "
    "Replace the example values based on the audio. is_music must be exactly "
    "the unquoted JSON boolean true or the unquoted JSON boolean false; never quote it "
    "and never write yes/no. vocal_presence must be exactly vocal, instrumental, "
    "mixed, or unknown. genres must use only these canonical labels: "
    + ", ".join(GENRES)
    + ". moods must use only these canonical labels: "
    + ", ".join(MOODS)
    + ". genres and moods must contain only strings. "
    "The first response character must be { and the last must be }. "
    "Use unescaped ASCII double quotation marks around every key and string; "
    "never backslash-escape quotation marks, never wrap the object in a string, "
    "and never use a code fence."
)


def parse_acoustic(value: str) -> dict[str, Any]:
    try:
        result = json.loads(canonical_json_container(value))
    except ValueError as exc:
        raise ValueError("text/audio model acoustic response is not JSON") from exc
    if not isinstance(result, dict) or set(result) != {
        "is_music",
        "vocal_presence",
        "genres",
        "moods",
    }:
        raise ValueError("text/audio model acoustic response has an invalid schema")
    if (
        not isinstance(result["is_music"], bool)
        or result["vocal_presence"]
        not in {"vocal", "instrumental", "mixed", "unknown"}
    ):
        raise ValueError("text/audio model acoustic response has invalid scalar values")
    try:
        result["genres"] = normalize_music_labels("genres", result["genres"])
        result["moods"] = normalize_music_labels("moods", result["moods"])
    except ValueError as exc:
        raise ValueError("text/audio model acoustic response has invalid labels") from exc
    return result


def _validate_probability(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise ValueError(f"text rerank {name} must be between zero and one")
    return float(value)


@lru_cache(maxsize=1)
def _official_taxonomy_ids() -> frozenset[str]:
    return frozenset(row.id for row in load_taxonomy())


def parse_text_rerank(value: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        result = json.loads(canonical_json_container(value))
    except ValueError as exc:
        raise ValueError("text/audio model text rerank response is not JSON") from exc
    if not isinstance(result, dict) or set(result) != {
        "taxonomy_ids",
        "keywords",
        "entities",
        "sentiment",
    }:
        raise ValueError("text/audio model text rerank response has an invalid schema")
    taxonomy_ids = result["taxonomy_ids"]
    if not isinstance(taxonomy_ids, list) or any(
        not isinstance(item, str) or not item for item in taxonomy_ids
    ):
        raise ValueError("text rerank taxonomy_ids must be a string list")
    if len(taxonomy_ids) > 3:
        raise ValueError("text rerank taxonomy_ids must contain at most three IDs")
    if len(taxonomy_ids) != len(set(taxonomy_ids)):
        raise ValueError("text rerank taxonomy_ids must not contain duplicates")
    if any(item not in _official_taxonomy_ids() for item in taxonomy_ids):
        raise ValueError("text rerank taxonomy_ids must be official taxonomy IDs")
    shortlist = {str(candidate.get("taxonomy_id")) for candidate in candidates}
    if any(item not in shortlist for item in taxonomy_ids):
        raise ValueError("text rerank taxonomy_ids must belong to the supplied shortlist")

    keywords = result["keywords"]
    if not isinstance(keywords, list):
        raise ValueError("text rerank keywords must be a list")
    seen_keywords: set[str] = set()
    for keyword in keywords:
        if not isinstance(keyword, dict) or set(keyword) != {"term", "confidence"}:
            raise ValueError("text rerank keyword has an invalid schema")
        term = keyword["term"]
        if (
            not isinstance(term, str)
            or not term.strip()
            or term.casefold() in seen_keywords
        ):
            raise ValueError("text rerank keywords must be unique non-empty strings")
        seen_keywords.add(term.casefold())
        keyword["confidence"] = _validate_probability(
            keyword["confidence"], "keyword confidence"
        )

    entities = result["entities"]
    if not isinstance(entities, list):
        raise ValueError("text rerank entities must be a list")
    for entity in entities:
        if not isinstance(entity, dict) or set(entity) != {
            "text",
            "type",
            "mention_count",
        }:
            raise ValueError("text rerank entity has an invalid schema")
        if any(
            not isinstance(entity[key], str) or not entity[key].strip()
            for key in ("text", "type")
        ):
            raise ValueError("text rerank entity strings must be non-empty")
        if (
            isinstance(entity["mention_count"], bool)
            or not isinstance(entity["mention_count"], int)
            or entity["mention_count"] < 1
        ):
            raise ValueError(
                "text rerank entity mention_count must be a positive integer"
            )

    sentiment = result["sentiment"]
    if not isinstance(sentiment, dict) or set(sentiment) != {"label", "score"}:
        raise ValueError("text rerank sentiment has an invalid schema")
    if sentiment["label"] not in {"negative", "neutral", "positive"}:
        raise ValueError("text rerank sentiment label is invalid")
    score = sentiment["score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not -1 <= score <= 1
    ):
        raise ValueError(
            "text rerank sentiment score must be between minus one and one"
        )
    sentiment["score"] = float(score)
    return result


def acoustic_description(acoustic: dict[str, Any]) -> str:
    return (
        f"Music. Vocal presence: {acoustic['vocal_presence']}. "
        f"Genres: {', '.join(acoustic['genres']) or 'none'}. "
        f"Moods: {', '.join(acoustic['moods']) or 'none'}."
    )


def text_rerank_prompt(
    audio_texts: list[str], candidates: list[dict[str, Any]]
) -> str:
    safe_candidates = [
        {"id": item["taxonomy_id"], "name": item["name"], "path": item["path"]}
        for item in candidates
    ]
    return (
        "Use only the audio-derived text and supplied taxonomy candidates below. "
        "Select up to three relevant candidate IDs. Do not add IDs or facts from outside this input. "
        "Output one JSON object and nothing else with exactly these keys: taxonomy_ids, keywords, "
        "entities, sentiment. taxonomy_ids is a unique list of zero to three IDs. keywords is a list "
        "of objects with term and confidence. entities is a list of objects with text, type, and "
        "mention_count. sentiment has label (negative, neutral, or positive) and score (-1 to 1).\n"
        f"Audio-derived text: {json.dumps(audio_texts, ensure_ascii=False)}\n"
        f"Candidates: {json.dumps(safe_candidates, ensure_ascii=False, sort_keys=True)}"
    )
