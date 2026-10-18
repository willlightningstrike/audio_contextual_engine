"""Validate/freeze experimental non-blind v2 inputs; never run inference.

All media paths are relative to an explicitly supplied source root. Frozen
references are inputs; predictions and post-result corrections belong in the
later evaluation report. Historical v1 code and artifacts are not consumed as
v2 protocols. Per-attempt dynamic verifier prompt hashes belong in run journals.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import wave


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/acceptance-english-transcript-only-v2.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SAMPLE_TOLERANCE = 1 / 16000 + 1e-9
IDENTITY = {
    "protocol_id": "english-audio-transcript-only-v2", "schema_version": 1,
    "non_blind": True, "cattax": 7, "evidence_mode": "transcript_only",
    "verifier_policy": "english-transcript-verifier-v1",
    "whole_file_policy": "english-audio-aggregation-v2",
}
TOP_KEYS = set(IDENTITY) | {
    "code_commit", "text_model", "verifier_source_sha256", "taxonomy",
    "inventory_ids", "cases", "missing_inputs", "canonical_sha256",
}
CASE_KEYS = {
    "case_id", "source_path", "source_sha256", "audio_stream_index",
    "selected_audio_path", "selected_audio_sha256", "duration_seconds",
    "coverage_scope", "source_range", "selection_provenance", "non_blind",
}
ALIAS_KEYS = {"source_path", "source_sha256", "audio_stream_index",
              "source_range", "selection_provenance"}
NORMALIZATIONS = {"pcm_or_ffmpeg_input_seek_v1", "ffmpeg_output_seek_v1",
                  "ffmpeg_output_seek_async_v1", "ffmpeg_full_async_pad_v1"}
ROUTING = {"english_speech", "english_singing", "instrumental_out_of_scope",
           "other_sound_out_of_scope", "silence_out_of_scope",
           "non_english_out_of_scope", "unknown_or_uncertain"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_digest(manifest: dict) -> str:
    payload = {k: v for k, v in manifest.items() if k != "canonical_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _keys(value, required, optional=(), *, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing, extra = required - value.keys(), value.keys() - required - set(optional)
    if missing or extra:
        raise ValueError(f"{label}: missing {sorted(missing)}, unsupported {sorted(extra)}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _number(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {label}")


def confined_path(source_root: Path, relative: str) -> Path:
    _text(relative, "path")
    path = Path(relative)
    root = source_root.resolve(strict=True)
    try:
        resolved = (root / path).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("path must resolve within source_root") from exc
    if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root):
        raise ValueError("path must be relative and resolve within source_root")
    return resolved


def _span(value):
    if value is None:
        return None
    _keys(value, {"start_seconds", "end_seconds"}, label="source_range")
    start, end = value["start_seconds"], value["end_seconds"]
    _number(start, "source_range start")
    _number(end, "source_range end", positive=True)
    if end <= start:
        raise ValueError("source_range must increase")
    return start, end


def source_streams(path: Path) -> dict[int, float]:
    """Probe audio metadata only; no frames, text, or model inference."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as stream:
                return {0: stream.getnframes() / stream.getframerate()}
        except wave.Error:
            pass
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index,duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return {row["index"]: float(row["duration"])
            for row in json.loads(result.stdout)["streams"]}


def pcm_identity(path: Path) -> tuple[float, str]:
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                stream.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("selected_audio must be normalized mono 16 kHz PCM16")
        duration = stream.getnframes() / 16000
        digest = hashlib.sha256()
        while block := stream.readframes(65536):
            digest.update(block)
        return duration, digest.hexdigest()


def normalized_identity(path: Path, index: int, span, source_duration: float,
                        method="pcm_or_ffmpeg_input_seek_v1") -> tuple[float, str]:
    """Reproduce selected bytes, without persisting or overwriting audio files."""
    if not isinstance(method, str) or method not in NORMALIZATIONS:
        raise ValueError("unsupported normalization")
    if method == "pcm_or_ffmpeg_input_seek_v1" and path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as stream:
                if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                        stream.getcomptype()) == (1, 2, 16000, "NONE"):
                    start, end = span or (0, stream.getnframes() / 16000)
                    first, last = round(start * 16000), round(end * 16000)
                    stream.setpos(first)
                    remaining, digest = last - first, hashlib.sha256()
                    while remaining:
                        block = stream.readframes(min(remaining, 65536))
                        if not block:
                            raise ValueError("source PCM interval truncated")
                        remaining -= len(block) // 2
                        digest.update(block)
                    return (last - first) / 16000, digest.hexdigest()
        except wave.Error:
            pass
    command = ["ffmpeg", "-nostdin", "-v", "error"]
    if span is not None and method == "pcm_or_ffmpeg_input_seek_v1":
        command += ["-ss", str(span[0])]
    command += ["-i", str(path)]
    if span is not None:
        if method != "pcm_or_ffmpeg_input_seek_v1":
            command += ["-ss", str(span[0])]
        command += ["-t", str(span[1] - span[0])]
    command += ["-map", f"0:{index}", "-vn"]
    if method == "ffmpeg_full_async_pad_v1":
        if span is not None:
            raise ValueError("complete-stream normalization cannot select a range")
        command += ["-af", f"aresample=async=1:first_pts=0,apad,atrim=end={source_duration}"]
    elif method == "ffmpeg_output_seek_async_v1":
        command += ["-af", "aresample=async=1:first_pts=0"]
    command += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    digest, count = hashlib.sha256(), 0
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as process:
            while block := process.stdout.read(65536):
                count += len(block)
                digest.update(block)
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError("source PCM normalization failed: " + errors.read().decode(errors="replace"))
    return count / 32000, digest.hexdigest()


def validate_confirmation(label: dict, *, source_root: Path, pcm_hash: str,
                          duration: float, routing=False):
    evidence = label["routing_confirmation"] if routing else label["confirmation_source"]
    _keys(evidence, {"path", "sha256", "record_id", "confirmation_stage", "audio_path",
        "audio_sha256", "audio_stream_index", "source_range"}, {"audio_record"}, label="confirmation_source")
    path = confined_path(source_root, evidence["path"])
    if evidence["sha256"] != sha256(path):
        raise ValueError("confirmation evidence sha256 mismatch")
    document = json.loads(path.read_text())
    # Inspect content, not the caller's filename or claimed stage. The real
    # paired-context file has this marker in scope, even when renamed.
    text = json.dumps(document, ensure_ascii=False).lower()
    if any(marker in text for marker in ("post-result", "post_result", "post result")):
        raise ValueError("post-result evidence cannot be a frozen reference")
    def record_in(value, record_id):
        matches = [row for row in value.get("cases", []) if row.get("id") == record_id]
        if len(matches) != 1:
            raise ValueError("confirmation evidence record_id must identify exactly one record")
        return matches[0]

    record_id = evidence["record_id"]
    _text(record_id, "confirmation record_id")
    if document.get("status") == "frozen_before_inference" and document.get("approval_context"):
        if evidence["confirmation_stage"] != "before_inference":
            raise ValueError("confirmation evidence stage mismatch")
        record = record_in(document, record_id)
        if routing:
            confirmed_value = record.get("disposition")
        else:
            values = ([record.get("primary_category")] if label["role"] == "primary"
                      else record.get("secondary_categories") or [])
            if label["category_id"] not in values:
                raise ValueError("confirmation evidence does not confirm this category and role")
    elif routing and document.get("source") == "User listening confirmation before validation inference":
        if evidence["confirmation_stage"] != "before_inference":
            raise ValueError("confirmation evidence stage mismatch")
        confirmed_value = document.get("labels", {}).get(record_id)
        companion = evidence.get("audio_record")
        _keys(companion, {"path", "sha256", "record_id"}, label="confirmation audio_record")
        companion_path = confined_path(source_root, companion["path"])
        if sha256(companion_path) != companion["sha256"] or companion["record_id"] != record_id:
            raise ValueError("confirmation companion identity mismatch")
        record = record_in(json.loads(companion_path.read_text()), record_id)
    elif routing and evidence["confirmation_stage"] == "user_listening_confirmation":
        record = record_in(document, record_id)
        if (record.get("label_source") != "user listening confirmation"
                and document.get("label_source") != "user listening confirmation in current conversation"):
            raise ValueError("confirmation evidence lacks user listening confirmation")
        confirmed_value = record.get("label_detail") or record.get("label")
    else:
        raise ValueError("confirmation evidence must record approval before inference or user listening confirmation")
    if routing:
        routing_values = {
            "non_english_out_of_scope": "non_english_out_of_scope",
            "instrumental without vocals": "instrumental_out_of_scope",
            "instrumental music, no vocals": "instrumental_out_of_scope",
            "instrumental music without vocals": "instrumental_out_of_scope",
            "instrumental music": "instrumental_out_of_scope",
            "other sounds without vocals": "other_sound_out_of_scope",
            "item-moving sounds only": "other_sound_out_of_scope",
            "silence": "silence_out_of_scope",
        }
        if routing_values.get(confirmed_value) != label["expected_routing"]:
            raise ValueError("routing confirmation evidence does not confirm this label")
    audio = confined_path(source_root, evidence["audio_path"])
    if "audio_path" in record:
        recorded = Path(record["audio_path"])
        if recorded.is_absolute():
            if not recorded.resolve().is_relative_to(source_root.resolve()):
                raise ValueError("confirmation recorded audio must remain within source_root")
            recorded = recorded.relative_to(source_root.resolve())
        recorded = confined_path(source_root, str(recorded))
        if sha256(recorded) != evidence["audio_sha256"]:
            raise ValueError("confirmation recorded audio_path hash mismatch")
    # SAM confirmation has an explicit source file rather than a saved hash.
    # Resolve that exact file against its evidence directory, then hash it now.
    if "file" in record:
        recorded_audio = (path.parent / record["file"]).resolve()
        if not recorded_audio.is_relative_to(source_root.resolve()) or recorded_audio != audio:
            raise ValueError("confirmation evidence source file mismatch")
        record = {**record, "audio_sha256": sha256(recorded_audio)}
    if record.get("audio_sha256") != evidence["audio_sha256"]:
        raise ValueError("confirmation audio identity differs from evidence record")
    if sha256(audio) != evidence["audio_sha256"]:
        raise ValueError("confirmation audio_sha256 mismatch")
    streams = source_streams(audio)
    index = evidence["audio_stream_index"]
    if type(index) is not int or index not in streams:
        raise ValueError("confirmation audio_stream_index invalid")
    # Existing prelabel evidence hashes the complete standalone WAV. It does
    # not authorize relabeling an arbitrary interval of that WAV.
    if evidence["source_range"] is not None:
        raise ValueError("confirmation range is not supported by evidence record")
    bound_duration, bound_pcm = normalized_identity(audio, index, None, streams[index])
    if abs(bound_duration - duration) > SAMPLE_TOLERANCE or bound_pcm != pcm_hash:
        raise ValueError("confirmation source PCM is not the exact selected clip")


def validate_protocol(manifest: object, *, source_root: Path) -> tuple[dict, ...]:
    """Check strict schema, local identities, complete inventory and media bytes."""
    _keys(manifest, TOP_KEYS, label="protocol")
    for field, expected in IDENTITY.items():
        if type(manifest[field]) is not type(expected) or manifest[field] != expected:
            raise ValueError(f"incorrect {field}")
    if not isinstance(manifest["code_commit"], str) or not COMMIT.fullmatch(manifest["code_commit"]):
        raise ValueError("invalid code_commit")
    _keys(manifest["text_model"], {"repository", "revision", "manifest_sha256"}, label="text_model")
    expected_model = {
        "repository": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "revision": "2938092373e5f97b95538884112085364c2da315",
        "manifest_sha256": "70769ae1399cfa9dab1e5e43a866c59afbdc18eaea2d00cdff302d9f9fdd4efe",
    }
    for field, value in expected_model.items():
        if manifest["text_model"][field] != value:
            raise ValueError(f"incorrect text_model {field}")
    if manifest["verifier_source_sha256"] != sha256(ROOT / "engine/signals/topic_verify.py"):
        raise ValueError("incorrect verifier_source_sha256")
    _keys(manifest["taxonomy"], {"version", "sha256"}, label="taxonomy")
    if manifest["taxonomy"]["version"] != "IAB Content Taxonomy 3.0":
        raise ValueError("incorrect taxonomy version")
    if manifest["taxonomy"]["sha256"] != sha256(ROOT / "data/standards/content-taxonomy-3.0.tsv"):
        raise ValueError("incorrect taxonomy sha256")
    if not isinstance(manifest["canonical_sha256"], str) or not SHA256.fullmatch(manifest["canonical_sha256"]):
        raise ValueError("invalid canonical_sha256")
    if manifest["canonical_sha256"] != canonical_digest(manifest):
        raise ValueError("canonical_sha256 mismatch: frozen inputs changed")

    inventory, cases, missing = (manifest[k] for k in ("inventory_ids", "cases", "missing_inputs"))
    if not isinstance(inventory, list) or not inventory or any(not isinstance(x, str) or not x for x in inventory):
        raise ValueError("inventory_ids must contain nonempty IDs")
    if not isinstance(cases, list) or not cases or not isinstance(missing, list):
        raise ValueError("inventory requires cases and missing_inputs arrays")
    if len(set(inventory)) != len(inventory):
        raise ValueError("duplicate inventory ID")
    ids = [row.get("case_id") if isinstance(row, dict) else None for row in cases + missing]
    if any(not isinstance(x, str) or not x for x in ids) or Counter(ids) != Counter(inventory):
        raise ValueError("inventory must reconcile exactly once to cases plus missing_inputs")

    hashes, streams, pcms, bindings = {}, {}, {}, {}
    def source(row):
        path = confined_path(Path(source_root), row["source_path"])
        declared = row["source_sha256"]
        if not isinstance(declared, str) or not SHA256.fullmatch(declared):
            raise ValueError("invalid source_sha256")
        if path not in hashes:
            hashes[path] = sha256(path)
        if hashes[path] != declared:
            raise ValueError("source_sha256 mismatch")
        if path not in streams:
            streams[path] = source_streams(path)
        index = row["audio_stream_index"]
        if type(index) is not int or index < 0 or index not in streams[path]:
            raise ValueError("invalid audio_stream_index")
        _text(row["selection_provenance"], "selection_provenance")
        return streams[path][index]

    def bind(row, source_duration, pcm_hash, duration, label):
        path = confined_path(Path(source_root), row["source_path"])
        span = _span(row["source_range"])
        method = row.get("normalization", "pcm_or_ffmpeg_input_seek_v1")
        key = path, row["audio_stream_index"], span, method
        if key not in bindings:
            bindings[key] = normalized_identity(path, row["audio_stream_index"], span, source_duration, method)
        bound_duration, bound_hash = bindings[key]
        if abs(bound_duration - duration) > SAMPLE_TOLERANCE or bound_hash != pcm_hash:
            raise ValueError(f"{label} source PCM is not byte-identical: {row['source_path']}")

    for row in missing:
        _keys(row, {"case_id", "source_path", "reason"}, {"expected_source_range"}, label="missing_input")
        confined_path(Path(source_root), row["source_path"])
        _text(row["reason"], "missing_input reason")
        # Historical endpoints may be unknown; never fill them from filenames.
        span = row.get("expected_source_range")
        if span is not None:
            _keys(span, {"start_seconds", "end_seconds"}, label="expected_source_range")
            for key, value in span.items():
                if value is not None:
                    _number(value, "expected_source_range " + key)
            if all(value is not None for value in span.values()):
                _span(span)

    for row in cases:
        _keys(row, CASE_KEYS, {"source_aliases", "expected_routing", "confirmed_reference_labels", "normalization", "routing_confirmation", "unverified_source_claims"}, label="case")
        if row["non_blind"] is not True:
            raise ValueError("case non_blind must be true")
        source_duration = source(row)
        audio = confined_path(Path(source_root), row["selected_audio_path"])
        declared = row["selected_audio_sha256"]
        if not isinstance(declared, str) or not SHA256.fullmatch(declared) or sha256(audio) != declared:
            raise ValueError("selected_audio_sha256 mismatch or malformed")
        duration, pcm_hash = pcm_identity(audio)
        _number(row["duration_seconds"], "duration_seconds", positive=True)
        if abs(duration - row["duration_seconds"]) > SAMPLE_TOLERANCE:
            raise ValueError("selected_audio duration mismatch")
        if pcm_hash in pcms:
            raise ValueError(f"duplicate PCM: use source_aliases for {pcms[pcm_hash]}")
        pcms[pcm_hash] = row["case_id"]
        scope, span = row["coverage_scope"], _span(row["source_range"])
        if scope not in {"full_track", "exact_excerpt", "unknown_extent"}:
            raise ValueError("invalid coverage_scope")
        if scope == "exact_excerpt":
            if span is None or span[1] > source_duration + SAMPLE_TOLERANCE or abs(span[1] - span[0] - duration) > SAMPLE_TOLERANCE:
                raise ValueError("source_range exceeds source or differs from selected duration")
        elif span is not None:
            raise ValueError("source_range must be null for full_track or unknown_extent")
        if scope == "full_track" and abs(source_duration - duration) > SAMPLE_TOLERANCE:
            raise ValueError("full_track must cover complete selected source stream")
        bind(row, source_duration, pcm_hash, duration, "selected")
        claims = row.get("unverified_source_claims", [])
        if not isinstance(claims, list):
            raise ValueError("unverified_source_claims must be an array")
        for claim in claims:
            _keys(claim, ALIAS_KEYS | {"reason"}, label="unverified_source_claim")
            claim_duration = source(claim)
            claim_span = _span(claim["source_range"])
            if claim_span is not None and claim_span[1] > claim_duration + SAMPLE_TOLERANCE:
                raise ValueError("unverified claim range exceeds source")
            _text(claim["reason"], "unverified source reason")
        aliases = row.get("source_aliases", [])
        if not isinstance(aliases, list):
            raise ValueError("source_aliases must be an array")
        for alias in aliases:
            _keys(alias, ALIAS_KEYS, {"normalization"}, label="source_alias")
            alias_duration = source(alias)
            alias_span = _span(alias["source_range"])
            if alias_span is not None and (alias_span[1] > alias_duration + SAMPLE_TOLERANCE or abs(alias_span[1] - alias_span[0] - duration) > SAMPLE_TOLERANCE):
                raise ValueError("alias source_range differs from selected duration")
            bind(alias, alias_duration, pcm_hash, duration, "alias")
        if "expected_routing" in row and row["expected_routing"] not in ROUTING:
            raise ValueError("invalid expected_routing")
        if ("expected_routing" in row) != ("routing_confirmation" in row):
            raise ValueError("routing expectation requires routing_confirmation evidence")
        if "expected_routing" in row:
            validate_confirmation(row, source_root=Path(source_root), pcm_hash=pcm_hash, duration=duration, routing=True)
        labels = row.get("confirmed_reference_labels", [])
        if not isinstance(labels, list):
            raise ValueError("confirmed_reference_labels must be an array")
        for label in labels:
            _keys(label, {"category_id", "role", "confirmation_source"}, label="reference label")
            _text(label["category_id"], "category_id")
            if label["role"] not in {"primary", "secondary"}:
                raise ValueError("invalid reference role")
            validate_confirmation(label, source_root=Path(source_root), pcm_hash=pcm_hash, duration=duration)
    return tuple(cases)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="new directory for freeze identity only; no evaluation")
    args = parser.parse_args(argv)
    try:
        validator_digest = sha256(Path(__file__))
        raw = args.manifest.read_bytes()
        manifest = json.loads(raw)
        cases = validate_protocol(manifest, source_root=args.source_root)
        record = {
            "status": "validated_experimental_non_blind_no_inference",
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_sha256": manifest["canonical_sha256"],
            "validator_sha256": validator_digest,
            "code_commit": manifest["code_commit"],
            "source_root": str(args.source_root.resolve()),
            "available_cases": len(cases), "missing_inputs": len(manifest["missing_inputs"]),
            "unverified_source_claims": sum(len(row.get("unverified_source_claims", [])) for row in cases),
            "normalizer": {
                "ffmpeg_path": str(Path(shutil.which("ffmpeg")).resolve(strict=True)),
                "ffmpeg_sha256": sha256(Path(shutil.which("ffmpeg"))),
                "ffmpeg_version": subprocess.run(["ffmpeg", "-version"],
                    capture_output=True, text=True, check=True).stdout.strip(),
                "binding": "exact raw mono16k PCM16 hash; declared per-source recipe; WAV intervals use rounded sample indices",
            },
        }
        if sha256(Path(__file__)) != validator_digest:
            raise ValueError("validator changed during freeze; rerun against stable code")
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            with (args.output_dir / "protocol-freeze.json").open("x") as stream:
                json.dump(record, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
        print(json.dumps(record, ensure_ascii=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, wave.Error, subprocess.SubprocessError) as exc:
        parser.exit(1, f"protocol validation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
