"""Behavioral contracts for the separate, model-free transcript-only freeze."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/transcript_only_acceptance.py"
# Literal content of the real paired-context-v1/label-review/confirmed-labels.json.
# Keep this regression portable: ignored local benchmark files are not test dependencies.
PAIRED_CONTEXT_LABELS = {
    "scope": "User-confirmed labels for exact 0-90 second excerpts; post-result adjudication, not blind held-out ground truth",
    "confirmation": "both confirmed",
    "cases": [
        {"id": "P01", "audio_sha256": "452718f9c4693d064cae2e2df16ecf8dc00902cf7d82e3bf2e9ad7d5a35c9557",
         "primary_category": "268", "primary_name": "Video Production", "secondary_categories": ["606"],
         "secondary_names": ["Video Software"], "review_status": "user_confirmed"},
        {"id": "P02", "audio_sha256": "281bb09c1c2df34a67b44941d049e2b0eaaedd27dcda5d24db0a3b3184826fd4",
         "primary_category": "601", "primary_name": "Computer Peripherals", "secondary_categories": [],
         "secondary_names": [], "review_status": "user_confirmed"},
    ],
    "does_not_confirm": ["30-second condition labels", "transcript word accuracy", "keywords", "entities",
                         "sentiment", "production readiness"],
}


def validator():
    # A missing implementation is an explicit RED assertion, not an import error.
    assert SCRIPT.is_file(), "transcript-only protocol validator is missing"
    spec = importlib.util.spec_from_file_location("transcript_only_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(value):
    return hashlib.sha256(json.dumps(
        {k: v for k, v in value.items() if k != "canonical_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()).hexdigest()


def seal(value):
    value["canonical_sha256"] = digest(value)
    return value


def wav(path, seconds=2, sample=1):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(sample.to_bytes(2, "little", signed=True) * (seconds * 16000))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def protocol(tmp_path):
    wav(tmp_path / "source.wav")
    wav(tmp_path / "selected.wav")
    return seal({
        "protocol_id": "english-audio-transcript-only-v2", "schema_version": 1,
        "non_blind": True, "cattax": 7, "evidence_mode": "transcript_only",
        "code_commit": "dfbb98de8df50265eea6229b35479150aa3985a6",
        "text_model": {
            "repository": "mlx-community/Qwen2.5-32B-Instruct-4bit",
            "revision": "2938092373e5f97b95538884112085364c2da315",
            "manifest_sha256": "70769ae1399cfa9dab1e5e43a866c59afbdc18eaea2d00cdff302d9f9fdd4efe",
        },
        "verifier_policy": "english-transcript-verifier-v1",
        "verifier_source_sha256": sha(ROOT / "engine/signals/topic_verify.py"),
        "taxonomy": {"version": "IAB Content Taxonomy 3.0",
                     "sha256": sha(ROOT / "data/standards/content-taxonomy-3.0.tsv")},
        "whole_file_policy": "english-audio-aggregation-v2",
        "inventory_ids": ["literal-two-second-control", "unavailable-historical-input"],
        "cases": [{
            "case_id": "literal-two-second-control", "source_path": "source.wav",
            "source_sha256": sha(tmp_path / "source.wav"), "audio_stream_index": 0,
            "selected_audio_path": "selected.wav",
            "selected_audio_sha256": sha(tmp_path / "selected.wav"),
            "duration_seconds": 2, "coverage_scope": "full_track", "source_range": None,
            "selection_provenance": "Literal PCM fixture, complete selected stream; no routing label.",
            "non_blind": True,
        }],
        "missing_inputs": [{"case_id": "unavailable-historical-input",
                            "source_path": "not-available.wav",
                            "expected_source_range": None,
                            "reason": "Historical source unavailable; original extent unknown."}],
    })


def test_accepts_complete_pcm_and_explicit_missing_inventory(protocol, tmp_path):
    rows = validator().validate_protocol(protocol, source_root=tmp_path)
    assert [r["case_id"] for r in rows] == ["literal-two-second-control"]


def test_canonical_digest_matches_hand_checked_json():
    # SHA256 of the literal UTF-8 bytes {"a":1}; digest field is excluded.
    assert validator().canonical_digest({"a": 1, "canonical_sha256": "ignored"}) == (
        "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    )


@pytest.mark.parametrize("field,value", [
    ("protocol_id", "english-phase1-acceptance-v1"), ("schema_version", True),
    ("non_blind", False), ("cattax", 8), ("evidence_mode", "dual_evidence"),
    ("code_commit", "dirty"), ("verifier_policy", "unversioned"),
    ("whole_file_policy", "english-audio-aggregation-v1"),
    ("verifier_source_sha256", "0" * 64),
])
def test_rejects_wrong_protocol_or_runtime_identity(protocol, tmp_path, field, value):
    protocol[field] = value
    with pytest.raises(ValueError, match=field):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


@pytest.mark.parametrize("section,field,value", [
    ("text_model", "repository", "other/model"),
    ("text_model", "revision", "0" * 40),
    ("text_model", "manifest_sha256", "0" * 64),
    ("taxonomy", "version", "3.1"), ("taxonomy", "sha256", "0" * 64),
])
def test_rejects_changed_model_and_taxonomy(protocol, tmp_path, section, field, value):
    protocol[section][field] = value
    with pytest.raises(ValueError, match=field):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


@pytest.mark.parametrize("field", ["source_sha256", "selected_audio_sha256"])
def test_requires_and_verifies_both_file_hashes(protocol, tmp_path, field):
    protocol["cases"][0][field] = "0" * 64
    with pytest.raises(ValueError, match=field):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)
    del protocol["cases"][0][field]
    with pytest.raises(ValueError, match=field):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


@pytest.mark.parametrize("field", ["source_path", "selected_audio_path"])
@pytest.mark.parametrize("escape", ["../outside.wav", "/tmp/outside.wav", "escape.wav"])
def test_rejects_traversal_absolute_paths_and_symlink_escape(protocol, tmp_path, field, escape):
    (tmp_path / "escape.wav").symlink_to(tmp_path.parent / "outside.wav")
    protocol["cases"][0][field] = escape
    with pytest.raises(ValueError, match="within source_root"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_rejects_escaping_missing_input_path(protocol, tmp_path):
    protocol["missing_inputs"][0]["source_path"] = "../lost.wav"
    with pytest.raises(ValueError, match="within source_root"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


@pytest.mark.parametrize("scope,span,duration", [
    ("exact_excerpt", None, 2),
    ("exact_excerpt", {"start_seconds": 1, "end_seconds": 1}, 2),
    ("exact_excerpt", {"start_seconds": 1, "end_seconds": 3}, 2),
    ("exact_excerpt", {"start_seconds": 0, "end_seconds": 1}, 2),
    ("full_track", {"start_seconds": 0, "end_seconds": 2}, 2),
    ("unknown_extent", {"start_seconds": 0, "end_seconds": 2}, 2),
    ("full_track", None, 1),
])
def test_rejects_range_scope_and_duration_inconsistency(protocol, tmp_path, scope, span, duration):
    protocol["cases"][0].update(coverage_scope=scope, source_range=span, duration_seconds=duration)
    with pytest.raises(ValueError, match="range|duration|full_track"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_excerpt_cannot_claim_full_original_stream(protocol, tmp_path):
    wav(tmp_path / "source.wav", seconds=3)
    protocol["cases"][0]["source_sha256"] = sha(tmp_path / "source.wav")
    with pytest.raises(ValueError, match="full_track"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)
    protocol["cases"][0].update(coverage_scope="exact_excerpt",
                                source_range={"start_seconds": 1, "end_seconds": 3})
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1


def test_unknown_original_extent_is_allowed_without_full_track_claim(protocol, tmp_path):
    protocol["cases"][0]["coverage_scope"] = "unknown_extent"
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1


@pytest.mark.parametrize("change", ["omit", "duplicate", "extra", "both", "duplicate_inventory"])
def test_inventory_reconciles_exactly_once(protocol, tmp_path, change):
    if change == "omit":
        protocol["missing_inputs"] = []
    elif change == "duplicate":
        protocol["cases"].append(deepcopy(protocol["cases"][0]))
    elif change == "extra":
        protocol["inventory_ids"].append("never-represented")
    elif change == "both":
        protocol["missing_inputs"].append({"case_id": "literal-two-second-control",
                                           "source_path": "lost.wav", "reason": "lost"})
    else:
        protocol["inventory_ids"].append("literal-two-second-control")
    with pytest.raises(ValueError, match="inventory"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_byte_identical_pcm_requires_alias_instead_of_second_case(protocol, tmp_path):
    duplicate = deepcopy(protocol["cases"][0])
    duplicate["case_id"] = "second-source"
    protocol["inventory_ids"].append("second-source")
    protocol["cases"].append(duplicate)
    with pytest.raises(ValueError, match="duplicate PCM"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_aliases_require_source_identity_and_range_provenance(protocol, tmp_path):
    wav(tmp_path / "alias.wav")
    alias = {"source_path": "alias.wav", "source_sha256": sha(tmp_path / "alias.wav"),
             "audio_stream_index": 0, "source_range": None,
             "selection_provenance": "Same complete two-second PCM from alternate source."}
    protocol["cases"][0]["source_aliases"] = [alias]
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1
    alias["source_range"] = {"start_seconds": 0, "end_seconds": 1}
    with pytest.raises(ValueError, match="alias.*range|range.*duration"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)
    alias["source_range"] = None
    del alias["selection_provenance"]
    with pytest.raises(ValueError, match="selection_provenance"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


@pytest.mark.parametrize("field", ["predictions", "model_prediction", "post_result_corrections",
                                  "post_result_labels", "numeric_accuracy_threshold"])
def test_frozen_inputs_exclude_predictions_and_post_result_labels(protocol, tmp_path, field):
    protocol["cases"][0][field] = []
    with pytest.raises(ValueError, match=field):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_reference_labels_require_explicit_confirmation_provenance(protocol, tmp_path):
    protocol["cases"][0]["confirmed_reference_labels"] = [
        {"category_id": "633", "role": "primary", "confirmation_source": ""}]
    with pytest.raises(ValueError, match="confirmation_source"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_digest_detects_mutation_without_resealing(protocol, tmp_path):
    protocol["cases"][0]["selection_provenance"] = "Changed after freeze"
    with pytest.raises(ValueError, match="canonical_sha256"):
        validator().validate_protocol(protocol, source_root=tmp_path)


def test_alias_rejects_different_pcm_even_with_valid_file_hash(protocol, tmp_path):
    wav(tmp_path / "different.wav", sample=2)
    protocol["cases"][0]["source_aliases"] = [{
        "source_path": "different.wav", "source_sha256": sha(tmp_path / "different.wav"),
        "audio_stream_index": 0, "source_range": None,
        "selection_provenance": "Claimed duplicate complete normalized PCM."}]
    with pytest.raises(ValueError, match="alias.*PCM"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_alias_rejects_escaping_symlink(protocol, tmp_path):
    (tmp_path / "escaped-alias.wav").symlink_to(tmp_path.parent / "outside.wav")
    protocol["cases"][0]["source_aliases"] = [{
        "source_path": "escaped-alias.wav", "source_sha256": "0" * 64,
        "audio_stream_index": 0, "source_range": None,
        "selection_provenance": "Historical alias"}]
    with pytest.raises(ValueError, match="within source_root"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_post_result_confirmation_cannot_be_relabelled_as_frozen_reference(protocol, tmp_path):
    protocol["cases"][0]["confirmed_reference_labels"] = [{
        "category_id": "633", "role": "primary",
        "confirmation_source": "paired-context-v1/label-review/confirmed-labels.json; post-result adjudication"}]
    with pytest.raises(ValueError, match="post-result|confirmation_source"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_cli_records_freeze_identity_exclusively_without_inference(protocol, tmp_path):
    validator()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(protocol))
    out = tmp_path / "new-freeze"
    command = [sys.executable, str(SCRIPT), "--manifest", str(manifest),
               "--source-root", str(tmp_path), "--output-dir", str(out)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads((out / "protocol-freeze.json").read_text())
    assert record["canonical_sha256"] == protocol["canonical_sha256"]
    assert record["manifest_sha256"] == sha(manifest)
    assert record["validator_sha256"] == sha(SCRIPT)
    assert record["code_commit"] == "dfbb98de8df50265eea6229b35479150aa3985a6"
    assert record["status"] == "validated_experimental_non_blind_no_inference"
    ffmpeg = Path(shutil.which("ffmpeg")).resolve()
    assert record["normalizer"]["ffmpeg_sha256"] == sha(ffmpeg)
    assert record["normalizer"]["ffmpeg_version"].startswith("ffmpeg version ")
    assert record["unverified_source_claims"] == 0
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_cross_wired_equal_duration_source_and_selected_pcm_is_rejected(protocol, tmp_path):
    # Both file hashes and durations are correct; only the relationship is false.
    wav(tmp_path / "source.wav", sample=2)
    protocol["cases"][0]["source_sha256"] = sha(tmp_path / "source.wav")
    with pytest.raises(ValueError, match="source.*PCM"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_ranged_five_second_wav_alias_compares_exact_interval(protocol, tmp_path):
    wav(tmp_path / "source.wav", seconds=5)
    wav(tmp_path / "selected.wav", seconds=5)
    wav(tmp_path / "Y10.wav", seconds=10, sample=2)
    row = protocol["cases"][0]
    row.update(source_sha256=sha(tmp_path / "source.wav"),
               selected_audio_sha256=sha(tmp_path / "selected.wav"), duration_seconds=5)
    row["source_aliases"] = [{"source_path": "Y10.wav", "source_sha256": sha(tmp_path / "Y10.wav"),
        "audio_stream_index": 0, "source_range": {"start_seconds": 5, "end_seconds": 10},
        "selection_provenance": "Claimed Y08/Y10 five-second ranged duplicate"}]
    with pytest.raises(ValueError, match="alias.*PCM"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)
    with wave.open(str(tmp_path / "Y10.wav"), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x02\x00" * 80000 + b"\x01\x00" * 80000)
    row["source_aliases"][0]["source_sha256"] = sha(tmp_path / "Y10.wav")
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1


def test_real_paired_context_bare_path_cannot_be_frozen_reference(protocol, tmp_path):
    (tmp_path / "confirmed-labels.json").write_text(json.dumps(PAIRED_CONTEXT_LABELS))
    protocol["cases"][0]["confirmed_reference_labels"] = [{"category_id": "268", "role": "primary",
        "confirmation_source": "confirmed-labels.json"}]
    with pytest.raises(ValueError, match="confirmation_source|post-result"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def confirmed_fixture(protocol, tmp_path):
    evidence = {"status": "frozen_before_inference", "approval_context": "User confirmed exact clip before inference",
        "cases": [{"id": "B02", "primary_category": "222", "secondary_categories": [],
                   "audio_sha256": sha(tmp_path / "source.wav")}]}
    path = tmp_path / "frozen-labels.json"
    path.write_text(json.dumps(evidence))
    confirmation = {"path": "frozen-labels.json", "sha256": sha(path), "record_id": "B02",
        "confirmation_stage": "before_inference", "audio_path": "source.wav",
        "audio_sha256": sha(tmp_path / "source.wav"), "audio_stream_index": 0, "source_range": None}
    protocol["cases"][0]["confirmed_reference_labels"] = [{"category_id": "222", "role": "primary",
        "confirmation_source": confirmation}]
    return confirmation


def test_structured_confirmation_reads_record_and_binds_exact_audio(protocol, tmp_path):
    confirmed_fixture(protocol, tmp_path)
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("record_id", "B01"),
    ("confirmation_stage", "after_inference"), ("audio_sha256", "0" * 64),
    ("path", "../frozen-labels.json"), ("source_range", {"start_seconds": 0, "end_seconds": 1})])
def test_structured_confirmation_rejects_unbound_evidence(protocol, tmp_path, field, value):
    confirmation = confirmed_fixture(protocol, tmp_path)
    confirmation[field] = value
    with pytest.raises(ValueError, match="confirmation|evidence|source_root|PCM|range"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_structured_confirmation_rejects_actual_post_result_content(protocol, tmp_path):
    confirmation = confirmed_fixture(protocol, tmp_path)
    path = tmp_path / "frozen-labels.json"
    path.write_text(json.dumps(PAIRED_CONTEXT_LABELS))
    confirmation.update(sha256=sha(path), record_id="P01")
    with pytest.raises(ValueError, match="post-result"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def routing_fixture(protocol, tmp_path):
    (tmp_path / "user-labels.json").write_text(json.dumps({
        "source": "User listening confirmation before validation inference",
        "labels": {"Y08": "instrumental without vocals"}}))
    (tmp_path / "audio-manifest.json").write_text(json.dumps({"cases": [{
        "id": "Y08", "audio_sha256": sha(tmp_path / "source.wav") }]}))
    evidence = {"path": "user-labels.json", "sha256": sha(tmp_path / "user-labels.json"),
        "record_id": "Y08", "confirmation_stage": "before_inference", "audio_path": "source.wav",
        "audio_sha256": sha(tmp_path / "source.wav"), "audio_stream_index": 0, "source_range": None,
        "audio_record": {"path": "audio-manifest.json", "sha256": sha(tmp_path / "audio-manifest.json"),
                         "record_id": "Y08"}}
    row = protocol["cases"][0]
    row.update(expected_routing="instrumental_out_of_scope", routing_confirmation=evidence)
    return evidence


def test_routing_requires_structured_content_checked_confirmation(protocol, tmp_path):
    protocol["cases"][0]["expected_routing"] = "instrumental_out_of_scope"
    with pytest.raises(ValueError, match="routing.*confirmation"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_routing_user_labels_bind_hashed_companion_audio_record(protocol, tmp_path):
    routing_fixture(protocol, tmp_path)
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1


@pytest.mark.parametrize("change", ["wrong_label", "wrong_audio", "wrong_record", "changed_file", "escape"])
def test_routing_rejects_unbound_user_label_evidence(protocol, tmp_path, change):
    evidence = routing_fixture(protocol, tmp_path)
    if change == "wrong_label":
        protocol["cases"][0]["expected_routing"] = "silence_out_of_scope"
    elif change == "wrong_audio":
        wav(tmp_path / "other.wav", sample=2)
        evidence.update(audio_path="other.wav", audio_sha256=sha(tmp_path / "other.wav"))
    elif change == "wrong_record":
        evidence["audio_record"]["record_id"] = "Y10"
    elif change == "changed_file":
        (tmp_path / "user-labels.json").write_text('{}')
    else:
        (tmp_path / "escaped.json").symlink_to(tmp_path.parent / "labels.json")
        evidence["audio_record"]["path"] = "escaped.json"
    with pytest.raises(ValueError, match="routing|confirmation|evidence|source_root"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_unreproduced_original_claim_is_explicit_not_a_pcm_alias(protocol, tmp_path):
    wav(tmp_path / "claimed-original.wav", sample=2)
    claim = {"source_path": "claimed-original.wav", "source_sha256": sha(tmp_path / "claimed-original.wav"),
        "audio_stream_index": 0, "source_range": {"start_seconds": 0, "end_seconds": 2},
        "selection_provenance": "Historical recorded original range; selected bytes not reproduced",
        "reason": "Strict normalization byte audit failed; exact historical WAV retained, original mapping unverified"}
    row = protocol["cases"][0]
    row.update(coverage_scope="unknown_extent", unverified_source_claims=[claim])
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1
    claim["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source_sha256"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


@pytest.mark.parametrize("change", ["category", "duplicate_record", "evidence_symlink", "audio_path_mismatch"])
def test_confirmation_checks_record_value_uniqueness_and_paths(protocol, tmp_path, change):
    evidence = confirmed_fixture(protocol, tmp_path)
    path = tmp_path / "frozen-labels.json"
    document = json.loads(path.read_text())
    if change == "category":
        protocol["cases"][0]["confirmed_reference_labels"][0]["category_id"] = "633"
    elif change == "duplicate_record":
        document["cases"].append(deepcopy(document["cases"][0]))
    elif change == "evidence_symlink":
        (tmp_path / "escaped-evidence.json").symlink_to(tmp_path.parent / "frozen-labels.json")
        evidence["path"] = "escaped-evidence.json"
    else:
        wav(tmp_path / "wrong-recorded.wav", sample=2)
        document["cases"][0]["audio_path"] = "wrong-recorded.wav"
    path.write_text(json.dumps(document))
    evidence["sha256"] = sha(path)
    with pytest.raises(ValueError, match="confirmation|evidence|source_root"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_compressed_source_is_decoded_and_bound_not_only_hashed(protocol, tmp_path):
    source = tmp_path / "source.flac"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(tmp_path / "source.wav"),
                    str(source)], check=True)
    row = protocol["cases"][0]
    row.update(source_path="source.flac", source_sha256=sha(source))
    assert len(validator().validate_protocol(seal(protocol), source_root=tmp_path)) == 1
    wav(tmp_path / "selected.wav", sample=2)
    row["selected_audio_sha256"] = sha(tmp_path / "selected.wav")
    with pytest.raises(ValueError, match="source.*PCM"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)


def test_undeclared_normalization_recipe_is_rejected(protocol, tmp_path):
    protocol["cases"][0]["normalization"] = "approximate-waveform-match"
    with pytest.raises(ValueError, match="normalization"):
        validator().validate_protocol(seal(protocol), source_root=tmp_path)
