import copy
import hashlib
import json
import wave

import pytest

from scripts.compare_acoustic_probes import compare_reports, validate_inventory


@pytest.fixture
def manifest(tmp_path):
    pcm = tmp_path / 'input.wav'
    with wave.open(str(pcm), 'wb') as out:
        out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        out.writeframes(b'\0\0' * 16000)
    digest = hashlib.sha256(pcm.read_bytes()).hexdigest()
    return {'non_blind': True, 'candidates': ['yamnet', 'moss'], 'cases': [{
        'id': 'control', 'source': str(pcm), 'source_sha256': digest,
        'selected_pcm': str(pcm), 'selected_pcm_sha256': digest,
        'audio_stream_index': 0, 'range': {'start_seconds': 0.0, 'end_seconds': 1.0,
                                         'extent': 'full_selected_audio'},
        'routing_role': 'speech_with_background_music', 'non_blind': True}]}


def test_valid_exact_pcm_is_returned(manifest):
    assert validate_inventory(manifest)[0]['id'] == 'control'


@pytest.mark.parametrize('change', ['missing_hash', 'duplicate', 'changed_hash', 'excerpt_as_full'])
def test_invalid_frozen_inventory_rejected(manifest, change):
    case = manifest['cases'][0]
    if change == 'missing_hash':
        del case['source_sha256']
    elif change == 'duplicate':
        manifest['cases'].append(copy.deepcopy(case))
    elif change == 'changed_hash':
        case['source_sha256'] = 'a' * 64
    else:
        case['range']['start_seconds'] = 0.5
    with pytest.raises(ValueError):
        validate_inventory(manifest)


def test_omitted_predeclared_case_cannot_disappear(manifest):
    with pytest.raises(ValueError, match='missing case'):
        compare_reports(manifest, [])


def test_missing_candidates_are_explicit_failures(manifest):
    result = compare_reports(manifest, [{'case_id': 'control', 'candidates': {}}])
    assert result['cases'][0]['candidates']['moss']['status'] == 'missing'
    assert result['admission'] == 'no_acoustic_candidate'


def moss_report(manifest, raw='{"foreground":"yes","native_genre":"pop"}'):
    return {'kind': 'moss_acoustic_diagnostic', 'status': 'complete',
            'audio': manifest['cases'][0]['selected_pcm'], 'duration_seconds': 1.0,
            'records': [{'start_seconds': 0.0, 'end_seconds': 1.0, 'raw': raw,
                         'generation_tokens': 50, 'status': 'parsed'}], 'errors': []}


def envelope(manifest, candidates):
    return {'case_id': 'control', 'candidates': candidates,
            'input_identity': {'selected_pcm': manifest['cases'][0]['selected_pcm'],
                               'selected_pcm_sha256': manifest['cases'][0]['selected_pcm_sha256'],
                               'verification': 'whole_corpus_before_and_after'}}


def test_speech_background_foreground_false_positive_disqualifies(manifest):
    result = compare_reports(manifest, [envelope(manifest, {'moss': moss_report(manifest)})])
    assert result['cases'][0]['candidates']['moss']['foreground_false_positive'] is True
    assert result['candidates']['moss']['admission'] == 'no_acoustic_candidate'


def test_invalid_genre_json_is_technical_failure(manifest):
    report = moss_report(manifest, '{"foreground":"yes","native_genre":99}')
    result = compare_reports(manifest, [envelope(manifest, {'moss': report})])
    assert result['cases'][0]['candidates']['moss']['status'] == 'technical_error'
    assert result['admission'] == 'no_acoustic_candidate'


@pytest.mark.parametrize('role', ['instrumental_out_of_scope', 'non_english_out_of_scope', 'environment_silence_out_of_scope'])
def test_acoustic_observations_cannot_override_original_routing(manifest, role):
    manifest['cases'][0]['routing_role'] = role
    result = compare_reports(manifest, [envelope(manifest, {'moss': moss_report(manifest)})])
    assert result['cases'][0]['routing_role'] == role
    assert result['cases'][0]['candidates']['moss']['native_genre_seconds'] == {'pop': 1.0}


def test_incomplete_success_report_fails_closed(manifest):
    report = moss_report(manifest)
    report['records'][0]['end_seconds'] = 0.5
    result = compare_reports(manifest, [envelope(manifest, {'moss': report})])
    assert result['cases'][0]['candidates']['moss']['status'] == 'technical_error'


def test_different_pcm_same_duration_report_is_rejected(manifest):
    report = moss_report(manifest)
    report['audio'] = '/different/input.wav'
    result = compare_reports(manifest, [envelope(manifest, {'moss': report})])
    assert result['cases'][0]['candidates']['moss']['status'] == 'technical_error'


def test_incomplete_token_cap_is_not_parsed_success(manifest):
    report = moss_report(manifest)
    report['records'][0]['generation_tokens'] = 2048
    result = compare_reports(manifest, [envelope(manifest, {'moss': report})])
    assert result['cases'][0]['candidates']['moss']['status'] == 'technical_error'


def test_admission_reasons_separate_quality_from_missing_review(manifest):
    result = compare_reports(manifest, [envelope(manifest, {'moss': moss_report(manifest)})])
    reasons = result['candidates']['moss']['reasons']
    assert 'speech_background_foreground_false_positive' in reasons
    assert 'unreviewed_native_to_iab_mapping' in reasons


def test_yamnet_frame_by_521_matrix_is_complete_diagnostic(manifest):
    report = {'status': 'complete', 'errors': [], 'audio': manifest['cases'][0]['selected_pcm'],
              'duration_seconds': 1.0, 'records': [{'start_seconds': 0.0, 'end_seconds': 1.0,
                                                   'scores': [[0.0] * 521, [0.2] * 521]}]}
    result = compare_reports(manifest, [envelope(manifest, {'yamnet': report})])
    assert result['cases'][0]['candidates']['yamnet']['status'] == 'complete'


@pytest.mark.parametrize('change', ['wrong_hash', 'missing_identity', 'unverified'])
def test_report_without_verified_pcm_hash_cannot_pass(manifest, change):
    wrapped = envelope(manifest, {'moss': moss_report(manifest)})
    if change == 'wrong_hash':
        wrapped['input_identity']['selected_pcm_sha256'] = 'a' * 64
    elif change == 'missing_identity':
        del wrapped['input_identity']
    else:
        wrapped['input_identity']['verification'] = 'failed'
    result = compare_reports(manifest, [wrapped])
    assert result['cases'][0]['candidates']['moss']['status'] == 'technical_error'


def test_attested_loader_rejects_changed_raw_report(manifest, tmp_path):
    from scripts.compare_acoustic_probes import load_attested_reports
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps(moss_report(manifest)))
    attestation = {'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'manifest_commit': '9767e8d', 'verification': 'whole_corpus_before_and_after',
                   'cases': [{'case_id': 'control', 'selected_pcm': manifest['cases'][0]['selected_pcm'],
                              'selected_pcm_sha256': manifest['cases'][0]['selected_pcm_sha256'],
                              'reports': {'moss': {'path': str(raw), 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}}}]}
    assert load_attested_reports(path, attestation)[0]['candidates']['moss']['status'] == 'complete'
    raw.write_text('{}')
    with pytest.raises(ValueError, match='report hash'):
        load_attested_reports(path, attestation)


def test_technical_failure_keeps_actual_probe_error(manifest):
    report = moss_report(manifest)
    report.update(status='technical_error', errors=[{'type': 'IncompleteGeneration',
                                                    'message': 'INCOMPLETE_GENERATION: token cap reached'}])
    result = compare_reports(manifest, [envelope(manifest, {'moss': report})])
    candidate = result['cases'][0]['candidates']['moss']
    assert candidate['status'] == 'technical_error'
    assert candidate['probe_errors'][0]['type'] == 'IncompleteGeneration'


@pytest.mark.parametrize('malformation', ['array_report', 'null_report', 'null_identity',
                                         'array_identity', 'null_records', 'object_records',
                                         'null_record', 'array_record', 'array_candidates'])
def test_malformed_candidate_preserves_other_models_and_cases(manifest, malformation):
    good = moss_report(manifest)
    bad = copy.deepcopy(good)
    first = envelope(manifest, {'moss': bad, 'yamnet': {
        'status': 'complete', 'errors': [], 'audio': manifest['cases'][0]['selected_pcm'],
        'duration_seconds': 1.0, 'records': [{'start_seconds': 0.0, 'end_seconds': 1.0,
                                           'scores': [[0.1] * 521]}]}})
    if malformation == 'array_report':
        first['candidates']['moss'] = []
    elif malformation == 'null_report':
        first['candidates']['moss'] = None
    elif malformation == 'array_candidates':
        first['candidates'] = []
    elif malformation.endswith('identity'):
        first['input_identity'] = None if malformation.startswith('null') else []
    elif malformation == 'null_records':
        bad['records'] = None
    elif malformation == 'object_records':
        bad['records'] = {}
    else:
        bad['records'] = [None if malformation == 'null_record' else []]
    second_case = copy.deepcopy(manifest['cases'][0])
    second_case['id'] = 'other'
    manifest['cases'].append(second_case)
    second = envelope(manifest, {'moss': good})
    second['case_id'] = 'other'
    result = compare_reports(manifest, [first, second])
    assert [case['id'] for case in result['cases']] == ['control', 'other']
    failed = result['cases'][0]['candidates']['moss']
    assert failed['status'] == 'technical_error'
    assert failed['errors']
    assert result['cases'][1]['candidates']['moss']['status'] == 'complete'
    assert result['cases'][0]['candidates']['yamnet']['status'] == (
        'technical_error' if malformation.endswith('identity') or malformation == 'array_candidates' else 'complete')


@pytest.mark.parametrize('candidate,field', [
    ('yamnet', 'score'), ('yamnet', 'duration'), ('moss', 'duration'),
])
def test_oversized_json_integer_preserves_other_candidates_and_cases(manifest, candidate, field):
    good = {'moss': moss_report(manifest), 'yamnet': {
        'status': 'complete', 'errors': [], 'audio': manifest['cases'][0]['selected_pcm'],
        'duration_seconds': 1.0, 'records': [{'start_seconds': 0.0, 'end_seconds': 1.0,
                                           'scores': [[0.1] * 521]}]}}
    bad = copy.deepcopy(good)
    if field == 'score':
        bad[candidate]['records'][0]['scores'][0][0] = 10 ** 1000
    else:
        bad[candidate]['duration_seconds'] = 10 ** 1000
    # These oversized integers are valid JSON, as in a saved raw report.
    first = envelope(manifest, json.loads(json.dumps(bad, allow_nan=False)))
    second_case = copy.deepcopy(manifest['cases'][0])
    second_case['id'] = 'other'
    manifest['cases'].append(second_case)
    second = envelope(manifest, good)
    second['case_id'] = 'other'

    result = compare_reports(manifest, [first, second])

    assert [case['id'] for case in result['cases']] == ['control', 'other']
    failed = result['cases'][0]['candidates'][candidate]
    assert failed['status'] == 'technical_error'
    assert failed['errors']
    other_candidate = 'moss' if candidate == 'yamnet' else 'yamnet'
    assert result['cases'][0]['candidates'][other_candidate]['status'] == 'complete'
    assert all(row['status'] == 'complete' for row in result['cases'][1]['candidates'].values())
    assert result['candidates'][candidate]['complete_cases'] == 1
    assert result['candidates'][other_candidate]['complete_cases'] == 2
    assert result['admission'] == 'no_acoustic_candidate'


def test_attested_null_raw_is_technical_but_absent_reference_is_missing(manifest, tmp_path):
    from scripts.compare_acoustic_probes import load_attested_reports
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    raw = tmp_path / 'null.json'
    raw.write_text('null')
    attestation = {'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'manifest_commit': '9767e8d', 'verification': 'whole_corpus_before_and_after',
                   'cases': [{'case_id': 'control', 'selected_pcm': manifest['cases'][0]['selected_pcm'],
                              'selected_pcm_sha256': manifest['cases'][0]['selected_pcm_sha256'],
                              'reports': {'moss': {'path': str(raw), 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()},
                                          'yamnet': None}}]}
    result = compare_reports(manifest, load_attested_reports(path, attestation))
    assert result['cases'][0]['candidates']['moss']['status'] == 'technical_error'
    assert result['cases'][0]['candidates']['yamnet']['status'] == 'missing'
