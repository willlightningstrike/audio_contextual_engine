"""Frozen non-blind acoustic comparison; observations never alter routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave
from collections.abc import Sequence

from engine.domain import TimeRange
from engine.signals.acoustic_probe import summarize_observations
from scripts.probe_moss_music import parse_moss_observation

REQUIRED_CASE_KEYS = {'id', 'source', 'source_sha256', 'selected_pcm',
                      'selected_pcm_sha256', 'audio_stream_index', 'range',
                      'routing_role', 'non_blind'}
_MISSING_REPORT = object()


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_streams(path: Path) -> list[dict]:
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a',
                             '-show_streams', '-of', 'json', str(path)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)['streams']


def validate_inventory(manifest: dict) -> tuple[dict, ...]:
    if manifest.get('non_blind') is not True or not manifest.get('cases'):
        raise ValueError('expected frozen non-blind inventory')
    seen, hashes, streams = set(), {}, {}
    for case in manifest['cases']:
        if set(case) != REQUIRED_CASE_KEYS or case['non_blind'] is not True:
            raise ValueError('acoustic case manifest is not frozen non-blind input')
        if not isinstance(case['id'], str) or not case['id'] or case['id'] in seen:
            raise ValueError('duplicate or invalid case ID')
        seen.add(case['id'])
        for field in ('source', 'selected_pcm'):
            path = Path(case[field])
            if not path.is_absolute() or not path.is_file():
                raise ValueError('input requires exact available absolute path')
            if path not in hashes:
                hashes[path] = sha256(path)
            if hashes[path] != case[field + '_sha256']:
                raise ValueError('changed input hash: ' + field)
        span = case['range']
        if set(span) != {'start_seconds', 'end_seconds', 'extent'}:
            raise ValueError('invalid range fields')
        start, end = span['start_seconds'], span['end_seconds']
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
            raise ValueError('invalid range')
        source = Path(case['source'])
        if source not in streams:
            streams[source] = source_streams(source)
        selected = [s for s in streams[source] if s['index'] == case['audio_stream_index']]
        if type(case['audio_stream_index']) is not int or len(selected) != 1:
            raise ValueError('invalid selected audio stream')
        duration = float(selected[0]['duration'])
        if end > duration + 1 / 16000:
            raise ValueError('range exceeds selected audio')
        if span['extent'] not in {'excerpt', 'full_selected_audio'}:
            raise ValueError('invalid extent')
        if span['extent'] == 'full_selected_audio' and (start != 0 or abs(end - duration) > 1 / 16000):
            raise ValueError('excerpt mislabelled full-file')
        with wave.open(case['selected_pcm'], 'rb') as pcm:
            if (pcm.getnchannels(), pcm.getsampwidth(), pcm.getframerate(), pcm.getcomptype()) != (1, 2, 16000, 'NONE'):
                raise ValueError('selected PCM must be mono 16 kHz PCM16')
            if abs(pcm.getnframes() / 16000 - (end - start)) > 1 / 16000:
                raise ValueError('selected PCM range mismatch')
        if not isinstance(case['routing_role'], str) or not case['routing_role']:
            raise ValueError('routing role required')
    return tuple(manifest['cases'])


def _compare_candidate(case: dict, name: str, report: object, identity: object) -> dict:
    result = {'status': 'missing' if report is _MISSING_REPORT else 'technical_error',
              'foreground_false_positive': False, 'native_genre_seconds': {},
              'foreground_seconds': 0.0, 'errors': [],
              'probe_errors': []}
    if report is _MISSING_REPORT:
        return result
    try:
        if not isinstance(report, dict):
            raise ValueError('candidate report must be an object')
        result['probe_errors'] = report.get('errors', [])
        if not isinstance(identity, dict):
            raise ValueError('input identity must be an object')
        if (identity.get('verification') != 'whole_corpus_before_and_after'
                or identity.get('selected_pcm') != case['selected_pcm']
                or identity.get('selected_pcm_sha256') != case['selected_pcm_sha256']):
            raise ValueError('input identity was not verified against frozen PCM hash')
        if report.get('status') != 'complete' or report.get('errors'):
            raise ValueError('probe incomplete or technical failure')
        if report.get('audio') != case['selected_pcm']:
            raise ValueError('report PCM identity mismatch')
        duration = case['range']['end_seconds'] - case['range']['start_seconds']
        if abs(report['duration_seconds'] - duration) > 1 / 16000:
            raise ValueError('report duration mismatch')
        if not isinstance(report.get('records'), list):
            raise ValueError('report records must be an array')
        if any(not isinstance(row, dict) for row in report['records']):
            raise ValueError('each report record must be an object')
        if name == 'moss':
            observations = []
            for row in report['records']:
                if row.get('status') != 'parsed' or type(row.get('generation_tokens')) is not int or not 0 <= row['generation_tokens'] < 2048:
                    raise ValueError('invalid or incomplete generation')
                observations.append(parse_moss_observation(row['raw'], TimeRange(
                    start_seconds=row['start_seconds'], end_seconds=row['end_seconds'])))
            summary = summarize_observations(report['duration_seconds'], observations)
            result.update(foreground_seconds=summary.foreground_seconds,
                          native_genre_seconds=summary.native_genre_seconds)
            result['foreground_false_positive'] = case['routing_role'] == 'speech_with_background_music' and summary.foreground_seconds > 0
        elif name == 'yamnet':
            cursor = 0.0
            for row in report['records']:
                scores = row['scores']
                if row['start_seconds'] != cursor or not cursor < row['end_seconds'] <= report['duration_seconds']:
                    raise ValueError('incomplete diagnostic coverage')
                if not isinstance(scores, list) or not scores or any(
                    not isinstance(frame, list) or len(frame) != 521 or any(
                        type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in frame
                    ) for frame in scores
                ):
                    raise ValueError('invalid raw scores')
                cursor = row['end_seconds']
            if cursor != report['duration_seconds']:
                raise ValueError('incomplete diagnostic coverage')
        else:
            raise ValueError('unknown candidate')
        result['status'] = 'complete'
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        result['errors'].append(str(exc))
    return result


def compare_reports(manifest: dict, reports: Sequence[dict]) -> dict:
    cases = validate_inventory(manifest)
    by_id = {}
    for report in reports:
        if report['case_id'] in by_id:
            raise ValueError('duplicate case report')
        by_id[report['case_id']] = report
    if set(by_id) != {case['id'] for case in cases}:
        raise ValueError('missing case or unexpected case report')
    results = []
    for case in cases:
        envelope = by_id[case['id']]
        candidate_reports = envelope.get('candidates')
        results.append({'id': case['id'], 'routing_role': case['routing_role'],
                        'candidates': {name: _compare_candidate(
                            case, name,
                            candidate_reports.get(name, _MISSING_REPORT)
                            if isinstance(candidate_reports, dict) else candidate_reports,
                            envelope.get('input_identity', {})) for name in manifest['candidates']}})
    # No reviewed foreground rule / native-to-IAB mapping has been frozen.
    # Completion alone cannot manufacture a calibrated semantic admission.
    candidates = {}
    for name in manifest['candidates']:
        values = [row['candidates'][name] for row in results]
        reasons = ['unreviewed_native_to_iab_mapping', 'unreviewed_foreground_calibration']
        if any(row['status'] == 'missing' for row in values):
            reasons.append('missing_candidate_case')
        if any(row['status'] == 'technical_error' for row in values):
            reasons.append('technical_or_incomplete_case')
        if any(row['foreground_false_positive'] for row in values):
            reasons.append('speech_background_foreground_false_positive')
        if any('unknown' in case['routing_role'] or 'provisional' in case['routing_role'] for case in cases):
            reasons.append('unreviewed_routing_gold')
        candidates[name] = {'admission': 'no_acoustic_candidate',
                            'complete_cases': sum(row['status'] == 'complete' for row in values),
                            'reasons': reasons}
    return {'non_blind': True, 'cases': results, 'candidates': candidates,
            'admission': 'no_acoustic_candidate', 'production_ready': False}


def load_attested_reports(manifest_path: Path, attestation: dict) -> list[dict]:
    """Bind parsed raw bytes to the frozen manifest and start/end input audit.

    This is an isolated experimental audit, not an atomic per-attempt guarantee.
    It cannot detect malicious input change-and-revert between the two audits.
    """
    if sha256(manifest_path) != attestation.get('manifest_sha256'):
        raise ValueError('manifest hash does not match attestation')
    if not attestation.get('manifest_commit'):
        raise ValueError('attestation requires corpus commit')
    results = []
    for case in attestation['cases']:
        candidates = {}
        for model, reference in case['reports'].items():
            if reference is None:
                continue
            raw = Path(reference['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference['sha256']:
                raise ValueError('raw report hash does not match attestation')
            candidates[model] = json.loads(raw)
        results.append({'case_id': case['case_id'], 'candidates': candidates,
                        'input_identity': {'selected_pcm': case['selected_pcm'],
                                           'selected_pcm_sha256': case['selected_pcm_sha256'],
                                           'verification': attestation.get('verification')}})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('attestation', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reports = load_attested_reports(args.manifest, json.loads(args.attestation.read_text()))
    result = compare_reports(json.loads(args.manifest.read_text()), reports)
    with args.out.open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
