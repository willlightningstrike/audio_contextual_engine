import json
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from scripts import probe_moss_music as probe


WINDOW = TimeRange(start_seconds=0, end_seconds=30)


@pytest.mark.parametrize('foreground,genre', [('yes', 'jazz'), ('yes', None), ('no', None), ('uncertain', None)])
def test_valid_observation(foreground, genre):
    raw = json.dumps({'foreground': foreground, 'native_genre': genre})
    observation = probe.parse_moss_observation(raw, WINDOW)
    assert observation.foreground == foreground
    assert observation.native_genre == genre
    assert observation.raw == raw
    assert observation.model_id == 'moss-music-8b-thinking-8bit'


@pytest.mark.parametrize('raw', [
    '```json\n{}\n```', '[]', '{}',
    '{"foreground":"yes","foreground":"no","native_genre":null}',
    '{"foreground":"yes","native_genre":null,"extra":1}',
    '{"foreground":[],"native_genre":null}',
    '{"foreground":true,"native_genre":null}',
    '{"foreground":"yes","native_genre":true}',
    '{"foreground":"yes","native_genre":"  "}',
    '{"foreground":"no","native_genre":"jazz"}',
    '{"foreground":"uncertain","native_genre":"jazz"}',
    '{"foreground":"yes","native_genre":NaN}',
    '{"foreground":"yes","native_genre":1e999}',
    '{"foreground":"yes","native_genre":' + '9' * 5000 + '}',
    None,
])
def test_invalid_output_is_technical(raw):
    with pytest.raises(probe.InvalidMossOutput):
        probe.parse_moss_observation(raw, WINDOW)


@pytest.fixture
def assets(tmp_path, monkeypatch):
    root = tmp_path / 'model'
    root.mkdir()
    (root / 'config.json').write_text('{"model_type":"moss_music"}')
    import hashlib
    monkeypatch.setattr(probe, 'EXPECTED_ASSET_HASHES', {'config.json': hashlib.sha256((root / 'config.json').read_bytes()).hexdigest()})
    return root


def wav(path, seconds=31):
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        output.writeframes(b'\0\0' * (seconds * 16000))


def run_fake(tmp_path, assets, monkeypatch, response, token=None):
    audio = tmp_path / 'private-title.wav'
    wav(audio)
    calls = []
    def generate(path, **kwargs):
        calls.append((Path(path), kwargs))
        with wave.open(path) as stream:
            assert stream.getnframes() == (480000 if len(calls) == 1 else 16000)
        if token:
            token.cancel()
        return response
    monkeypatch.setattr(probe, '_load_model', lambda root: SimpleNamespace(generate=generate))
    out = tmp_path / 'report.json'
    return audio, out, calls


def test_complete_probe_records_windows_and_frozen_generation(tmp_path, assets, monkeypatch):
    response = SimpleNamespace(text='{"foreground":"yes","native_genre":"jazz"}', generation_tokens=12, prompt_tokens=4, total_time=0.1)
    audio, out, calls = run_fake(tmp_path, assets, monkeypatch, response)
    report = probe.probe_moss(audio, local_model=assets, out=out, token=CancellationToken())
    assert report['status'] == 'complete'
    assert len(report['records']) == 2
    assert report['records'][1]['end_sample'] == 496000
    assert report['records'][0]['raw'] == response.text
    assert report['license']['license_file'] is None
    assert all(path.name == 'audio-window.wav' and not path.exists() for path, _ in calls)
    assert calls[0][1] == {'prompt': 'Describe only this audio window. Return one JSON object with foreground (yes, no, or uncertain) and native_genre (a specific audible genre string or null). Speech over incidental background music is foreground=no. If uncertain, use foreground=uncertain and native_genre=null.', 'enable_time_marker': False, 'temperature': 0, 'max_tokens': 2048}


@pytest.mark.parametrize('text,count,error', [('garbage', 5, 'InvalidMossOutput'), ('{"foreground":"no","native_genre":null}', 2048, 'IncompleteGeneration')])
def test_bad_response_preserves_raw_and_stops(tmp_path, assets, monkeypatch, text, count, error):
    audio, out, calls = run_fake(tmp_path, assets, monkeypatch, SimpleNamespace(text=text, generation_tokens=count, prompt_tokens=4, total_time=0.1))
    with pytest.raises(Exception) as caught:
        probe.probe_moss(audio, local_model=assets, out=out, token=CancellationToken())
    assert type(caught.value).__name__ == error
    report = json.loads(out.read_text())
    assert report['status'] == 'technical_error'
    assert report['records'][0]['raw'] == text
    assert 'summary' not in report
    assert len(calls) == 1


def test_cancel_after_return_preserves_raw_before_parsing(tmp_path, assets, monkeypatch):
    token = CancellationToken()
    audio, out, calls = run_fake(tmp_path, assets, monkeypatch, SimpleNamespace(text='garbage', generation_tokens=5, prompt_tokens=4, total_time=0.1), token)
    with pytest.raises(JobCancelled):
        probe.probe_moss(audio, local_model=assets, out=out, token=token)
    report = json.loads(out.read_text())
    assert report['status'] == 'cancelled'
    assert report['records'][0]['raw'] == 'garbage'
    assert len(calls) == 1


def test_existing_out_blocks_loading(tmp_path, assets, monkeypatch):
    audio = tmp_path / 'audio.wav'
    wav(audio, 1)
    out = tmp_path / 'out.json'
    out.write_text('original')
    monkeypatch.setattr(probe, '_load_model', lambda root: pytest.fail('must not load'))
    with pytest.raises(FileExistsError):
        probe.probe_moss(audio, local_model=assets, out=out, token=CancellationToken())
    assert out.read_text() == 'original'


@pytest.mark.parametrize('count', [None, True, -1, float('nan'), '5', pytest.param(10 ** 5000, id='oversized-integer')])
def test_invalid_response_token_metadata_is_technical(tmp_path, assets, monkeypatch, count):
    audio, out, calls = run_fake(tmp_path, assets, monkeypatch, SimpleNamespace(text='{"foreground":"no","native_genre":null}', generation_tokens=count))
    with pytest.raises(probe.InvalidMossOutput):
        probe.probe_moss(audio, local_model=assets, out=out, token=CancellationToken())
    report = json.loads(out.read_text())
    assert report['status'] == 'technical_error'
    assert len(calls) == 1


def test_inference_fault_keeps_prior_evidence_and_cleans_temp(tmp_path, assets, monkeypatch):
    audio = tmp_path / 'audio.wav'
    wav(audio)
    paths = []
    def generate(path, **kwargs):
        paths.append(Path(path))
        if len(paths) == 2:
            raise RuntimeError('inference failed')
        return SimpleNamespace(text='{"foreground":"no","native_genre":null}', generation_tokens=5)
    monkeypatch.setattr(probe, '_load_model', lambda root: SimpleNamespace(generate=generate))
    out = tmp_path / 'out.json'
    with pytest.raises(RuntimeError, match='inference failed'):
        probe.probe_moss(audio, local_model=assets, out=out, token=CancellationToken())
    report = json.loads(out.read_text())
    assert report['progress']['completed_windows'] == 1
    assert len(report['records']) == 1
    assert all(not path.exists() for path in paths)


def test_extraction_rejects_out_of_bounds_and_unsafe_name(tmp_path):
    audio = tmp_path / 'audio.wav'
    wav(audio, 1)
    with pytest.raises(ValueError, match='bounds'):
        probe.extract_pcm_window(audio, WINDOW, tmp_path)
    with pytest.raises(ValueError, match='sanitized'):
        probe.extract_pcm_window(audio, WINDOW, tmp_path, basename='../escape.wav')


def test_one_sample_tail_and_bounded_source_reads(tmp_path, assets, monkeypatch):
    audio = tmp_path / 'audio.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0' * 480001)
    reads = []
    original = wave.Wave_read.readframes
    def readframes(self, count):
        reads.append(count)
        return original(self, count)
    monkeypatch.setattr(wave.Wave_read, 'readframes', readframes)
    monkeypatch.setattr(probe, '_load_model', lambda root: SimpleNamespace(generate=lambda *args, **kwargs: SimpleNamespace(text='{"foreground":"no","native_genre":null}', generation_tokens=5)))
    report = probe.probe_moss(audio, local_model=assets, out=tmp_path / 'out.json', token=CancellationToken())
    assert reads == [480000, 1]
    assert report['records'][-1]['end_sample'] == 480001


@pytest.mark.parametrize('mutation', ['hash', 'symlink', 'python', 'extra', 'custom'])
def test_snapshot_rejects_untrusted_runtime(tmp_path, assets, mutation):
    if mutation == 'hash':
        (assets / 'config.json').write_text('{}')
    elif mutation == 'symlink':
        (assets / 'escape').symlink_to(tmp_path)
    elif mutation == 'python':
        (assets / 'code.py').write_text('pass')
    elif mutation == 'extra':
        (assets / 'other.json').write_text('{}')
    else:
        (assets / 'config.json').write_text('{"auto_map":{"AutoModel":"code.Model"}}')
        import hashlib
        probe.EXPECTED_ASSET_HASHES['config.json'] = hashlib.sha256((assets / 'config.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        probe._verify_snapshot(assets)


@pytest.mark.parametrize('mutation', [None, 'wrong-preprocessor', 'wrong-file', 'wrong-file-null'])
def test_snapshot_auto_maps_are_approved_per_file(tmp_path, monkeypatch, mutation):
    import hashlib
    root = tmp_path / 'model'
    root.mkdir()
    configs = {
        'config.json': {'auto_map': {
            'AutoConfig': 'configuration_moss_music.MossMusicConfig',
            'AutoProcessor': 'processing_moss_music.MossMusicProcessor',
        }},
        'preprocessor_config.json': {'auto_map': {
            'AutoProcessor': 'processing_moss_music.MossMusicProcessor',
        }},
    }
    if mutation == 'wrong-preprocessor':
        configs['preprocessor_config.json'] = configs['config.json']
    elif mutation == 'wrong-file':
        configs['tokenizer_config.json'] = configs['config.json']
    elif mutation == 'wrong-file-null':
        configs['tokenizer_config.json'] = {'auto_map': None}
    hashes = {}
    for name, config in configs.items():
        raw = json.dumps(config).encode()
        (root / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(probe, 'EXPECTED_ASSET_HASHES', hashes)
    if mutation:
        with pytest.raises(ValueError, match='unapproved custom implementation metadata'):
            probe._verify_snapshot(root)
    else:
        assert probe._verify_snapshot(root) == root.resolve()
