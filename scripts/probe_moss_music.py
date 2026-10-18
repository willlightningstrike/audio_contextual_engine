"""Isolated pinned MOSS diagnostic; does not admit models or map IAB genres."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import resource
import tempfile
import time
import wave

from engine.cancel import CancellationToken, JobCancelled
from engine.domain import TimeRange
from engine.signals.acoustic_probe import AcousticObservation, schedule_audio_windows


MODEL_ID = 'moss-music-8b-thinking-8bit'
REVISION = '7b2c41e689a46bbd7b395619ce9fd96041ff984c'
MAX_TOKENS = 2048
FIXED_MUSIC_PROMPT = (
    'Describe only this audio window. Return one JSON object with foreground '
    '(yes, no, or uncertain) and native_genre (a specific audible genre string or null). '
    'Speech over incidental background music is foreground=no. '
    'If uncertain, use foreground=uncertain and native_genre=null.'
)
EXPECTED_ASSET_HASHES = {
    'model.safetensors': '6da829b420899501d03c72658eac3850875b98e299ac712e42588c6eae126cda',
    'added_tokens.json': '1fabcc983ee0051b922009b1b5a7af1844385f38783e62679ac35d6dd2aa3355',
    'config.json': 'e3af4a08c245a056f1cf8f88a9c6d185a638bb2f030d30807c9f56c39dfeca79',
    'generation_config.json': '7914bf1e97eebd99700a1e0fa689a2cc66449ee748bf0d8eba69ea8470a5a83b',
    'preprocessor_config.json': '365bb03467925a83c979779d04a8767a7dc057f6e138691c06820884a70849b6',
    'special_tokens_map.json': '76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd',
    'tokenizer.json': 'fe09df9d0df539ba3c7c8812750bda67907c889d0fbcc500454ba7037cc34466',
    'tokenizer_config.json': '0869e41f5d123ff144a811f0d83c5d18871dcd4b4064f46bf9def194bfbc6f41',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'README.md': '93fb33664c2339cd68a783fa6992f446f0f9c71a8c6bbf6511db164effc5be99',
    'chat_template.jinja': '87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5',
    '.gitattributes': '34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930',
}


class InvalidMossOutput(ValueError):
    """Returned model evidence violates the strict observation contract."""


class IncompleteGeneration(InvalidMossOutput):
    """STTOutput has no finish reason, so reaching the cap is incomplete."""


def strict_json_object(raw: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InvalidMossOutput('duplicate JSON key')
            result[key] = value
        return result

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            raise InvalidMossOutput('nonfinite JSON number')
        return value

    def constant(text):
        raise InvalidMossOutput('nonfinite JSON constant')

    try:
        if type(raw) is not str:
            raise InvalidMossOutput('output must be text')
        value = json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float, parse_constant=constant)
        if type(value) is not dict:
            raise InvalidMossOutput('expected one JSON object')
        return value
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise InvalidMossOutput(str(exc)) from exc


def parse_moss_observation(raw: str, window: TimeRange) -> AcousticObservation:
    value = strict_json_object(raw)
    if set(value) != {'foreground', 'native_genre'}:
        raise InvalidMossOutput('unexpected MOSS fields')
    foreground, genre = value['foreground'], value['native_genre']
    if type(foreground) is not str or foreground not in {'yes', 'no', 'uncertain'}:
        raise InvalidMossOutput('invalid foreground')
    if genre is not None and (type(genre) is not str or not genre.strip()):
        raise InvalidMossOutput('invalid genre')
    if foreground != 'yes' and genre is not None:
        raise InvalidMossOutput('nonforeground audio cannot name a genre')
    try:
        return AcousticObservation(time_range=window, foreground=foreground,
                                   native_genre=genre, raw=raw, model_id=MODEL_ID)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InvalidMossOutput(str(exc)) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _verify_snapshot(path: Path) -> Path:
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('model must be a directory')
    for entry in root.rglob('*'):
        if not entry.resolve(strict=True).is_relative_to(root):
            raise ValueError('model symlink escapes snapshot')
        if entry.suffix.lower() in {'.py', '.pyc', '.pyo'}:
            raise ValueError('custom Python is forbidden')
        if entry.relative_to(root).parts[0] == '.cache':
            continue
        if entry.is_file() and entry.relative_to(root).as_posix() not in EXPECTED_ASSET_HASHES:
            raise ValueError('unmanifested runtime file')
    for name, expected in EXPECTED_ASSET_HASHES.items():
        if _sha256(root / name) != expected:
            raise ValueError(f'asset SHA-256 mismatch: {name}')
        if name.endswith('.json'):
            config = strict_json_object((root / name).read_text(encoding='utf-8'))
            approved_maps = {
                'config.json': {
                    'AutoConfig': 'configuration_moss_music.MossMusicConfig',
                    'AutoProcessor': 'processing_moss_music.MossMusicProcessor',
                },
                'preprocessor_config.json': {
                    'AutoProcessor': 'processing_moss_music.MossMusicProcessor',
                },
            }
            if 'auto_map' in config and (
                name not in approved_maps or config['auto_map'] != approved_maps[name]
            ):
                raise ValueError('unapproved custom implementation metadata')
    return root


def _load_model(root: Path):
    # Native registered moss_music implementation; no remote Python execution.
    from mlx_audio.stt import load
    return load(str(root))


def _environment() -> dict:
    result = {}
    for package in ('mlx-audio', 'mlx', 'numpy', 'pydantic', 'mlx-lm', 'transformers'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _mlx_peak():
    # Import only when live loading has already brought MLX into the process.
    import sys
    mx = sys.modules.get('mlx.core')
    return mx.get_peak_memory() if mx is not None else None


def extract_pcm_window(audio: Path, window: TimeRange, work_dir: Path, *, basename='audio-window.wav') -> Path:
    if basename != 'audio-window.wav':
        raise ValueError('window filename must be sanitized')
    with wave.open(str(audio), 'rb') as source:
        rate = source.getframerate()
        start, end = round(window.start_seconds * rate), round(window.end_seconds * rate)
        if not 0 <= start < end <= source.getnframes():
            raise ValueError('window outside exact source-frame bounds')
        source.setpos(start)
        raw = source.readframes(end - start)
        if len(raw) != (end - start) * source.getnchannels() * source.getsampwidth():
            raise ValueError('short PCM read')
        path = work_dir / basename
        with wave.open(str(path), 'wb') as output:
            output.setparams(source.getparams())
            output.writeframes(raw)
    with wave.open(str(path), 'rb') as decoded:
        if decoded.getnframes() != end - start or decoded.getframerate() != rate:
            raise ValueError('decoded window duration mismatch')
    return path


def probe_moss(audio: Path, *, local_model: Path, out: Path, token: CancellationToken) -> dict:
    path = Path(audio).resolve(strict=True)
    with wave.open(str(path), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('probe requires mono 16 kHz PCM16 WAV')
        frames, rate = source.getnframes(), source.getframerate()
    windows = schedule_audio_windows(frames / rate)
    report = {'schema_version': 1, 'kind': 'moss_acoustic_diagnostic', 'status': 'running',
              'model_id': MODEL_ID, 'audio': str(path), 'sample_count': frames, 'sample_rate': rate,
              'duration_seconds': frames / rate, 'records': [], 'errors': [],
              'progress': {'completed_windows': 0, 'expected_windows': len(windows)},
              'source_provenance': {'repository': 'mlx-community/MOSS-Music-8B-Thinking-8bit', 'revision': REVISION,
                                    'conversion': 'community 8-bit conversion',
                                    'upstream': 'OpenMOSS-Team/MOSS-Music-8B-Thinking'},
              'license': {'name': 'Apache-2.0', 'license_file': None,
                          'cards': [f'https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-8bit/blob/{REVISION}/README.md',
                                    'https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking']},
              'generation': {'temperature': 0, 'max_tokens': MAX_TOKENS, 'enable_time_marker': False},
              'environment': _environment(), 'asset_hashes': {}}
    started = time.monotonic()
    with Path(out).open('x', encoding='utf-8') as output:
        def persist():
            report['metrics'] = {'elapsed_seconds': time.monotonic() - started,
                                 'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 'mlx_peak_bytes': _mlx_peak()}
            output.seek(0)
            output.write(json.dumps(report, allow_nan=False, indent=2) + '\n')
            output.truncate()
            output.flush()
        current = None
        persist()
        try:
            token.raise_if_cancelled()
            root = _verify_snapshot(Path(local_model))
            report['local_model'] = str(root)
            report['asset_hashes'] = dict(EXPECTED_ASSET_HASHES)
            token.raise_if_cancelled()
            model = _load_model(root)
            with tempfile.TemporaryDirectory(prefix='moss-audio-') as temporary:
                for window in windows:
                    token.raise_if_cancelled()
                    current = {'start_seconds': window.start_seconds, 'end_seconds': window.end_seconds,
                               'start_sample': round(window.start_seconds * rate), 'end_sample': round(window.end_seconds * rate)}
                    window_wav = extract_pcm_window(path, window, Path(temporary))
                    token.raise_if_cancelled()
                    response = model.generate(str(window_wav), prompt=FIXED_MUSIC_PROMPT,
                                              enable_time_marker=False, temperature=0, max_tokens=MAX_TOKENS)
                    raw = response.text
                    record = {**current, 'raw': raw if type(raw) is str else f'<non-text:{type(raw).__name__}>', 'status': 'returned'}
                    report['records'].append(record)
                    persist()
                    token.raise_if_cancelled()
                    count = response.generation_tokens
                    if type(count) is not int or not 0 <= count <= 2 ** 63 - 1:
                        raise InvalidMossOutput('invalid generation token count')
                    record['generation_tokens'] = count
                    if count >= MAX_TOKENS:
                        raise IncompleteGeneration('INCOMPLETE_GENERATION: token cap reached')
                    observation = parse_moss_observation(raw, window)
                    record.update(status='parsed', observation=observation.model_dump(mode='json'))
                    report['progress']['completed_windows'] += 1
                    persist()
                    token.raise_if_cancelled()
            report['status'] = 'complete'
            token.run_if_not_cancelled(persist)
            return report
        except Exception as exc:
            report['status'] = 'cancelled' if isinstance(exc, JobCancelled) else 'technical_error'
            report['errors'].append({'type': type(exc).__name__, 'message': str(exc), 'interval': current})
            persist()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--local-model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    probe_moss(args.audio, local_model=args.local_model, out=args.out, token=CancellationToken())


if __name__ == '__main__':
    main()
