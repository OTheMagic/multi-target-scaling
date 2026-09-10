"""Versioned storage tiers for simulation checkpoints.

Scientific trial IDs and measurements are independent of storage. ``scores``
retains every existing array except the six original X/y observation arrays;
``compact`` retains the JSON measurements and provenance only. Old checkpoints
without storage metadata continue to mean ``full``.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

STORAGE_MODES = ('compact', 'scores', 'full')
RAW_OBSERVATION_KEYS = frozenset(
    f'{variable}_{split}' for variable in ('X', 'y') for split in ('train', 'cal', 'test'))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def records_sha256(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':'),
                                     default=str).encode('utf-8')).hexdigest()


def get_storage_mode(checkpoint):
    if 'storage' in checkpoint and checkpoint['storage'].get('schema_version') != 1:
        raise ValueError('Unsupported checkpoint storage schema version.')
    mode = checkpoint.get('storage', {}).get('mode', 'full')
    if mode not in STORAGE_MODES:
        raise ValueError(f'Unknown checkpoint storage mode: {mode!r}')
    return mode


def source_archive_sha256(checkpoint):
    return (checkpoint.get('storage', {}).get('source_archive_sha256')
            or checkpoint.get('archive_sha256'))


def archive_hash_matches(checkpoint, expected):
    """Check provenance links, not archive bytes; validate_archive checks bytes."""
    return bool(expected) and expected in {
        checkpoint.get('archive_sha256'), source_archive_sha256(checkpoint)}


def validate_records(checkpoint):
    if 'storage' in checkpoint:
        get_storage_mode(checkpoint)
        if not checkpoint.get('records') or not checkpoint['storage'].get('records_sha256'):
            raise ValueError('Versioned checkpoint requires trial measurements and their checksum.')
    expected = checkpoint.get('storage', {}).get('records_sha256')
    if expected and expected != records_sha256(checkpoint['records']):
        raise ValueError('Checkpoint trial measurements do not match their saved checksum.')


def validate_archive(path, checkpoint, required_keys=(), verify_hash=True):
    path = Path(path)
    validate_records(checkpoint)
    mode = get_storage_mode(checkpoint)
    if mode == 'compact':
        raise FileNotFoundError(
            f'{path}: compact checkpoint retains measurements only. This operation '
            'requires saved arrays; use a scores/full archive or refit in a separate output directory.')
    if not path.is_file():
        raise FileNotFoundError(f'Missing {mode} checkpoint archive: {path}')
    expected = checkpoint.get('archive_sha256')
    if 'storage' in checkpoint:
        declared = checkpoint['storage']
        if not expected or declared.get('archive_sha256') != expected or not isinstance(declared.get('retained_keys'), list):
            raise ValueError(f'Versioned checkpoint requires matching archive checksums and member metadata: {path}')
    if verify_hash and expected and sha256_file(path) != expected:
        raise ValueError(f'Archive checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as arrays:
        keys = set(arrays.files)
        missing = set(required_keys) - keys
        if missing:
            raise ValueError(f'{path}: {mode} archive lacks required arrays {sorted(missing)}. '
                             'Raw-data/refitting diagnostics require full storage.')
        declared = checkpoint.get('storage', {}).get('retained_keys')
        if declared is not None and set(declared) != keys:
            raise ValueError(f'Archive member list differs from checkpoint: {path}')
        if mode == 'scores' and keys & RAW_OBSERVATION_KEYS:
            raise ValueError(f'Scores archive unexpectedly contains raw observations: {path}')


@contextmanager
def load_archive(path, checkpoint=None, required_keys=(), verify_hash=True):
    if checkpoint is None:
        checkpoint = json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
    validate_archive(path, checkpoint, required_keys=required_keys, verify_hash=verify_hash)
    with np.load(path, allow_pickle=False) as arrays:
        yield arrays


def require_storage(checkpoint, requested):
    """Richer retained evidence satisfies a lean request; upgrades need refitting."""
    if requested not in STORAGE_MODES:
        raise ValueError(f'Unknown requested storage mode: {requested!r}')
    actual = get_storage_mode(checkpoint)
    if STORAGE_MODES.index(actual) < STORAGE_MODES.index(requested):
        raise ValueError(f'Existing checkpoint is {actual}, but {requested} was requested. '
                         'Use a separate output directory to refit with richer retention; '
                         'storage upgrades never silently reuse incomplete evidence.')
    validate_records(checkpoint)


def save_trial_archive(path, arrays, mode='scores'):
    """Atomically save selected arrays. Return metadata for the JSON written last."""
    if mode not in STORAGE_MODES:
        raise ValueError(f'Unknown storage mode: {mode!r}')
    path = Path(path)
    if mode == 'compact':
        if path.exists():
            raise FileExistsError(f'Refusing to orphan an existing archive in compact mode: {path}')
        return dict(schema_version=1, mode=mode, retained_keys=[], archive_bytes=0,
                    archive_sha256=None)
    retained = {key: value for key, value in arrays.items()
                if mode == 'full' or key not in RAW_OBSERVATION_KEYS}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, **retained)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return dict(schema_version=1, mode=mode, retained_keys=sorted(retained),
                archive_bytes=path.stat().st_size, archive_sha256=sha256_file(path))
