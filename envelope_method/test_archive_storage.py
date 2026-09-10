"""Storage changes must preserve scientific measurements and resume integrity."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from envelope_method import experiments
from envelope_method.archive_storage import (RAW_OBSERVATION_KEYS, archive_hash_matches,
    get_storage_mode, records_sha256, save_trial_archive, source_archive_sha256,
    validate_archive, validate_records)


def spec(kind='absolute'):
    cfg = dict(kind=kind, d=2, n_cal=20, alpha=.1, noise_type='Gaussian',
               noise_levels=[2, 1], n_features=2, n_train=50, n_test=12,
               redraw=True, generator_kwargs={})
    if kind == 'cqr':
        cfg.update(base_alpha=.1, quantile_params=dict(n_estimators=5, max_depth=1,
                    learning_rate=.1, min_samples_leaf=3))
    return dict(id='storage_test_' + kind, config=cfg, trials=2,
                transforms=[dict(name='capped', shift=0.)] if kind == 'cqr' else [], sources=['test'])


def scientific_rows(path):
    ignored = {'runtime', 'fit_seconds'}
    return [{key: value for key, value in row.items() if key not in ignored}
            for row in json.loads(path.read_text())['records']]


@pytest.mark.parametrize('kind', ['absolute', 'cqr'])
def test_storage_preserves_results_and_fresh_trials(tmp_path, kind):
    item = spec(kind)
    checkpoints = {}
    for mode in ('full', 'scores', 'compact'):
        root = tmp_path / mode
        experiments.run_spec(item, mode, root)
        folder = root / kind / item['id']
        checkpoints[mode] = folder
        first = json.loads((folder / 'trial_000.json').read_text())
        assert get_storage_mode(first) == mode
        validate_records(first)
        if mode == 'compact':
            assert not list(folder.glob('*.npz'))
        else:
            validate_archive(folder / 'trial_000.npz', first, ['scores_cal', 'scores_test'])
            with np.load(folder / 'trial_000.npz') as arrays:
                assert bool(RAW_OBSERVATION_KEYS & set(arrays.files)) == (mode == 'full')
        before = (folder / 'trial_000.json').read_bytes()
        experiments.run_spec(item, mode, root)
        assert before == (folder / 'trial_000.json').read_bytes()
        second = json.loads((folder / 'trial_001.json').read_text())
        assert first['records'][0]['training_seed'] != second['records'][0]['training_seed']
    for trial in range(2):
        name = f'trial_{trial:03d}.json'
        expected = scientific_rows(checkpoints['full'] / name)
        # JSON canonicalization also compares intentional NaN timings already removed.
        for mode in ('scores', 'compact'):
            assert json.dumps(scientific_rows(checkpoints[mode] / name), sort_keys=True) == json.dumps(expected, sort_keys=True)
    for mode in ('scores', 'compact'):
        with pytest.raises(ValueError, match='separate output directory'):
            experiments.run_spec(item, 'full', tmp_path / mode)


def test_missing_archive_is_not_silently_skipped(tmp_path):
    item = spec()
    experiments.run_spec(item, 'scores', tmp_path)
    archive = tmp_path / 'absolute' / item['id'] / 'trial_000.npz'
    archive.unlink()
    with pytest.raises(FileNotFoundError, match='Missing scores'):
        experiments.run_spec(item, 'scores', tmp_path)


def test_corrupt_metrics_and_arrays_are_detected(tmp_path):
    arrays = dict(scores_cal=np.arange(12).reshape(6, 2), scores_test=np.ones((3, 2)), X_train=np.ones((9, 2)))
    archive = tmp_path / 'trial_000.npz'
    storage = save_trial_archive(archive, arrays, 'scores')
    checkpoint = dict(records=[dict(test_coverage=.9)], storage=storage,
                      archive_sha256=storage['archive_sha256'])
    storage['records_sha256'] = records_sha256(checkpoint['records'])
    validate_archive(archive, checkpoint)
    checkpoint['records'][0]['test_coverage'] = 1.
    with pytest.raises(ValueError, match='measurements'):
        validate_archive(archive, checkpoint)
    checkpoint['records'][0]['test_coverage'] = .9
    with archive.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        validate_archive(archive, checkpoint)


def test_original_hash_is_provenance_not_active_validation(tmp_path):
    archive = tmp_path / 'trial.npz'
    storage = save_trial_archive(archive, dict(scores_cal=np.ones((3, 2))), 'scores')
    storage['source_archive_sha256'] = 'original'
    checkpoint = dict(records=[{'coverage': .9}], storage=storage, archive_sha256=storage['archive_sha256'])
    storage['records_sha256'] = records_sha256(checkpoint['records'])
    assert source_archive_sha256(checkpoint) == 'original'
    assert archive_hash_matches(checkpoint, 'original')
    assert archive_hash_matches(checkpoint, storage['archive_sha256'])
    assert not archive_hash_matches(checkpoint, 'unrelated')
    validate_archive(archive, checkpoint)
    with pytest.raises(ValueError, match='required arrays'):
        validate_archive(archive, checkpoint, ['X_train'])


def test_versioned_checkpoints_require_checksums(tmp_path):
    archive = tmp_path / 'trial.npz'
    storage = save_trial_archive(archive, dict(scores_cal=np.ones((3, 2))), 'scores')
    checkpoint = dict(records=[{'coverage': .9}], storage=storage, archive_sha256=storage['archive_sha256'])
    with pytest.raises(ValueError, match='measurements and their checksum'):
        validate_records(checkpoint)
    storage['records_sha256'] = records_sha256(checkpoint['records'])
    checkpoint.pop('archive_sha256')
    with pytest.raises(ValueError, match='matching archive checksums'):
        validate_archive(archive, checkpoint)


def test_new_checkpoint_rejects_environment_drift(tmp_path, monkeypatch):
    item = spec()
    experiments.run_spec(item, 'compact', tmp_path)
    monkeypatch.setattr(experiments, 'execution_provenance', lambda: {'changed': True})
    with pytest.raises(ValueError, match='Code/environment changed'):
        experiments.run_spec(item, 'compact', tmp_path)


def test_configuration_lock_blocks_overlapping_writers(tmp_path):
    item = spec()
    dest = tmp_path / 'absolute' / item['id']
    dest.mkdir(parents=True)
    lock = dest / '.run_spec.lock'
    lock.write_text('another worker')
    with pytest.raises(RuntimeError, match='Configuration locked'):
        experiments.run_spec(item, 'scores', tmp_path)
    assert lock.read_text() == 'another worker'
    assert not (dest / 'config.json').exists()
