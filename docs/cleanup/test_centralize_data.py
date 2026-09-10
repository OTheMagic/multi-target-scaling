import gzip
import json
from pathlib import Path

import pytest

from docs.cleanup import centralize_data as migration


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(migration, 'ROOT', tmp_path)
    state = tmp_path / 'data/_organization/test'
    monkeypatch.setattr(migration, 'STATE', state)
    monkeypatch.setattr(migration, 'PLAN', state / 'manifest.json.gz')
    migration.safe_parent.cache_clear()
    return tmp_path


def put(root, name, content=b'data'):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def test_move_preserves_bytes_settings_figures_and_resume(workspace):
    put(workspace, 'envelope_method/results/demo/trial_000.json', b'{"records":[]}')
    put(workspace, 'envelope_method/results/demo/trials.csv', b'trial,coverage\n0,.9\n')
    put(workspace, 'envelope_method/settings.json', b'[]')
    put(workspace, 'manuscript/figures/plot.pdf', b'PDF fixture')
    put(workspace, 'tmp/scratch.csv')
    put(workspace, 'real_exps/data/raw.arff', b'raw dataset')
    put(workspace, 'study/data/trial.joblib', b'saved model')
    migration.prepare()
    before = migration.records()
    assert len(before) == 4
    migration.move()
    migration.move()
    migration.verify()
    for row in before:
        assert not (workspace / row['source']).exists()
        assert migration.digest(workspace / row['destination']) == row['sha256']
    assert (workspace / 'envelope_method/settings.json').read_bytes() == b'[]'
    assert (workspace / 'manuscript/figures/plot.pdf').read_bytes() == b'PDF fixture'
    assert (workspace / 'tmp/scratch.csv').exists()


def test_collision_is_rejected_before_any_move(workspace):
    put(workspace, 'results/a.csv', b'original')
    put(workspace, 'data/results/a.csv', b'existing')
    with pytest.raises(FileExistsError):
        migration.prepare()
    assert (workspace / 'results/a.csv').read_bytes() == b'original'
    assert (workspace / 'data/results/a.csv').read_bytes() == b'existing'


def test_changed_source_and_corrupt_destination_are_rejected(workspace):
    source = put(workspace, 'results/a.csv', b'original')
    migration.prepare()
    source.write_bytes(b'changed!')
    with pytest.raises(ValueError, match='Source changed'):
        migration.move()
    source.write_bytes(b'original')
    migration.move()
    (workspace / 'data/results/a.csv').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='verification'):
        migration.verify()


@pytest.mark.parametrize('name', ['../outside.csv', '.git/index', '.agents/config.json'])
def test_protected_paths_are_rejected(workspace, name):
    with pytest.raises(ValueError):
        migration.safe_path(name)
