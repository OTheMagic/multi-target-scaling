import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pytest
from envelope_method.cqr10 import runner
from envelope_method.experiments import evaluate_cqr


def tiny_config():
    cfg = runner.make_config(n_cal=15)
    cfg.update(n_train=50, n_validation=20, n_test=8, seed=909001)
    cfg['model_params'].update(n_estimators=3, n_iter_no_change=None, min_samples_leaf=3)
    return cfg


def test_grid():
    items = runner.experiment_grid()
    assert len(items) == 22
    assert sum(i['trials'] for i in items) == 1430
    assert len({runner.digest(i['config']) for i in items}) == len(items)
    assert all(i['config']['d'] == 10 for i in items)
    assert len(runner.experiment_grid(True)) == 26


def test_fresh_draws_and_seeds():
    cfg = tiny_config()
    a, sa = runner.draw_data(cfg, 0)
    b, sb = runner.draw_data(cfg, 1)
    assert len(set(sa.values())) == len(sa)
    np.testing.assert_array_equal(a['dgp_coef'], b['dgp_coef'])
    for split in ['train', 'validation', 'cal', 'test']:
        assert sa[split] != sb[split]
        assert not np.array_equal(a[f'X_{split}'], b[f'X_{split}'])
        assert a[f'y_{split}'].shape == (cfg[f'n_{split}'], 10)


def test_checkpoint_and_model_reproduction(tmp_path, monkeypatch):
    cfg = tiny_config()
    result = runner.run_trial(cfg, 0, tmp_path, storage='full')
    directory = tmp_path/runner.digest(cfg)[:16]
    assert len(result['quality']) == 20
    assert len(result['records']) == 9
    assert all('full' not in row['method'].lower() for row in result['records'])
    models = joblib.load(directory/'trial_0000.joblib')['models']
    with np.load(directory/'trial_0000.npz') as a:
        assert a['scores_test'].shape == (8, 10)
        pred = np.column_stack([pair[0].predict(a['X_test']) for pair in models])
        np.testing.assert_allclose(pred, a['quantile_lower_raw_test'])
        lo = np.minimum(a['quantile_lower_raw_test'], a['quantile_upper_raw_test'])
        hi = np.maximum(a['quantile_lower_raw_test'], a['quantile_upper_raw_test'])
        np.testing.assert_allclose(a['scores_test'], np.maximum(lo-a['y_test'], a['y_test']-hi))
    monkeypatch.setattr(runner, 'fit_models', lambda *args: pytest.fail('Resume refitted a model'))
    repeated = runner.run_trial(cfg, 0, tmp_path)
    assert repeated['files'] == result['files']
    assert not runner.summarize_config(cfg, tmp_path).empty
    archive = directory/'trial_0000.npz'
    with archive.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        runner.run_trial(cfg, 0, tmp_path)


@pytest.mark.parametrize('storage', runner.STORAGE_LEVELS)
def test_storage_contents_and_resume(storage, tmp_path, monkeypatch):
    cfg = tiny_config()
    out = tmp_path/'cluster output with spaces'
    result = runner.run_trial(cfg, 0, out, storage=storage)
    directory = out/runner.digest(cfg)[:16]
    assert result['storage'] == storage
    assert set(result['files']) == runner.checkpoint_files(0, storage)
    assert {p.name for p in directory.iterdir()} == {'config.json', 'trial_0000.json', *result['files']}
    assert result['seeds'] == runner.trial_seeds(cfg, 0)
    assert result['array_fingerprints']['X_train']['shape'] == [50, 5]
    if storage == 'scores':
        with np.load(directory/'trial_0000.npz') as saved:
            assert {'scores_cal', 'scores_test', 'base_lengths_cal', 'base_lengths_test'} <= set(saved.files)
            assert all(k.startswith(('scores_', 'base_lengths_', 'bound__')) for k in saved.files)
    monkeypatch.setattr(runner, 'fit_models', lambda *args: pytest.fail('Resume refitted a model'))
    assert runner.digest(runner.run_trial(cfg, 0, out, storage=storage)) == runner.digest(result)
    assert runner.digest(runner.run_trial(cfg, 0, out, storage='compact')) == runner.digest(result)
    summary = runner.summarize_config(cfg, out)
    assert ('outcome_volume', 'median') in summary.columns
    assert ('outcome_volume', 'q95') in summary.columns
    assert (directory/'volume_states.csv').exists()


def test_storage_does_not_change_scientific_results(tmp_path):
    cfg = tiny_config()
    outputs = [runner.run_trial(cfg, 0, tmp_path/tier, storage=tier) for tier in runner.STORAGE_LEVELS]
    for result in outputs[1:]:
        assert result['config_hash'] == outputs[0]['config_hash']
        assert result['seeds'] == outputs[0]['seeds']
        assert result['array_fingerprints'] == outputs[0]['array_fingerprints']
        for first, second in zip(outputs[0]['records'], result['records']):
            for field in first.keys() - {'runtime'}:
                if isinstance(first[field], (int, float, list)):
                    np.testing.assert_allclose(first[field], second[field], rtol=0, atol=0)
                else:
                    assert first[field] == second[field]
        for first, second in zip(outputs[0]['quality'], result['quality']):
            assert {k: v for k, v in first.items() if k != 'fit_seconds'} == {
                k: v for k, v in second.items() if k != 'fit_seconds'}


@pytest.mark.parametrize('storage', runner.STORAGE_LEVELS)
def test_checkpoint_metadata_integrity(storage, tmp_path):
    cfg = tiny_config()
    runner.run_trial(cfg, 0, tmp_path, storage=storage)
    path = tmp_path/runner.digest(cfg)[:16]/'trial_0000.json'
    changed = json.loads(path.read_text())
    changed['records'][0]['test_coverage'] = -123
    runner.dump(path, changed)
    with pytest.raises(ValueError, match='metadata checksum'):
        runner.verified_checkpoint(path.parent, cfg, 0)


@pytest.mark.parametrize('storage,missing', [('scores', '.npz'), ('full', '.npz'), ('full', '.joblib')])
def test_missing_required_artifact_rejected(storage, missing, tmp_path):
    cfg = tiny_config()
    runner.run_trial(cfg, 0, tmp_path, storage=storage)
    directory = tmp_path/runner.digest(cfg)[:16]
    (directory/f'trial_0000{missing}').unlink()
    with pytest.raises(ValueError, match='required file missing'):
        runner.run_trial(cfg, 0, tmp_path, storage=storage)


def test_storage_upgrade_is_explicit(tmp_path, monkeypatch):
    cfg = tiny_config()
    runner.run_trial(cfg, 0, tmp_path)
    monkeypatch.setattr(runner, 'fit_models', lambda *args: pytest.fail('Upgrade silently refitted'))
    with pytest.raises(ValueError, match='separate output directory'):
        runner.run_trial(cfg, 0, tmp_path, storage='full')


def test_missing_file_index_rejected(tmp_path):
    cfg = tiny_config()
    result = runner.run_trial(cfg, 0, tmp_path, storage='scores')
    directory = tmp_path/runner.digest(cfg)[:16]
    result['files'] = {}
    result['checkpoint_sha256'] = runner.digest({k: v for k, v in result.items() if k != 'checkpoint_sha256'})
    runner.dump(directory/'trial_0000.json', result)
    with pytest.raises(ValueError, match='required files'):
        runner.verified_checkpoint(directory, cfg, 0)


def test_compact_does_not_hide_incomplete_full_artifacts(tmp_path):
    cfg = tiny_config()
    runner.run_trial(cfg, 0, tmp_path, storage='full')
    directory = tmp_path/runner.digest(cfg)[:16]
    (directory/'trial_0000.json').unlink()
    with pytest.raises(ValueError, match='Unfinished richer artifacts'):
        runner.run_trial(cfg, 0, tmp_path)


def test_manifest_change_rejected(tmp_path):
    cfg = tiny_config()
    runner.run_trial(cfg, 0, tmp_path)
    directory = tmp_path/runner.digest(cfg)[:16]
    manifest = json.loads((directory/'config.json').read_text())
    manifest['implementation_hash'] = 'changed'
    runner.dump(directory/'config.json', manifest)
    with pytest.raises(ValueError, match='changed'):
        runner.run_trial(cfg, 1, tmp_path)


def test_rank_and_backward_equivalent():
    rng = np.random.default_rng(901)
    cal, test = rng.normal(size=(30, 10)), rng.normal(size=(4, 10))
    lc, lt = np.full_like(cal, 4), np.full_like(test, 4)
    _, a = evaluate_cqr(cal, test, lc, lt, .1, [], envelope_search='rank')
    _, b = evaluate_cqr(cal, test, lc, lt, .1, [], envelope_search='backward')
    for key in a:
        np.testing.assert_allclose(a[key], b[key])


def test_no_calibration_or_test_used_for_fitting():
    cfg = tiny_config()
    a, seeds = runner.draw_data(cfg, 0)
    b = copy.deepcopy(a)
    b['y_cal'][:] = 1e12
    b['y_test'][:] = -1e12
    b['y_validation'][:] = 5e12
    ma, _ = runner.fit_models(cfg, a, seeds['fit'])
    mb, _ = runner.fit_models(cfg, b, seeds['fit'])
    for pa, pb in zip(ma, mb):
        for qa, qb in zip(pa, pb):
            np.testing.assert_allclose(qa.predict(a['X_test']), qb.predict(a['X_test']))


def test_two_process_local_execution(tmp_path):
    a, b = tiny_config(), tiny_config()
    b['base_alpha'] = .5
    items = [dict(config=cfg, trials=1) for cfg in [a, b]]
    completed = runner.run_study(items, tmp_path, workers=2)
    assert len(completed) == 2
    for cfg in [a, b]:
        assert runner.verified_checkpoint(tmp_path/runner.digest(cfg)[:16], cfg, 0)['status'] == 'complete'


def test_runtime_plan_has_no_double_counting():
    from envelope_method.cqr10.planning import estimate_runtime, experiment_totals
    table = estimate_runtime(runner.experiment_grid())
    totals = experiment_totals(table)
    assert totals.trials.sum() == table.trials.sum() == 1430
    assert totals.serial_hours.sum() == pytest.approx(table.serial_hours.sum())
    assert np.all(table.seconds_per_trial > table.fit_seconds)
    changed = runner.experiment_grid()[:1]
    changed[0]['config']['n_train'] = 20000
    with pytest.raises(ValueError, match='n_train'):
        estimate_runtime(changed)
