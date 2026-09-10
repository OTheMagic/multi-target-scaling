"""Audit saved timing evidence and execute the manual notebook with training blocked."""
import contextlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from envelope_method.experiments import dump, metric
from envelope_method.cqr10 import runner


def main():
    pilot = json.loads((runner.DATA/'timing_pilot/study.json').read_text())
    checked = []
    for row in pilot['rows']:
        cfg = row['config']
        directory = runner.DATA/'timing_pilot/full_trials'/row['config_id']
        result = runner.verified_checkpoint(directory, cfg, 0)
        assert result is not None
        fresh, seeds = runner.draw_data(cfg, 0)
        assert seeds == result['seeds']
        models = joblib.load(directory/'trial_0000.joblib')['models']
        with np.load(directory/'trial_0000.npz') as archive:
            for key, value in fresh.items():
                np.testing.assert_array_equal(value, archive[key])
            for split in ['validation', 'cal', 'test']:
                lo = np.column_stack([pair[0].predict(archive[f'X_{split}']) for pair in models])
                hi = np.column_stack([pair[1].predict(archive[f'X_{split}']) for pair in models])
                np.testing.assert_array_equal(lo, archive[f'quantile_lower_raw_{split}'])
                np.testing.assert_array_equal(hi, archive[f'quantile_upper_raw_{split}'])
                lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                np.testing.assert_array_equal(hi-lo, archive[f'base_lengths_{split}'])
                np.testing.assert_array_equal(np.maximum(lo-archive[f'y_{split}'], archive[f'y_{split}']-hi),
                                              archive[f'scores_{split}'])
            base = archive['base_lengths_test']
            for record in result['records']:
                name = record['method']
                bound = 0. if name == 'Base' else archive[f'bound__{name}']
                assert not np.isnan(bound).any()
                recalculated = metric(name, bound, archive['scores_test'], base, 0.,
                                      np.any(base+2*bound < 0, axis=1))
                for key in ['test_coverage', 'outcome_volume', 'mean_log_volume', 'empty_rate']:
                    np.testing.assert_allclose(recalculated[key], record[key], rtol=1e-12)
        checked.append(dict(config_id=row['config_id'], arrays_and_models_verified=True,
                            quality_flag=result['quality_flag'], files=result['files']))
    notebook_path = runner.HERE/'cqr10_experiments.ipynb'
    notebook = json.loads(notebook_path.read_text())
    assert notebook['nbformat'] == 4 and notebook['nbformat_minor'] == 5
    assert len({c['id'] for c in notebook['cells']}) == len(notebook['cells'])
    assert notebook['metadata']['kernelspec']['name'] == 'python3'
    for cell in notebook['cells']:
        assert cell['cell_type'] in ['markdown', 'code']
        assert isinstance(cell['source'], list) and isinstance(cell['metadata'], dict)
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and cell['outputs'] == []
    schema_valid = None
    try:
        import nbformat
    except ImportError:
        pass
    else:
        nbformat.validate(nbformat.read(notebook_path, as_version=4))
        schema_valid = True
    scope = {'__name__': '__notebook_validation__'}
    with patch.object(runner, 'run_study', side_effect=AssertionError('Notebook unexpectedly ran the study')), \
         patch.object(runner, 'fit_models', side_effect=AssertionError('Notebook unexpectedly trained')), \
         contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), f'{notebook_path.name}:cell{index}', 'exec'), scope)
    assert scope['RUN_EXPERIMENTS'] is False
    report = dict(status='passed', full_pilot_trials=len(checked), fitted_models=20*len(checked),
                  pilot_checks=checked, notebook_structure_checked=True, notebook_schema_valid=schema_valid,
                  notebook_executed_with_training_blocked=True,
                  notebook_sha256=runner.sha256(notebook_path),
                  notebook_code_cells=sum(c['cell_type'] == 'code' for c in notebook['cells']),
                  formal_study_started=runner.DEFAULT_OUT.exists())
    dump(runner.DATA/'verification.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'pilot_checks'}, indent=2))


if __name__ == '__main__':
    main()
