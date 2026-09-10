"""Independent outcome-space metric checks on prespecified saved trials.

Regenerates data from seeds and forms response intervals directly; does not
call either experiment script's measure/replay/fit functions. Run after both
experiment scripts: python qa/audit_results.py (any working directory).
"""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPORT/'code'))
from report_paths import artifact, data_path
from data_model import generate
from utility.envelope import EnvelopeCalibration, envelope_prediction
from utility.res_rescaled import standardized_prediction
from utility.conformal_utils import conformal_rank


def old(cal, alpha):
    if conformal_rank(len(cal), alpha) > len(cal) or np.any(cal.std(0) == 0):
        return np.full(cal.shape[1], np.inf)
    with np.errstate(all='ignore'):
        return standardized_prediction(cal, alpha).upper


def equal(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-10)


def direct_metrics(a, bounds, empty, absolute=False):
    """Derive membership and widths from outcome-space interval endpoints."""
    y = a['Y_test']
    if absolute:
        lo = a['pred_test'] - bounds
        hi = a['pred_test'] + bounds
    else:
        lo = a['lo_test'] - bounds[a['group_test']]
        hi = a['hi_test'] + bounds[a['group_test']]
    e = np.broadcast_to(empty, (len(y),)) if np.ndim(empty) == 0 else empty[a['group_test']]
    assert np.array_equal(e, np.any(hi < lo, axis=1))
    lengths = np.maximum(hi-lo, 0)
    lengths[e] = 0
    hits = ((lo <= y) & (y <= hi)).all(1) & ~e
    with np.errstate(invalid='ignore'):
        vol = lengths.prod(1)
    vol[np.any(lengths == 0, axis=1)] = 0
    result = dict(coverage=hits.mean(), volume=vol.mean(),
                  empty=e.mean(), infinite=np.isinf(vol).mean())
    result.update({f'length_{j+1}':lengths[:,j].mean() for j in range(y.shape[1])})
    return result


def check_absolute():
    path = data_path('data/absolute')
    manifest = json.loads((path/'manifest.json').read_text())
    rows = pd.read_csv(path/'trials.csv')
    count = dict(archive_hashes=0, regenerated_fitted_trials=0,
                 recomputed_bounds=0, direct_outcome_metrics=0,
                 all_saved_coordinate_containment=0)
    for entry in manifest['configurations']:
        p = path/entry['archive']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256']
        count['archive_hashes'] += 1
        spec = entry['spec']
        with np.load(p) as saved:
            assert np.all(saved['env'] <= saved['old']+1e-8*(1+abs(saved['old'])))
            count['all_saved_coordinate_containment'] += saved['env'].size
            trial = 59  # Fixed before inspecting any trial's outcome.
            a = generate(int(saved['seed'][trial]), spec['family'], spec['n'],
                         d=spec['d'], base_alpha=spec['base_alpha'])
            equal(a['model_coef'], saved['model_coef'][trial])
            equal(a['abs_cal'], saved['cal'][trial])
            equal(a['abs_test'], saved['test'][trial])
            count['regenerated_fitted_trials'] += 1
            reg = envelope_prediction(a['abs_cal'], spec['alpha'])
            bounds = dict(env=reg.upper, old=old(a['abs_cal'],spec['alpha']))
            for method,b in bounds.items():
                equal(b,saved[method][trial])
                count['recomputed_bounds'] += 1
                metrics = direct_metrics(a,b,reg.empty if method=='env' else False,True)
                row = rows[(rows.config==entry['config'])&(rows.trial==trial)&(rows.method==method)].iloc[0]
                for key,value in metrics.items():
                    if key in row: equal(value,row[key])
                count['direct_outcome_metrics'] += 1
    return count


def check_signed():
    path = data_path('data/signed')
    source = json.loads((path/'source_manifest.json').read_text())
    hashes = {e['path']:e['sha256'] for e in json.loads((path/'manifest.json').read_text())}
    rows = pd.read_csv(path/'trials.csv')
    methods = ['signed_env','cap_old','constant_old','width_old']
    count = dict(archive_hashes=0, regenerated_fitted_trials=0,
                 recomputed_bounds=0, direct_outcome_metrics=0,
                 training_fixed_constant_shift=0, width_shift_identity=0)
    for entry in source:
        if entry['trial'] not in (0,59,119):
            continue
        p = artifact(entry['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[entry['path']]
        count['archive_hashes'] += 1
        spec = entry['spec']
        a = generate(entry['seed'],spec['family'],spec['n'],d=spec['d'],
                     base_alpha=spec['base_alpha'])
        with np.load(p) as saved:
            for key in ('raw_cal','raw_test','width_cal','width_test','fitted_error_quantiles'):
                equal(a[key],saved[key])
            count['regenerated_fitted_trials'] += 1
            half = (a['fitted_error_quantiles'][:,1]-a['fitted_error_quantiles'][:,0])/2
            constant = half.max(0)
            equal(constant,saved['constant_shift'])
            # Valid over both possible groups, not merely the observed cal set.
            assert np.all(constant[None,:] >= half)
            for split in ('cal','test'):
                midpoint = (a['lo_'+split]+a['hi_'+split])/2
                absolute = abs(a['Y_'+split]-midpoint)
                shifted = a['raw_'+split]+half[a['group_'+split]]
                equal(absolute,shifted)
                assert np.all(a['raw_'+split]+constant >= -1e-12)
                count['width_shift_identity'] += 1
            count['training_fixed_constant_shift'] += 1
            model = EnvelopeCalibration(a['raw_cal'],spec['alpha'])
            regs = [model.predict(-h) for h in half]
            bounds = dict(signed_env=np.array([r.upper for r in regs]),
                cap_old=np.tile(old(np.maximum(a['raw_cal'],0),spec['alpha']),(2,1)),
                constant_old=np.tile(old(np.maximum(a['raw_cal']+constant,0),spec['alpha'])-constant,(2,1)),
                width_old=old(np.maximum(a['raw_cal']+half[a['group_cal']],0),spec['alpha'])[None,:]-half)
            for method in methods:
                equal(bounds[method],saved['upper_'+method])
                count['recomputed_bounds'] += 1
                empty = np.any(bounds[method] < -half,axis=1)
                np.testing.assert_array_equal(empty,saved['empty_'+method])
                metrics = direct_metrics(a,bounds[method],empty)
                row = rows[(rows.config==entry['config'])&(rows.trial==entry['trial'])&(rows.method==method)].iloc[0]
                for key,value in metrics.items(): equal(value,row[key])
                count['direct_outcome_metrics'] += 1
    return count


def check_boundary_metrics():
    # Exercise the reporting helper only on synthetic edge cases; all selected
    # saved-trial metrics above are independently derived from actual outcomes.
    # Load only the actual NumPy metric function, keeping this bounded audit
    # independent of the plotting/SciPy dependencies imported by its module.
    tree = ast.parse((REPORT/'code/signed_experiments.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measure')
    methods = ['signed_env','cap_old','constant_old','width_old']
    namespace = {'np':np,'METHODS':methods}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'signed_experiments.measure','exec'),namespace)
    measure = namespace['measure']
    a = dict(group_test=np.array([0]), width_test=np.array([[2.,4.]]),
             raw_test=np.array([[-1.,-2.]]))
    for b,e,coverage in [(np.array([[-2.,np.inf]]),True,0.),
                         (np.array([[-1.,np.inf]]),False,1.)]:
        result = measure(a,{m:b for m in methods},{m:np.array([e]) for m in methods})
        for row in result:
            assert row['volume'] == 0. and row['coverage'] == coverage
            assert row['empty'] == float(e) and row['infinite'] == 0.
            assert row['length_1'] == 0.
            assert row['length_2'] == (0. if e else np.inf)
    return dict(empty_times_infinity=4, closed_zero_width_times_infinity=4)


if __name__ == '__main__':
    report = dict(status='passed', absolute=check_absolute(), signed=check_signed(),
        metric_edge_cases=check_boundary_metrics(),
        sampling='Absolute trial 59 in all 18 configurations; signed trials 0, 59, 119 in all three configurations.',
        method='Regenerated training/calibration/test data; direct response intervals, no experiment metric helper; bundled algorithm entry points.',
        scope='Finite reproducibility and metric audit, not an exhaustive numerical proof.')
    (data_path('qa/independent_audit.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
