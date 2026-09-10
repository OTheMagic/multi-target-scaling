"""Rebuild the real comparison with consistent full outcome-space volumes."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utility.project_paths import resolve_artifact
from utility.envelope import envelope_prediction
from utility.data_splitting import data_spliting_CHR_prediction

OUT = ROOT / 'data/envelope_method/results'
REPORT_OUT = ROOT / 'envelope_method/results'
REPORT_OUT.mkdir(parents=True, exist_ok=True)
rows, audit = [], []
for folder in sorted((OUT / 'real').iterdir()):
    if not folder.is_dir():
        continue
    files = sorted(folder.glob('split_*.json'))
    assert len(files) == 200
    for p in files:
        payload = json.loads(p.read_text())
        source = resolve_artifact(payload['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == payload['source_sha256']
        with np.load(source) as source_data, np.load(p.with_suffix('.npz')) as bounds:
            test = source_data['scores_test']
            assert len(payload['trials']) == 15
            assert {(r['alpha'], r['method']) for r in payload['trials']} == {(a,m) for a in [.1,.3,.5,.7,.9] for m in ['Envelope','TSCP_R','Signed_GWC']}
            for r in payload['trials']:
                u = bounds[f"{r['method']}_{r['alpha']}"]
                cov = ((test <= u).all(axis=1) & ~np.bool_(r['empty'])).mean()
                vol = 0. if r['empty'] else np.prod(2*u)
                np.testing.assert_allclose([cov,vol], [r['test_coverage'],r['outcome_volume']], rtol=1e-10)
                rows.append(r)
            if p == files[0]:
                for a in [.1,.3,.5,.7,.9]:
                    current = envelope_prediction(source_data['scores_cal'], a)
                    np.testing.assert_allclose(current.upper, bounds[f'Envelope_{a}'],rtol=1e-8,atol=1e-9)
    audit.append(dict(dataset=folder.name,splits=200,alpha_levels=5,source_hashes_checked=200,formula_replays=5))

for name in ['air','crime']:
    folder = OUT/'extra_real'/name
    files = sorted(folder.glob('trial_*.json'))
    assert len(files) == 200
    for p in files:
        records = json.loads(p.read_text())
        assert {r['method'] for r in records} == {'Envelope','TSCP_R','TSCP_GWC','Unscaled','Point_CHR','Empirical_copula'}
        with np.load(p.with_suffix('.npz')) as data:
            for r in records:
                u = data[r['method']]
                np.testing.assert_allclose([(data['scores_test']<=u).all(axis=1).mean(), np.prod(2*u)], [r['test_coverage'],r['outcome_volume']],rtol=1e-10)
                if r['method'] == 'Point_CHR':
                    # The historical helper reused the first-half rank when
                    # n_cal was odd. Keep the source audit, then use the fixed
                    # second-half rank for the current comparison.
                    u = data_spliting_CHR_prediction(data['scores_cal'], .1).upper
                    r = {**r, 'test_coverage': float((data['scores_test']<=u).all(axis=1).mean()),
                         'outcome_volume': float(np.prod(2*u)),
                         'provenance': 'Point CHR second-half rank correction; same saved fitted residuals'}
                rows.append(r)
            if p == files[0]:
                np.testing.assert_allclose(envelope_prediction(data['scores_cal'],.1).upper,data['Envelope'],rtol=1e-8,atol=1e-9)
    audit.append(dict(dataset=name,splits=200,alpha_levels=1,formula_replays=1,provenance='new resplits/refits; original splits unavailable'))

# Original real baselines share these exact cached fits. Their residual_volume
# is a product of half-widths, so reconstruct full volumes from full_length.
base = pd.read_csv(ROOT/'data/reviewer_update/real_diagnostics/data/real_joint_trials.csv', float_precision='round_trip')
coords = pd.read_csv(ROOT/'data/reviewer_update/real_diagnostics/data/real_coordinate_trials.csv', float_precision='round_trip')
base = base[base.method != 'TSCP_R'].copy()
for (name, trial), group in base.groupby(['dataset','trial']):
    with np.load(ROOT/f'data/reviewer_update/real_diagnostics/cache/{name}/split_{trial:03d}.npz') as data:
        for r in group.to_dict('records'):
            c = coords[(coords.dataset==name)&(coords.trial==trial)&(coords.method==r['method'])].sort_values('coordinate')
            assert len(c)==r['n_dim']
            lengths=c.full_length.to_numpy()
            scores = data['scores_test']
            upper = lengths/2
            tolerance = np.where(np.isfinite(upper), 1e-12 * np.maximum(1,abs(upper)), 0.)
            low_cov = (scores < upper - tolerance).all(axis=1).mean()
            high_cov = (scores <= upper + tolerance).all(axis=1).mean()
            assert low_cov-1e-14 <= r['joint_coverage'] <= high_cov+1e-14, (name,trial,r['method'],low_cov,high_cov,r['joint_coverage'])
            volume = np.prod(lengths)
            np.testing.assert_allclose(volume,r['residual_volume']*2**r['n_dim'])
            rows.append(dict(dataset=name,trial=trial,alpha=.1,method=r['method'],test_coverage=r['joint_coverage'],outcome_volume=volume))

frame=pd.DataFrame(rows)
frame['method']=frame.method.replace({'Signed_GWC':'TSCP_GWC'})
assert not frame.duplicated(['dataset','trial','alpha','method']).any()
summary=[]
for (name,a,method),g in frame.groupby(['dataset','alpha','method']):
    assert set(g.trial)==set(range(200))
    summary.append(dict(dataset=name,alpha=a,method=method,trials=len(g),coverage=g.test_coverage.mean(),coverage_sd=g.test_coverage.std(),volume=g.outcome_volume.mean(),volume_sd=g.outcome_volume.std() if np.isfinite(g.outcome_volume).all() else np.nan,infinite_trials=int(np.isinf(g.outcome_volume).sum())))
s=pd.DataFrame(summary)
s.to_csv(OUT/'real_comparison_audited.csv',index=False)
labels={'Envelope':'TSCP (envelope)','TSCP_R':'Old TSCP shortcut','TSCP_GWC':'GWC','Unscaled':'Unscaled Max','Empirical_copula':'Empirical copula','Point_CHR':'Point CHR'}
lines=['# Real-data comparison using envelope TSCP','', 'Target joint coverage: 90%; 200 splits per dataset. Entries are mean (SD across splits). Volume is full outcome-space volume, product of full interval lengths. Baselines for the six cached datasets reuse the same saved fits; air/crime are reconstructed resplits/refits.','', '| Dataset | Method | Coverage | Volume | Infinite trials |','|---|---|---:|---:|---:|']
for name in ['stock','rf2','scm1d','scm20d','energy','student','air','crime']:
    for method,label in labels.items():
        r=s[(s.dataset==name)&(s.alpha==.1)&(s.method==method)].iloc[0]
        volume='âˆž' if r.infinite_trials else f'{r.volume:.3e} ({r.volume_sd:.3e})'
        lines.append(f'| {name} | {label} | {r.coverage:.4f} ({r.coverage_sd:.4f}) | {volume} | {r.infinite_trials}/200 |')
lines += ['', 'Audit: 1,600 saved real splits verified; six datasets have all five envelope alpha levels (0.1, 0.3, 0.5, 0.7, 0.9), air/crime have alpha=0.1. All 1,200 cached-source hashes match. Saved coverage and volume recomputed from binary bounds/test scores for every envelope-cohort record; 32 envelope formula spot-replays match current code. Baseline coverage checked against paired test scores allowing 1e-12 boundary tolerance for decimal CSV lengths. No full-LWC runs launched.', '', 'The historical manuscript table uses residual-space half-width volumes; these full outcome volumes differ by 2^d. Do not mix the two conventions. The old manuscript TSCP row is not the new envelope row.', '', 'Scope: the separate 1,430-trial ten-output CQR extension remains unrun by design; full LWC is deferred. Existing synthetic completion evidence is results/final_audit.json (225 configurations, 35,463 archives); that full synthetic audit was not rerun by this table builder.']
(REPORT_OUT / 'REAL_COMPARISON.md').write_text('\n'.join(lines),encoding='utf-8')
with (REPORT_OUT / 'REAL_COMPARISON.md').open('a', encoding='utf-8') as report:
    report.write('\n\nPoint CHR correction (2026-09-09): Air and Crime now use the second calibration half\'s own conformal rank. Their earlier rows were too narrow. Original rf2 is unchanged; the one-row-deleted rf2 study already used the correct ranks. See rf2_remaining/chr_rank_fix_verification.json and chr_rank_correction_extra_real.csv.\n')
(OUT/'real_table_audit.json').write_text(json.dumps(dict(status='passed',datasets=audit,baseline_records_checked=len(base),saved_method_records_checked=len(frame)-len(base)),indent=2))
print('\n'.join(lines))
