"""Regression against independent saved bounds, plus real-table impact audit."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utility.data_splitting import data_spliting_CHR_prediction
OUT=ROOT/'data/envelope_method/results/rf2_remaining'
base=pd.read_csv(ROOT/'data/reviewer_update/real_diagnostics/data/real_joint_trials.csv')
old_count=0;new_count=0;extra=[]
for trial in range(200):
    with np.load(ROOT/f'data/reviewer_update/real_diagnostics/cache/rf2/split_{trial:03d}.npz') as z:
        u=data_spliting_CHR_prediction(z['scores_cal'],.1).upper
        r=base[(base.dataset=='rf2')&(base.trial==trial)&(base.method=='Point_CHR')].iloc[0]
        np.testing.assert_allclose(np.prod(u),r.residual_volume,rtol=1e-12)
        np.testing.assert_allclose((z['scores_test']<=u).all(1).mean(),r.joint_coverage)
        old_count+=1
    with np.load(ROOT/f'data/envelope_method/results/rf2_remove_one/trial_{trial:03d}.npz') as z:
        u=data_spliting_CHR_prediction(z['scores_cal'],.1).upper
        np.testing.assert_allclose(u,z['Point_CHR'],rtol=1e-12)
        new_count+=1
for dataset in ['air','crime']:
    for trial in range(200):
        folder=ROOT/f'data/envelope_method/results/extra_real/{dataset}'
        original=json.loads((folder/f'trial_{trial:03d}.json').read_text())
        old=next(r for r in original if r['method']=='Point_CHR')
        with np.load(folder/f'trial_{trial:03d}.npz') as z:
            u=data_spliting_CHR_prediction(z['scores_cal'],.1).upper
            extra.append(dict(dataset=dataset,trial=trial,n_cal=len(z['scores_cal']),old_coverage=old['test_coverage'],corrected_coverage=float((z['scores_test']<=u).all(1).mean()),old_volume=old['outcome_volume'],corrected_volume=float(np.prod(2*u))))
e=pd.DataFrame(extra);e.to_csv(OUT/'chr_rank_correction_extra_real.csv',index=False)
summary=e.groupby('dataset').agg(n_cal=('n_cal','first'),old_coverage=('old_coverage','mean'),corrected_coverage=('corrected_coverage','mean'),old_volume=('old_volume','mean'),corrected_volume=('corrected_volume','mean')).reset_index()
report=dict(status='passed',original_even_rf2_unchanged=old_count,independent_reduced_rf2_bounds_matched=new_count,other_real_impact=summary.to_dict('records'),scope='Only second-half conformal rank corrected. Existing saved historical baseline tables were not overwritten; corrected sidecars are separately saved. Original rf2 and all one-row-deleted rf2 results remain unchanged.')
(OUT/'chr_rank_fix_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
