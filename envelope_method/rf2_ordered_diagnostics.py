"""Trace ordered-fold coverage failures to held-out coordinate errors."""
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/envelope_method/results/rf2_remaining/blocked_pilot'
with (ROOT/'data/reviewer_update/real_diagnostics/cache/rf2.pkl').open('rb') as f:X,y,_,_=pickle.load(f)
rows=[]
for k in range(5):
    p=OUT/f'fold_{k}_ordered.npz'
    with np.load(p) as z:
        payload=json.loads(p.with_suffix('.json').read_text())
        for r in payload:
            np.testing.assert_allclose([(z['scores_test']<=z[r['method']]).all(1).mean(),np.prod(2*z[r['method']])],[r['coverage'],r['volume']],rtol=1e-12)
        for j,t in enumerate(y.columns):
            rows.append(dict(fold=k,target=t,env_marginal_coverage=float((z['scores_test'][:,j]<=z['Envelope'][j]).mean()),chr_marginal_coverage=float((z['scores_test'][:,j]<=z['Point_CHR'][j]).mean()),cal_q90=float(np.quantile(z['scores_cal'][:,j],.9)),test_q90=float(np.quantile(z['scores_test'][:,j],.9)),env_half_width=z['Envelope'][j],train_y_min=y.iloc[z['train_indices'],j].min(),train_y_max=y.iloc[z['train_indices'],j].max(),test_y_min=y.iloc[z['test_indices'],j].min(),test_y_max=y.iloc[z['test_indices'],j].max(),test_target_outside_training_range=float(((y.iloc[z['test_indices'],j]<y.iloc[z['train_indices'],j].min())|(y.iloc[z['test_indices'],j]>y.iloc[z['train_indices'],j].max())).mean())))
pd.DataFrame(rows).to_csv(OUT/'coordinate_failures.csv',index=False)
print(pd.DataFrame(rows).sort_values(['fold','env_marginal_coverage']).groupby('fold').head(2).to_string(index=False))
