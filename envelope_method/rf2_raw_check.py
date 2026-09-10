import sys,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
cache=ROOT/'data/reviewer_update/real_diagnostics/cache'
with (cache/'rf2.pkl').open('rb') as f:
    X,y,model,meta=pickle.load(f)
rows=[]; placement=[]
for trial in range(200):
    with np.load(cache/f'rf2/split_{trial:03d}.npz') as z:
        e=z['scores_cal'][:,1]
        placement.append(dict(trial=trial,anomaly_partition=next(k for k in ['train','cal','test'] if 3660 in z[k+'_indices']),nasi2_std=e.std(),nasi2_q90=np.quantile(e,.9)))
        for k in np.flatnonzero(e>1):
            idx=int(z['cal_indices'][k])
            rows.append(dict(trial=trial,row=idx,residual=float(e[k]),target=float(y.iloc[idx,1]),current_feature=float(X.iloc[idx,1])))
r=pd.DataFrame(rows)
v=pd.read_csv(ROOT/'data/envelope_method/results/rf2_diagnosis/paired_trials.csv').pivot(index='trial',columns='method',values='volume')
placements=pd.DataFrame(placement).join(v,on='trial')
placements['paired_ratio']=placements.Envelope/placements.Point_CHR
groups=placements.groupby('anomaly_partition').agg(trials=('trial','size'),envelope_volume=('Envelope','mean'),chr_volume=('Point_CHR','mean'),median_paired_ratio=('paired_ratio','median'),nasi2_std=('nasi2_std','mean'),nasi2_q90=('nasi2_q90','mean'))
groups['ratio_of_means']=groups.envelope_volume/groups.chr_volume
out=dict(anomaly_placement=groups.reset_index().to_dict('records'),neighbor_targets=y.iloc[3657:3664,1].to_dict(),nasi2_target_quantiles=y.iloc[:,1].quantile([0,.5,.9,.99,1]).to_dict(),nasi2_extreme_calibration_residual_count=len(r),splits_with_residual_above_1=r.trial.nunique(),distinct_rows=r.row.nunique(),largest_residuals=r.sort_values('residual',ascending=False).head(12).to_dict('records'),frequent_rows=r.groupby('row').agg(occurrences=('trial','size'),mean_residual=('residual','mean'),target=('target','first')).sort_values('occurrences',ascending=False).head(12).reset_index().to_dict('records'))
(ROOT/'data/envelope_method/results/rf2_diagnosis/raw_target_check.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
