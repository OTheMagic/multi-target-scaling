"""Remaining shape, tail, chronology and implementation diagnostics."""
import os,sys,json,pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utility.conformal_utils import conformal_quantile
from utility.envelope import envelope_prediction
from utility.data_splitting import data_spliting_CHR_prediction
OUT=ROOT/'data/envelope_method/results/rf2_remaining'
OUT.mkdir(exist_ok=True)
CLEAN=ROOT/'data/envelope_method/results/rf2_remove_one'
with (ROOT/'data/reviewer_update/real_diagnostics/cache/rf2.pkl').open('rb') as f:
    X,y,model,meta=pickle.load(f)
rows=[]; tails=[]; bugs=[]; checks=[]; overlap=[]
for trial in range(200):
    with np.load(CLEAN/f'trial_{trial:03d}.npz') as z:
        cal,test=z['scores_cal'],z['scores_test'];ci=z['cal_indices'];ti=z['test_indices'];tr=z['train_indices']
        e=z['Envelope'];ch=z['Point_CHR']
    def record(method,u,seed=42):
        rows.append(dict(trial=trial,seed=seed,method=method,coverage=float((test<=u).all(1).mean()),volume=float(np.prod(2*u))))
    record('Envelope',e)
    for seed in [42,7,19,83,151]:
        a,b=train_test_split(cal,test_size=.5,random_state=seed)
        for method,center,scale in [
            ('Mean_SD',a.mean(0),a.std(0)),
            ('Quantile90',np.zeros(8),conformal_quantile(a,.1,axis=0)),
            ('Winsor99_mean_SD',np.minimum(a,np.quantile(a,.99,axis=0)).mean(0),np.minimum(a,np.quantile(a,.99,axis=0)).std(0)),
            ('Winsor95_mean_SD',np.minimum(a,np.quantile(a,.95,axis=0)).mean(0),np.minimum(a,np.quantile(a,.95,axis=0)).std(0))]:
            u=center+scale*conformal_quantile(((b-center)/scale).max(1),.1)
            record(method,u,seed)
            if method=='Quantile90' and seed==42:np.testing.assert_allclose(u,ch,rtol=1e-12)
    if len(cal)%2:
        wrong=data_spliting_CHR_prediction(cal,.1).upper
        bugs.append(dict(trial=trial,n_cal=len(cal),old_coverage=float((test<=wrong).all(1).mean()),correct_coverage=float((test<=ch).all(1).mean()),old_volume=float(np.prod(2*wrong)),correct_volume=float(np.prod(2*ch))))
    if trial<10:
        np.testing.assert_allclose(envelope_prediction(cal,.1,search='exhaustive').upper,e,rtol=1e-10,atol=1e-12)
        factors=np.geomspace(.01,100,8)
        np.testing.assert_allclose(envelope_prediction(cal*factors,.1).upper/factors,e,rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(envelope_prediction(cal[:,::-1],.1).upper[::-1],e,rtol=1e-9,atol=1e-10)
        checks.append(trial)
    for j in [0,3,6]:
        idx=np.argsort(cal[:,j])[-4:]
        for k in idx:
            pos=int(ci[k]);orig=int(y.index[pos])
            tails.append(dict(trial=trial,target=y.columns[j],position=pos,original_index=orig,residual=cal[k,j],observed_target=y.iloc[pos,j],current_feature=X.iloc[pos,j],residual_over_q90=cal[k,j]/np.quantile(cal[:,j],.9)))
    # Neighbor distance in original ARFF row index (not complete-case position).
    train_idx=np.sort(y.index.to_numpy()[tr]);test_idx=y.index.to_numpy()[ti]
    ix=np.searchsorted(train_idx,test_idx)
    near=np.minimum(abs(test_idx-train_idx[np.minimum(ix,len(train_idx)-1)]),abs(test_idx-train_idx[np.maximum(ix-1,0)]))
    overlap.append(dict(trial=trial,test_neighbor_within_1=float((near<=1).mean()),test_neighbor_within_48=float((near<=48).mean())))

f=pd.DataFrame(rows);f.to_csv(OUT/'shape_trials.csv',index=False)
summary=f.groupby(['method','seed']).agg(coverage=('coverage','mean'),volume=('volume','mean'),trials=('trial','size')).reset_index()
summary.to_csv(OUT/'shape_summary.csv',index=False)
t=pd.DataFrame(tails);t.to_csv(OUT/'tail_rows.csv',index=False)
frequent=t.groupby(['target','position','original_index']).agg(appearances=('trial','size'),mean_residual=('residual','mean'),mean_normalized_residual=('residual_over_q90','mean'),observed_target=('observed_target','first'),current_feature=('current_feature','first')).reset_index().sort_values(['target','appearances'],ascending=[True,False])
frequent.to_csv(OUT/'tail_row_summary.csv',index=False)
target_details=[]
for j in [0,3,6]:
    target=y.columns[j];a=y.iloc[:,j];top=frequent[frequent.target==target].head(5)
    target_details.append(dict(target=target,raw_quantiles=a.quantile([.5,.9,.99,.999,1]).to_dict(),most_frequent_top_residual_rows=top.to_dict('records'),distinct_top_residual_rows=int((frequent.target==target).sum()),largest_target_neighbor_values={str(int(i)):float(y.loc[i,target]) for i in y.index[abs(y.index-int(a.idxmax()))<=3]}))
# Test exact lag identities to distinguish stored chronology from guessed order.
lag=[]
for j in range(8):
    current=X.iloc[:,j]
    common=y.index.intersection(current.index-48)
    original_target=y.loc[common,y.columns[j]].to_numpy()
    future_feature=current.loc[common+48].to_numpy()
    lag.append(dict(target=y.columns[j],pairs=len(common),target_matches_feature_48_rows_later=float(np.isclose(original_target,future_feature,rtol=0,atol=1e-8).mean()),lag1_correlation=float(current.corr(current.shift(1)))))
scale=y.drop(index=4782).std()
result=dict(shape_summary=summary.to_dict('records'),tails=target_details,code_checks=dict(exhaustive_envelope_matches=checks,coordinate_rescaling_matches=checks,coordinate_permutation_matches=checks),odd_calibration_chr_bug=bugs,neighbor_overlap=pd.DataFrame(overlap).mean().to_dict(),lag_identity=lag,target_variance_shares=(scale**2/(scale**2).sum()).to_dict(),scope='Exploratory checks on the one-row-deleted cohort; robust shape statistics fitted only on the first calibration half, second half untouched; no extra rows deleted. Exhaustive means exhaustive envelope cells, not full LWC.')
(OUT/'diagnostics.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');print(json.dumps(result,indent=2))
