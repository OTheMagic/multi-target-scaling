"""Observable-regime coverage and independently trained adaptive score scales."""
import os,sys,json,pickle,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from utility.envelope import envelope_prediction
from utility.conformal_utils import conformal_quantile
OUT=ROOT/'data/envelope_method/results/rf2_remaining/conditional'
SCALED=ROOT/'data/envelope_method/results/rf2_remaining/model_time'

def init():
    global X,y,meta
    with (ROOT/'data/reviewer_update/real_diagnostics/cache/rf2.pkl').open('rb') as f:X,y,_,meta=pickle.load(f)

def one(trial):
    path=OUT/f'trial_{trial:03d}.json'
    if path.exists():return json.loads(path.read_text())
    with np.load(SCALED/f'random_{trial:03d}_scaled.npz') as z:
        cal,test=z['scores_cal'],z['scores_test'];tr,ci,ti=z['train_indices'],z['cal_indices'],z['test_indices']
        old_env,old_chr=z['Envelope'],z['Point_CHR']
    a,b=train_test_split(np.arange(len(cal)),test_size=.5,random_state=42)
    # Only first-half residuals fit the scale model; b and test labels are unused.
    floor=np.maximum(np.quantile(cal[a],.1,axis=0),1e-10)
    h=ExtraTreesRegressor(n_estimators=100,max_depth=4,min_samples_leaf=20,random_state=42,n_jobs=1)
    with threadpool_limits(limits=1):
        h.fit(X.iloc[ci[a],:64],np.log(np.maximum(cal[a],floor)))
        hb=np.exp(h.predict(X.iloc[ci[b],:64]));ht=np.exp(h.predict(X.iloc[ti,:64]))
        adapt=envelope_prediction(cal[b]/hb,.1)
        split=envelope_prediction(cal[b],.1)
    bounds=dict(Envelope_full=old_env,Point_CHR=old_chr,Envelope_half=split.upper,Envelope_adaptive=ht*adapt.upper)
    # Observable change over 48 rows, standardized from the location training set.
    current=X.iloc[:,:8].to_numpy();past=X.iloc[:,48:56].to_numpy()
    assert all('__-48' in str(c) for c in X.columns[48:56])
    delta=abs(current-past);scale=np.maximum(delta[tr].std(0),1e-10)
    train_risk=(delta[tr]/scale).max(1);threshold=np.quantile(train_risk,.8)
    high=(delta[ti]/scale).max(1)>threshold
    rows=[]
    for method,u in bounds.items():
        covered=(test<=u).all(1);v=np.prod(2*u,axis=-1)
        for group,mask in [('all',np.ones(len(test),dtype=bool)),('high_change',high),('ordinary_change',~high)]:
            assert mask.any()
            rows.append(dict(trial=trial,method=method,group=group,n=int(mask.sum()),covered=int(covered[mask].sum()),coverage=float(covered[mask].mean()),volume=float(np.mean(v[mask])) if np.ndim(v) else float(v)))
    np.savez_compressed(path.with_suffix('.npz'),adaptive_shape_cal=hb,adaptive_shape_test=ht,adaptive_upper=adapt.upper,half_upper=split.upper,high_change=high,shape_indices=ci[a],calibration_indices=ci[b],test_indices=ti,train_risk_threshold=threshold)
    path.write_text(json.dumps(rows,indent=2));return rows

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
        fs=[pool.submit(one,i) for i in range(200)]
        for n,f in enumerate(as_completed(fs),1):
            rows.extend(f.result())
            if n%50==0:print(f'{n}/200 adaptive-scale fits complete',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'trials.csv',index=False)
    summary=frame.groupby(['method','group']).agg(trials=('trial','size'),coverage=('coverage','mean'),volume=('volume','mean'),test_count=('n','sum'),covered_count=('covered','sum')).reset_index()
    summary['pooled_coverage']=summary.covered_count/summary.test_count
    summary.to_csv(OUT/'summary.csv',index=False)
    print(summary.to_string(index=False))
