"""Paired target-normalization refits plus separate chronological pilots."""
import os,sys,json,pickle,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from utility.envelope import envelope_prediction
from utility.conformal_utils import conformal_quantile
OUT=ROOT/'data/envelope_method/results/rf2_remaining/model_time'
CLEAN=ROOT/'data/envelope_method/results/rf2_remove_one'

def init():
    global X,y,template,meta
    with (ROOT/'data/reviewer_update/real_diagnostics/cache/rf2.pkl').open('rb') as f:X,y,template,meta=pickle.load(f)

def one(task):
    kind,trial,scaled=task
    name=f'{kind}_{trial:03d}_'+('scaled' if scaled else 'raw')
    dest=OUT/(name+'.json')
    if dest.exists():return name,0
    if kind=='random':
        with np.load(CLEAN/f'trial_{trial:03d}.npz') as z:parts={k:z[k+'_indices'] for k in ['train','cal','test']}
    else:
        # Five overlapping expanding-history folds; not independent replications.
        ids=np.array([i for i in range(len(y)) if i!=3660])
        cutoff=[3000,3600,4200,4800,5400][trial]
        tr=ids[:cutoff];orig=y.index.to_numpy()
        eligible=ids[orig[ids]>orig[tr[-1]]+48]
        cal=eligible[:384]
        te=ids[orig[ids]>orig[cal[-1]]+48][:1024]
        assert len(te)==1024
        parts=dict(train=tr,cal=cal,test=te)
        assert orig[tr[-1]]+48<orig[cal[0]] and orig[cal[-1]]+48<orig[te[0]]
    model=clone(template);model.set_params(n_jobs=1)
    train_y=y.iloc[parts['train']].to_numpy()
    center=train_y.mean(0) if scaled else np.zeros(8)
    scale=train_y.std(0) if scaled else np.ones(8)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X.iloc[parts['train']],(train_y-center)/scale)
        cal=abs(model.predict(X.iloc[parts['cal']])*scale+center-y.iloc[parts['cal']].to_numpy())
        test=abs(model.predict(X.iloc[parts['test']])*scale+center-y.iloc[parts['test']].to_numpy())
        env=envelope_prediction(cal,.1)
        a,b=train_test_split(cal,test_size=.5,random_state=42)
        shape=conformal_quantile(a,.1,axis=0)
        chr_u=shape*conformal_quantile((b/shape).max(1),.1)
    seconds=time.perf_counter()-start
    rows=[]
    for method,u in [('Envelope',env.upper),('Point_CHR',chr_u)]:
        rows.append(dict(kind=kind,trial=trial,scaled=scaled,method=method,coverage=float((test<=u).all(1).mean()),volume=float(np.prod(2*u)),coordinate_mae=test.mean(0).tolist(),coordinate_rmse=np.sqrt((test**2).mean(0)).tolist(),coordinate_lengths=(2*u).tolist(),n_train=len(parts['train']),n_cal=len(cal),n_test=len(test),fit_seconds=seconds))
    np.savez_compressed(OUT/(name+'.npz'),scores_cal=cal,scores_test=test,train_center=center,train_scale=scale,Envelope=env.upper,Point_CHR=chr_u,**{k+'_indices':v for k,v in parts.items()})
    dest.write_text(json.dumps(rows,indent=2))
    return name,seconds

def finish():
    files=sorted(OUT.glob('*.json'))
    rows=[r for p in files if p.stem.startswith(('random_','chronological_')) for r in json.loads(p.read_text())]
    f=pd.DataFrame(rows)
    assert len(f)==420
    baseline=pd.read_csv(CLEAN/'trials.csv')
    baseline['kind']='random';baseline['scaled']=False
    # Recover MAE/RMSE for the existing unscaled-target fits.
    extra=[]
    for trial in range(200):
        with np.load(CLEAN/f'trial_{trial:03d}.npz') as z:
            extra.append(dict(trial=trial,coordinate_mae=z['scores_test'].mean(0).tolist(),coordinate_rmse=np.sqrt((z['scores_test']**2).mean(0)).tolist()))
    baseline=baseline.merge(pd.DataFrame(extra),on='trial')
    allf=pd.concat([f,baseline],ignore_index=True)
    allf.to_csv(OUT/'trials.csv',index=False)
    summary=allf.groupby(['kind','scaled','method']).agg(trials=('trial','size'),coverage=('coverage','mean'),volume=('volume','mean'),median_volume=('volume','median')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    checks=0
    for p in files:
        if not p.stem.startswith(('random_','chronological_')):continue
        records=json.loads(p.read_text())
        with np.load(p.with_suffix('.npz')) as z:
            assert all(3660 not in z[k+'_indices'] for k in ['train','cal','test'])
            for r in records:
                np.testing.assert_allclose([np.prod(2*z[r['method']]),(z['scores_test']<=z[r['method']]).all(1).mean()],[r['volume'],r['coverage']],rtol=1e-12)
                checks+=1
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',fresh_fits=210,checked_method_records=checks,random_trials=200,chronological_folds=5,qualification='Chronological folds overlap and have different training sizes; they are a deployment stress test, not an equal-size isolated causal comparison.'),indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(parents=True,exist_ok=True)
    tasks=[('random',i,True) for i in range(200)]+[('chronological',i,s) for i in range(5) for s in [False,True]]
    with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
        fs=[pool.submit(one,t) for t in tasks]
        for n,f in enumerate(as_completed(fs),1):
            name,sec=f.result()
            if n%20==0 or n==len(tasks):print(f'{n}/{len(tasks)} completed; {name} {sec:.1f}s',flush=True)
    finish()
