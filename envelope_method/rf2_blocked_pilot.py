"""Matched-size ordered-vs-shuffled pilots within a single ARFF sequence."""
import os,sys,json,pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from utility.envelope import envelope_prediction
from utility.conformal_utils import conformal_quantile
OUT=ROOT/'data/envelope_method/results/rf2_remaining/blocked_pilot'

def one(fold,ordered):
    path=OUT/f'fold_{fold}_{"ordered" if ordered else "shuffled"}.json'
    if path.exists():return json.loads(path.read_text())
    with (ROOT/'data/reviewer_update/real_diagnostics/cache/rf2.pkl').open('rb') as f:X,y,model,meta=pickle.load(f)
    orig=y.index.to_numpy();ids=np.flatnonzero((orig>=4108)&(orig!=4782))
    tr=ids[:[1000,1400,1800,2200,2600][fold]]
    ca=ids[orig[ids]>orig[tr[-1]]+48][:384]
    te=ids[orig[ids]>orig[ca[-1]]+48][:1024]
    assert len(te)==1024
    assert orig[tr[-1]]+48<orig[ca[0]] and orig[ca[-1]]+48<orig[te[0]]
    if not ordered:
        pool=np.concatenate([tr,ca,te]);r=np.random.default_rng(20260909+fold).permutation(pool)
        nt=len(tr);nc=len(ca);tr,ca,te=r[:nt],r[nt:nt+nc],r[nt+nc:]
    model=clone(model);model.set_params(n_jobs=1)
    with threadpool_limits(limits=1):
        model.fit(X.iloc[tr],y.iloc[tr])
        cal=abs(model.predict(X.iloc[ca])-y.iloc[ca].to_numpy());test=abs(model.predict(X.iloc[te])-y.iloc[te].to_numpy())
        env=envelope_prediction(cal,.1).upper
        a,b=train_test_split(cal,test_size=.5,random_state=42);shape=conformal_quantile(a,.1,axis=0)
        chr_u=shape*conformal_quantile((b/shape).max(1),.1)
    bounds=dict(Envelope=env,Point_CHR=chr_u)
    np.savez_compressed(path.with_suffix('.npz'),train_indices=tr,cal_indices=ca,test_indices=te,scores_cal=cal,scores_test=test,**bounds)
    rows=[dict(fold=fold,ordered=ordered,method=m,coverage=float((test<=u).all(1).mean()),volume=float(np.prod(2*u)),n_train=len(tr),n_cal=len(ca),n_test=len(te)) for m,u in bounds.items()]
    path.write_text(json.dumps(rows,indent=2));return rows

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(one,k,b) for k in range(5) for b in [False,True]]
        for f in as_completed(fs):rows.extend(f.result())
    pd.DataFrame(rows).sort_values(['fold','ordered','method']).to_csv(OUT/'trials.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['ordered','method']).agg(coverage=('coverage','mean'),volume=('volume','mean'),min_coverage=('coverage','min'),max_coverage=('coverage','max'))
    summary.to_csv(OUT/'summary.csv')
    checks=0
    for fold in range(5):
        with np.load(OUT/f'fold_{fold}_ordered.npz') as a,np.load(OUT/f'fold_{fold}_shuffled.npz') as b:
            ka=['train_indices','cal_indices','test_indices']
            np.testing.assert_array_equal(np.sort(np.concatenate([a[k] for k in ka])),np.sort(np.concatenate([b[k] for k in ka])))
            for z in [a,b]:
                for m in ['Envelope','Point_CHR']:
                    assert np.isfinite(z[m]).all();checks+=1
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',matched_fold_pools=5,finite_regions=checks,sequence='original ARFF index >=4108 only, avoiding detected concatenation boundary',embargo='48 original row indices between training and calibration and between calibration and test',scope='Five overlapping folds, with identical per-fold row pools and train/cal/test sizes for ordered and shuffled conditions. No independent-replication confidence intervals. Row order inferred from exact 48-row label/input identities; absolute timestamps unavailable.'),indent=2))
    print(summary.to_string())
