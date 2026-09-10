"""Matched leave-one-row-out sensitivity; immutable original cohort."""
import os,sys,json,pickle,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from utility.envelope import envelope_prediction
from utility.conformal_utils import conformal_quantile
CACHE=ROOT/'data/reviewer_update/real_diagnostics/cache'
OUT=ROOT/'data/envelope_method/results/rf2_remove_one'
REMOVED=3660

def load():
    with (CACHE/'rf2.pkl').open('rb') as f:
        return pickle.load(f)

def prepare():
    OUT.mkdir(exist_ok=True)
    X,y,_,meta=load()
    assert y.index[REMOVED]==4782 and y.iloc[REMOVED,1]==78.9
    a=y.iloc[:,1].to_numpy(); med=np.median(a);mad=np.median(abs(a-med));q1,q3=np.quantile(a,[.25,.75]);iqr=q3-q1
    features=[]
    for j in range(X.shape[1]):
        indices=np.flatnonzero(X.iloc[:,j].to_numpy()==78.9)
        if len(indices):features.append(dict(feature=X.columns[j],positions=indices.tolist(),original_indices=X.index[indices].tolist()))
    info=dict(removed_position=REMOVED,removed_original_index=int(y.index[REMOVED]),target=y.columns[1],value=float(a[REMOVED]),rows_before=len(y),rows_after=len(y)-1,median=med,mad=mad,modified_z=float(.67448975*(a[REMOVED]-med)/mad),q1=q1,q3=q3,upper_outer_fence=float(q3+3*iqr),count_above_outer_fence=int((a>q3+3*iqr).sum()),largest_targets=[dict(position=int(i),original_index=int(y.index[i]),value=float(a[i])) for i in np.argsort(a)[-10:][::-1]],feature_occurrences_78_9=features,protocol='200 original split memberships with one specified row removed wherever present; refit the same random forest on each reduced training partition; evaluate both methods on identical reduced calibration/test sets. Original files untouched. Point CHR uses each calibration half\'s own conformal rank.',source_pickle_sha256=hashlib.sha256((CACHE/'rf2.pkl').read_bytes()).hexdigest())
    (OUT/'metadata.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2),flush=True)

def one(trial):
    dest=OUT/f'trial_{trial:03d}.json'
    if dest.exists():return trial,0
    X,y,template,meta=load()
    src=CACHE/f'rf2/split_{trial:03d}.npz'
    with np.load(src) as z:
        original={k:z[k+'_indices'].copy() for k in ['train','cal','test']}
        old_test=z['scores_test'].copy()
    parts={k:v[v!=REMOVED] for k,v in original.items()}
    assert len(set(np.concatenate(list(parts.values()))))==len(y)-1
    assert all(REMOVED not in v for v in parts.values())
    model=clone(template);model.set_params(n_jobs=1)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X.iloc[parts['train']],y.iloc[parts['train']])
        cal=abs(model.predict(X.iloc[parts['cal']])-y.iloc[parts['cal']].to_numpy())
        test=abs(model.predict(X.iloc[parts['test']])-y.iloc[parts['test']].to_numpy())
        env=envelope_prediction(cal,.1)
        a,b=train_test_split(cal,test_size=.5,random_state=42)
        shape=conformal_quantile(a,.1,axis=0)
        chr_upper=shape*conformal_quantile((b/shape).max(1),.1)
    seconds=time.perf_counter()-start
    bounds={'Envelope':env.upper,'Point_CHR':chr_upper}
    rows=[]
    for method,u in bounds.items():
        rows.append(dict(trial=trial,method=method,coverage=float((test<=u).all(1).mean()),volume=float(np.prod(2*u)),coordinate_lengths=(2*u).tolist(),coordinate_coverage=(test<=u).mean(0).tolist(),n_train=len(parts['train']),n_cal=len(cal),n_test=len(test),removed_partition=next(k for k,v in original.items() if REMOVED in v),fit_seconds=seconds,nasi2_cal_std=float(cal[:,1].std()),nasi2_cal_q90=float(np.quantile(cal[:,1],.9)),nasi2_cal_max=float(cal[:,1].max())))
    np.savez_compressed(OUT/f'trial_{trial:03d}.npz',scores_cal=cal,scores_test=test,**{k+'_indices':v for k,v in parts.items()},**bounds)
    dest.write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),records=rows),indent=2))
    return trial,seconds

def summarize():
    paths=sorted(OUT.glob('trial_*.json'))
    assert len(paths)==200
    rows=[r for p in paths for r in json.loads(p.read_text())['records']]
    f=pd.DataFrame(rows); f.to_csv(OUT/'trials.csv',index=False)
    summaries=[]
    for method,g in f.groupby('method'):
        assert set(g.trial)==set(range(200))
        se=g.coverage.std()/np.sqrt(200)
        summaries.append(dict(method=method,trials=200,coverage=g.coverage.mean(),coverage_sd=g.coverage.std(),coverage_mc95=[g.coverage.mean()-1.96*se,g.coverage.mean()+1.96*se],mean_volume=g.volume.mean(),median_volume=g.volume.median()))
    pv=f.pivot(index='trial',columns='method',values='volume')
    pc=f.pivot(index='trial',columns='method',values='coverage')
    original=pd.read_csv(ROOT/'data/envelope_method/results/rf2_diagnosis/paired_trials.csv')
    before=original[original.method.isin(['Envelope','Point_CHR'])]
    merged=f.merge(before,on=['trial','method'],suffixes=('_after','_before'))
    changes={}
    for method,g in merged.groupby('method'):
        delta=g.coverage_after-g.coverage_before
        changes[method]=dict(coverage_change=delta.mean(),coverage_change_mc95=[delta.mean()-1.96*delta.std()/np.sqrt(200),delta.mean()+1.96*delta.std()/np.sqrt(200)],volume_after_over_before_means=g.volume_after.mean()/g.volume_before.mean())
    # Independent saved-bound and split-integrity audit, without model reruns.
    for p in paths:
        payload=json.loads(p.read_text()); t=payload['records'][0]['trial']
        assert payload['source_sha256']==hashlib.sha256((CACHE/f'rf2/split_{t:03d}.npz').read_bytes()).hexdigest()
        with np.load(p.with_suffix('.npz')) as z:
            assert all(REMOVED not in z[k+'_indices'] for k in ['train','cal','test'])
            for r in payload['records']:
                np.testing.assert_allclose([np.prod(2*z[r['method']]),(z['scores_test']<=z[r['method']]).all(1).mean()],[r['volume'],r['coverage']],rtol=1e-12)
    result=dict(status='complete',summaries=summaries,ratio_of_mean_volumes=pv.Envelope.mean()/pv.Point_CHR.mean(),median_paired_volume_ratio=(pv.Envelope/pv.Point_CHR).median(),envelope_smaller_fraction=float((pv.Envelope<pv.Point_CHR).mean()),paired_coverage_difference=float((pc.Envelope-pc.Point_CHR).mean()),before_after=changes,by_removed_partition=f.groupby(['removed_partition','method'])[['coverage','volume','nasi2_cal_std','nasi2_cal_max']].mean().reset_index().to_dict('records'),verification='All 200 original-source hashes and reduced split exclusions verified; all 400 saved method metrics independently recomputed from binary bounds/test residuals.',qualification='Post-hoc removal sensitivity conditional on a modified dataset; not evidence that the original observation was erroneous or that nominal coverage on the original population is preserved.')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    prepare()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,i) for i in range(200)]
        for n,f in enumerate(as_completed(futures),1):
            t,s=f.result()
            print(f'Completed {n}/200: trial {t}, {s:.1f}s',flush=True)
    summarize()
