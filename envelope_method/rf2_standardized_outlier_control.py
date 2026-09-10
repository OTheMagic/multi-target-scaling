"""Complete the original-vs-deleted rf2 comparison with identical target scaling."""
import os,sys,json,pickle,time,hashlib
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
CACHE=ROOT/'data/reviewer_update/real_diagnostics/cache'
REMOVED=ROOT/'data/envelope_method/results/rf2_remaining/model_time'
OUT=ROOT/'data/envelope_method/results/rf2_standardized_outlier_control'

def init():
    global X,y,template,meta
    with (CACHE/'rf2.pkl').open('rb') as f:X,y,template,meta=pickle.load(f)

def one(trial):
    dest=OUT/f'trial_{trial:03d}.json'
    if dest.exists():return trial,0
    src=CACHE/f'rf2/split_{trial:03d}.npz'
    with np.load(src) as z:parts={k:z[k+'_indices'].copy() for k in ['train','cal','test']}
    a=y.iloc[parts['train']].to_numpy();center=a.mean(0);scale=a.std(0)
    model=clone(template);model.set_params(n_jobs=1)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X.iloc[parts['train']],(a-center)/scale)
        cal=abs(model.predict(X.iloc[parts['cal']])*scale+center-y.iloc[parts['cal']].to_numpy())
        test=abs(model.predict(X.iloc[parts['test']])*scale+center-y.iloc[parts['test']].to_numpy())
        env=envelope_prediction(cal,.1).upper
        s1,s2=train_test_split(cal,test_size=.5,random_state=42)
        q=conformal_quantile(s1,.1,axis=0)
        chr_u=q*conformal_quantile((s2/q).max(1),.1)
    sec=time.perf_counter()-start
    rows=[];mask=parts['test']!=3660
    for method,u in [('Envelope',env),('Point_CHR',chr_u)]:
        inside=(test<=u).all(1)
        rows.append(dict(trial=trial,treatment='Retained',method=method,coverage=float(inside.mean()),common_test_coverage=float(inside[mask].mean()),volume=float(np.prod(2*u)),coordinate_lengths=(2*u).tolist(),n_cal=len(cal),n_test=len(test),outlier_partition=next(k for k,v in parts.items() if 3660 in v),fit_seconds=sec))
    archive=OUT/f'trial_{trial:03d}.npz'
    np.savez_compressed(archive,scores_cal=cal,scores_test=test,train_center=center,train_scale=scale,Envelope=env,Point_CHR=chr_u,**{k+'_indices':v for k,v in parts.items()})
    dest.write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),records=rows),indent=2))
    return trial,sec

def finish():
    init();rows=[];checks=0;manifest=[]
    for trial in range(200):
        p=OUT/f'trial_{trial:03d}.json';payload=json.loads(p.read_text())
        source=CACHE/f'rf2/split_{trial:03d}.npz'
        assert payload['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        rows.extend(payload['records'])
        with np.load(p.with_suffix('.npz')) as z,np.load(REMOVED/f'random_{trial:03d}_scaled.npz') as clean:
            for key in ['train','cal','test']:
                original=z[key+'_indices'];np.testing.assert_array_equal(original[original!=3660],clean[key+'_indices'])
            for obj in [z,clean]:
                target=y.iloc[obj['train_indices']].to_numpy()
                np.testing.assert_allclose(target.mean(0),obj['train_center'],rtol=1e-13)
                np.testing.assert_allclose(target.std(0),obj['train_scale'],rtol=1e-13)
            for r in payload['records']:
                inside=(z['scores_test']<=z[r['method']]).all(1)
                np.testing.assert_allclose([inside.mean(),inside[z['test_indices']!=3660].mean(),np.prod(2*z[r['method']])],[r['coverage'],r['common_test_coverage'],r['volume']],rtol=1e-12)
                checks+=1
            for method in ['Envelope','Point_CHR']:
                coverage=float((clean['scores_test']<=clean[method]).all(1).mean())
                rows.append(dict(trial=trial,treatment='Removed',method=method,coverage=coverage,common_test_coverage=coverage,volume=float(np.prod(2*clean[method])),coordinate_lengths=(2*clean[method]).tolist(),n_cal=len(clean['scores_cal']),n_test=len(clean['scores_test']),outlier_partition=payload['records'][0]['outlier_partition']))
            if 3660 in z['test_indices']:
                # Same training/calibration data should reproduce exactly.
                for method in ['Envelope','Point_CHR']:np.testing.assert_allclose(z[method],clean[method],rtol=1e-12)
            manifest.append(dict(trial=trial,retained_sha256=hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest(),reused_removed_sha256=hashlib.sha256((REMOVED/f'random_{trial:03d}_scaled.npz').read_bytes()).hexdigest()))
    f=pd.DataFrame(rows);f.to_csv(OUT/'trials.csv',index=False)
    summary=f.groupby(['treatment','method']).agg(trials=('trial','size'),coverage=('coverage','mean'),coverage_sd=('coverage','std'),common_test_coverage=('common_test_coverage','mean'),volume=('volume','mean'),volume_sd=('volume','std'),median_volume=('volume','median')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    effects=[]
    for method in ['Envelope','Point_CHR']:
        g=f[f.method==method];v=g.pivot(index='trial',columns='treatment',values='volume');c=g.pivot(index='trial',columns='treatment',values='common_test_coverage');delta=c.Removed-c.Retained
        effects.append(dict(method=method,removed_over_retained_mean_volume=v.Removed.mean()/v.Retained.mean(),mean_paired_volume_ratio=(v.Removed/v.Retained).mean(),median_paired_volume_ratio=(v.Removed/v.Retained).median(),common_coverage_difference=delta.mean(),coverage_difference_mc95=[delta.mean()-1.96*delta.std()/np.sqrt(200),delta.mean()+1.96*delta.std()/np.sqrt(200)]))
    ratios=[]
    for treatment in ['Retained','Removed']:
        g=f[f.treatment==treatment].pivot(index='trial',columns='method',values='volume')
        ratios.append(dict(treatment=treatment,ratio_of_means=g.Envelope.mean()/g.Point_CHR.mean(),median_paired_ratio=(g.Envelope/g.Point_CHR).median(),envelope_smaller_fraction=float((g.Envelope<g.Point_CHR).mean())))
    report=dict(status='passed',new_fits=200,reused_removed_fits=200,method_records_checked=checks,scaling='Training-only target mean and population SD, same forest hyperparameters/model seed. Predictions inverted before residual calibration.',pairing='Original split membership retained; the removed case differs only by deletion of complete-case row 3660 (original index 4782) and training-dependent transformations/refits.',effects=effects,method_ratios=ratios,by_partition=f.groupby(['treatment','outlier_partition','method'])[['coverage','volume']].mean().reset_index().to_dict('records'),manifest=manifest)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
    print(summary.to_string(index=False));print(json.dumps(dict(effects=effects,method_ratios=ratios),indent=2))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
        fs=[pool.submit(one,i) for i in range(200)]
        for n,f in enumerate(as_completed(fs),1):
            t,s=f.result()
            if n%25==0:print(f'{n}/200 standardized retained-outlier fits complete',flush=True)
    finish()
