"""Single predeclared largest-population removal; no iterative win-seeking trim."""
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
SOURCE=ROOT/'data/envelope_method/results/extra_real/crime'
OUT=ROOT/'data/envelope_method/results/real_outlier_screen/crime_control'

def init():
    global X,y,template,removed
    with (SOURCE/'dataset.pkl').open('rb') as f:X,y,template=pickle.load(f)
    removed=int(np.argmax(X['pop'].to_numpy()))

def evaluate(cal,test):
    env=envelope_prediction(cal,.1).upper
    a,b=train_test_split(cal,test_size=.5,random_state=42)
    shape=conformal_quantile(a,.1,axis=0)
    chr_u=shape*conformal_quantile((b/shape).max(1),.1)
    return dict(Envelope=env,Point_CHR=chr_u)

def one(task):
    trial,drop,scaled=task
    name=f'trial_{trial:03d}_{"removed" if drop else "retained"}_{"scaled" if scaled else "raw"}'
    path=OUT/(name+'.json')
    if path.exists():return name,0
    with np.load(SOURCE/f'trial_{trial:03d}.npz') as z:parts={k:z[k+'_indices'].copy() for k in ['train','cal','test']}
    where=next(k for k,v in parts.items() if removed in v)
    if drop:parts={k:v[v!=removed] for k,v in parts.items()}
    a=y.iloc[parts['train']].to_numpy();center=a.mean(0) if scaled else np.zeros(a.shape[1]);scale=a.std(0) if scaled else np.ones(a.shape[1])
    model=clone(template);model.set_params(n_jobs=1)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X.iloc[parts['train']],(a-center)/scale)
        cal=abs(model.predict(X.iloc[parts['cal']])*scale+center-y.iloc[parts['cal']].to_numpy())
        test=abs(model.predict(X.iloc[parts['test']])*scale+center-y.iloc[parts['test']].to_numpy())
        bounds=evaluate(cal,test)
    sec=time.perf_counter()-start;rows=[];common=parts['test']!=removed
    for method,u in bounds.items():
        covered=(test<=u).all(1)
        rows.append(dict(trial=trial,removed=drop,scaled=scaled,method=method,coverage=float(covered.mean()),common_test_coverage=float(covered[common].mean()),volume=float(np.prod(2*u)),outlier_partition=where,n_train=len(parts['train']),n_cal=len(cal),n_test=len(test),fit_seconds=sec))
    np.savez_compressed(path.with_suffix('.npz'),scores_cal=cal,scores_test=test,train_center=center,train_scale=scale,**bounds,**{k+'_indices':v for k,v in parts.items()})
    path.write_text(json.dumps(rows,indent=2));return name,sec

def finish():
    init();rows=[];checked=0
    files=sorted(OUT.glob('trial_*.json'));assert len(files)==600
    for p in files:
        rr=json.loads(p.read_text());rows.extend(rr)
        with np.load(p.with_suffix('.npz')) as z:
            for r in rr:
                np.testing.assert_allclose([(z['scores_test']<=z[r['method']]).all(1).mean(),np.prod(2*z[r['method']])],[r['coverage'],r['volume']],rtol=1e-12)
                checked+=1
            if rr[0]['removed']:assert all(removed not in z[k+'_indices'] for k in ['train','cal','test'])
            trial=rr[0]['trial']
            with np.load(SOURCE/f'trial_{trial:03d}.npz') as original:
                for k in ['train','cal','test']:
                    expected=original[k+'_indices'];expected=expected[expected!=removed] if rr[0]['removed'] else expected
                    np.testing.assert_array_equal(z[k+'_indices'],expected)
            if rr[0]['scaled']:
                yy=y.iloc[z['train_indices']].to_numpy()
                np.testing.assert_allclose(z['train_center'],yy.mean(0),rtol=1e-13)
                np.testing.assert_allclose(z['train_scale'],yy.std(0),rtol=1e-13)
    # Reuse original raw-model fits, with the corrected CHR formula.
    for trial in range(200):
        with np.load(SOURCE/f'trial_{trial:03d}.npz') as z:
            cal,test=z['scores_cal'],z['scores_test'];bounds=evaluate(cal,test)
            where=next(k for k in ['train','cal','test'] if removed in z[k+'_indices'])
            for method,u in bounds.items():
                inside=(test<=u).all(1)
                rows.append(dict(trial=trial,removed=False,scaled=False,method=method,coverage=float(inside.mean()),common_test_coverage=float(inside[z['test_indices']!=removed].mean()),volume=float(np.prod(2*u)),outlier_partition=where,n_cal=len(cal),n_test=len(test)))
    f=pd.DataFrame(rows);f.to_csv(OUT/'trials.csv',index=False)
    summary=f.groupby(['scaled','removed','method']).agg(trials=('trial','size'),coverage=('coverage','mean'),coverage_sd=('coverage','std'),common_test_coverage=('common_test_coverage','mean'),volume=('volume','mean'),volume_sd=('volume','std'),median_volume=('volume','median')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    ratios=[]
    for (scaled,drop),g in f.groupby(['scaled','removed']):
        v=g.pivot(index='trial',columns='method',values='volume')
        ratios.append(dict(scaled=bool(scaled),removed=bool(drop),ratio_of_means=v.Envelope.mean()/v.Point_CHR.mean(),median_paired_ratio=(v.Envelope/v.Point_CHR).median(),envelope_smaller_fraction=float((v.Envelope<v.Point_CHR).mean())))
    report=dict(status='passed',new_fits=600,reused_original_fits=200,new_method_metrics_checked=checked,method_ratios=ratios,qualification='Largest-population row chosen once using an input feature before this deletion experiment. This is a population-sensitivity test, not an assertion of erroneous data; no additional deletions or threshold search.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(summary.to_string(index=False));print(json.dumps(ratios,indent=2))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(exist_ok=True);init()
    metadata=dict(removal_rule='Remove the one row with the maximum pop feature; fixed before examining deletion outcomes.',removed_position=removed,original_index=str(y.index[removed]),population=float(X.iloc[removed]['pop']),next_largest_population=float(np.sort(X['pop'].to_numpy())[-2]),rows_before=len(y),rows_after=len(y)-1,model_params=template.get_params(),source_sha256=hashlib.sha256((SOURCE/'dataset.pkl').read_bytes()).hexdigest())
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2),flush=True)
    tasks=[(i,drop,scaled) for i in range(200) for drop,scaled in [(True,False),(False,True),(True,True)]]
    with ProcessPoolExecutor(max_workers=2,initializer=init) as pool:
        fs=[pool.submit(one,t) for t in tasks]
        for n,f in enumerate(as_completed(fs),1):
            name,s=f.result()
            if n%100==0:print(f'{n}/600 Crime fits complete',flush=True)
    finish()
