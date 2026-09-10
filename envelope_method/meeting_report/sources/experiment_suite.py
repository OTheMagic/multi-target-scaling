"""Fresh fitted, paired toys for the September 9 envelope/shortcut report.

Run from the repository root. No external datasets, reused fitted pools, or
full-cell LWC computations. Checkpoints retain full observations and predictions.
"""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/mpl'))
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from utility.envelope import EnvelopeCalibration, envelope_prediction, score_at
from utility.res_rescaled import standardized_prediction
from utility.conformal_utils import conformal_rank, conformal_quantile

OUT = Path(__file__).resolve().parent
DATA = OUT / 'data'
FAMILIES = ['gaussian', 'correlated', 'laplace', 'skewed']
LABELS = dict(gaussian='Gaussian', correlated='Correlated Gaussian',
              laplace='Laplace', skewed='Skewed lognormal')
MASTER_SEED = 2026090917
REPS = 120

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=True), encoding='utf-8')

def generate(seed, family, n, d=6, train=800, test=1200, base_alpha=.1):
    rng = np.random.default_rng(seed)
    beta = np.sin(np.arange(3*d).reshape(3,d)+1)/2
    scales = np.geomspace(.6, 2.4, d)
    arrays = {}
    for split, count in [('train',train), ('cal',n), ('test',test)]:
        x = rng.normal(size=(count,3))
        group = (x[:,0] > 0).astype(int)
        if family in ('gaussian','correlated'):
            noise = rng.normal(size=(count,d))
            if family == 'correlated':
                noise = np.sqrt(.2)*noise + np.sqrt(.8)*rng.normal(size=(count,1))
        elif family == 'laplace':
            noise = rng.laplace(size=(count,d))/np.sqrt(2)
        elif family == 'skewed':
            noise = (np.exp(.8*rng.normal(size=(count,d))) - np.exp(.8**2/2))
            noise /= np.sqrt((np.exp(.8**2)-1)*np.exp(.8**2))
        else:
            raise ValueError(family)
        y = x@beta + (1+.8*group[:,None])*scales*noise
        arrays['X_'+split], arrays['Y_'+split], arrays['group_'+split] = x,y,group
    xt = np.column_stack([np.ones(train), arrays['X_train']])
    fitted = np.linalg.lstsq(xt, arrays['Y_train'], rcond=None)[0]
    train_error = arrays['Y_train']-xt@fitted
    quantiles = np.array([np.quantile(train_error[arrays['group_train']==g],
                                     [base_alpha/2, 1-base_alpha/2], axis=0)
                          for g in [0,1]])
    arrays.update(model_coef=fitted, fitted_error_quantiles=quantiles,
                  beta=beta, scales=scales)
    for split in ['cal','test']:
        pred = np.column_stack([np.ones(len(arrays['X_'+split])),arrays['X_'+split]])@fitted
        group = arrays['group_'+split]
        lo,hi = pred+quantiles[group,0],pred+quantiles[group,1]
        raw = np.maximum(lo-arrays['Y_'+split],arrays['Y_'+split]-hi)
        arrays.update({f'pred_{split}':pred, f'lo_{split}':lo, f'hi_{split}':hi,
                       f'abs_{split}':abs(arrays['Y_'+split]-pred),
                       f'raw_{split}':raw, f'width_{split}':hi-lo})
    return arrays

def old_bound(scores, alpha, diagnostics=None):
    # Identical conservative extension for the singular case, excluded by the theorem.
    if conformal_rank(len(scores),alpha)>len(scores) or np.any(scores.std(0)==0):
        return np.full(scores.shape[1],np.inf)
    with np.errstate(all='ignore'):
        return standardized_prediction(scores,alpha, diagnostics=diagnostics).upper

def fit_regions(a, alpha):
    bounds, flags = {}, {}
    for score_name, prefix in [('abs','abs'),('raw','cap')]:
        cal = a[score_name+'_cal']
        if prefix == 'cap':
            cal = np.maximum(cal,0)
        reg = envelope_prediction(cal,alpha)
        bounds[prefix+'_env'] = reg.upper
        bounds[prefix+'_old'] = old_bound(cal,alpha)
        flags[prefix+'_env'] = reg.empty
        flags[prefix+'_old'] = False
    model = EnvelopeCalibration(a['raw_cal'],alpha)
    widths = a['fitted_error_quantiles'][:,1]-a['fitted_error_quantiles'][:,0]
    regs = [model.predict(-w/2) for w in widths]
    bounds['signed_env'] = np.array([r.upper for r in regs])
    bounds['signed_gwc'] = np.array([r.gwc_upper for r in regs])
    flags['signed_env'] = np.array([r.empty for r in regs])
    flags['signed_gwc'] = np.any(bounds['signed_gwc'] < -widths/2,axis=1)
    bounds['base'] = np.zeros((2, a['raw_cal'].shape[1]))
    flags['base'] = np.zeros(2,dtype=bool)
    return bounds,flags

def measures(a, bounds, flags):
    records=[]
    length_arrays={}
    for method,b in bounds.items():
        absolute=method.startswith('abs')
        raw=a['abs_test'] if absolute else a['raw_test']
        upper=np.broadcast_to(b,raw.shape) if b.ndim==1 else b[a['group_test']]
        empty=np.broadcast_to(flags[method],(len(raw),)) if np.ndim(flags[method])==0 else flags[method][a['group_test']]
        coordinate_hit=(raw<=upper) & ~empty[:,None]
        width=np.maximum((0 if absolute else a['width_test'])+2*upper,0)
        width[empty]=0
        with np.errstate(all='ignore'):
            vol=np.prod(width,axis=1)
        vol[np.any(width==0,axis=1)]=0
        length_arrays[method]=width
        row=dict(method=method,coverage=float(coordinate_hit.all(1).mean()),
                 volume=float(vol.mean()),infinite=float(np.isinf(vol).mean()),
                 empty=float(empty.mean()),negative_adjustment=float((upper<0).mean()))
        for j in range(raw.shape[1]):
            row['length_'+str(j+1)]=float(width[:,j].mean())
            row['coverage_'+str(j+1)]=float(coordinate_hit[:,j].mean())
        records.append(row)
    checks={}
    for prefix in ['abs','cap']:
        en,old=bounds[prefix+'_env'],bounds[prefix+'_old']
        nondegenerate=bool(np.all((a['abs_cal'] if prefix=='abs' else np.maximum(a['raw_cal'],0)).std(0)>0))
        tol=1e-8*(1+abs(old))
        violation=bool(not flags[prefix+'_env'] and np.any(en>old+tol))
        checks[prefix+'_violation']=violation
        checks[prefix+'_nondegenerate']=nondegenerate
        assert not violation
        f=np.isfinite(old)&np.isfinite(en)
        checks[prefix+'_max_excess']=float(np.max((en[f]-old[f])/(1+abs(old[f])))) if np.any(f) else None
        # Nested-set coverage must also hold for the SAME realized test outcomes.
        rr={r['method']:r for r in records}
        assert rr[prefix+'_env']['coverage']<=rr[prefix+'_old']['coverage']+1e-14
    return records,checks

def configurations():
    specs=[]
    for family in FAMILIES:
        for n in [30,80,200]:
            specs.append(dict(family=family,n=n,d=6,alpha=.1,base_alpha=.1,reps=REPS,study='main'))
    for family in ['gaussian','laplace']:
        for alpha in [.05,.2,.3]:
            specs.append(dict(family=family,n=80,d=6,alpha=alpha,base_alpha=.1,reps=REPS,study='alpha'))
    for ba in [.1,.02]:
        specs.append(dict(family='gaussian',n=80,d=2,alpha=.1,base_alpha=ba,reps=REPS,study='base_width'))
    return specs

def simulations():
    DATA.mkdir(parents=True,exist_ok=True)
    specs=configurations()
    dump(OUT/'design.json', dict(seed=MASTER_SEED,n_train=800,n_test=1200,
         independent_fresh_trials=True, quantile_fit='OLS plus groupwise training-error quantiles',
         specs=specs,protocol_version=1))
    all_rows=[];checks=[];manifest=[]
    started=time.perf_counter()
    for k,spec in enumerate(specs):
        folder=DATA/f'config_{k:02d}'
        folder.mkdir(exist_ok=True)
        for trial in range(spec['reps']):
            seed=MASTER_SEED+10000*k+trial
            path=folder/f'trial_{trial:03d}.json'
            archive=path.with_suffix('.npz')
            if path.exists() and archive.exists():
                saved=json.loads(path.read_text())
            else:
                a=generate(seed,spec['family'],spec['n'],spec['d'],base_alpha=spec['base_alpha'])
                bounds,flags=fit_regions(a,spec['alpha'])
                rows,audit=measures(a,bounds,flags)
                a.update({'bound_'+m:b for m,b in bounds.items()})
                a.update({'empty_'+m:np.asarray(v) for m,v in flags.items()})
                np.savez_compressed(archive,**a)
                digest=hashlib.sha256(archive.read_bytes()).hexdigest()
                saved=dict(seed=seed,trial=trial,config=k,spec=spec,records=rows,checks=audit,sha256=digest)
                dump(path,saved)
            info=dict(config=k,trial=trial,seed=seed,**spec)
            all_rows.extend([dict(**info,**row) for row in saved['records']])
            checks.append(dict(**info,**saved['checks']))
            manifest.append(dict(config=k,trial=trial,path=str(archive.relative_to(OUT)),sha256=saved['sha256']))
        print(f'Completed {k+1}/{len(specs)}: {spec}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    pd.DataFrame(all_rows).to_csv(OUT/'trials.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'containment.csv',index=False)
    dump(OUT/'manifest.json',manifest)
    dump(OUT/'run_metadata.json',dict(completed_trials=len(checks),seconds=time.perf_counter()-started,
        numpy=np.__version__,pandas=pd.__version__,python=platform.python_version(),platform=platform.platform(),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in [Path(__file__),ROOT/'utility/envelope.py',ROOT/'utility/res_rescaled.py',ROOT/'utility/conformal_utils.py']}))

def benchmark():
    """Serial, randomized paired call order. Includes statistics/sorting/prediction.

    Each time is the median of three independent calls after an untimed warmup;
    15 new fitted arrays per (n,d) cell. Excludes training, IO and coverage testing.
    """
    rows=[]
    for d in [2,6,12]:
        for n in [30,80,200,500,1000]:
            for trial in range(15):
                a=generate(MASTER_SEED+500000+d*10000+n*20+trial,'gaussian',n,d,test=10)
                for mode in ['abs','cap']:
                    cal=a['abs_cal'] if mode=='abs' else np.maximum(a['raw_cal'],0)
                    methods=dict(old=lambda:old_bound(cal,.1),
                        env=lambda:envelope_prediction(cal,.1),
                        rank=lambda:envelope_prediction(cal,.1,search='rank'))
                    if mode=='cap':
                        def signed_call():
                            model=EnvelopeCalibration(a['raw_cal'],.1)
                            widths=a['fitted_error_quantiles'][:,1]-a['fitted_error_quantiles'][:,0]
                            return [model.predict(-w/2) for w in widths]
                        methods['signed']=signed_call
                    for call in methods.values():call()
                    times={m:[] for m in methods}
                    rng=np.random.default_rng(trial+n+d)
                    for _ in range(3):
                        for method in rng.permutation(list(methods)):
                            start=time.perf_counter_ns();methods[method]()
                            times[method].append((time.perf_counter_ns()-start)/1e6)
                    env=envelope_prediction(cal,.1)
                    rank=envelope_prediction(cal,.1,search='rank')
                    diag={};old_bound(cal,.1,diag)
                    np.testing.assert_allclose(env.upper,rank.upper,atol=1e-11,rtol=1e-11)
                    for method in methods:
                        evaluations=sum(env.evaluations) if method=='env' else sum(rank.evaluations) if method=='rank' else sum(sum(r.evaluations) for r in signed_call()) if method=='signed' else sum(diag.get('binary_evaluations',[]))+sum(diag.get('backward_evaluations',[]))
                        rows.append(dict(mode=mode,n=n,d=d,trial=trial,method=method,
                             ms=float(np.median(times[method])),evaluations=evaluations))
            print(f'Benchmark d={d}, n={n}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'runtime.csv',index=False)

def geometry():
    # A priori trial and test indices; no search for maximum visual gain.
    rows=[]
    for family,seed in [('laplace',MASTER_SEED+900001),('skewed',MASTER_SEED+900002)]:
        a=generate(seed,family,30,d=2,test=1200)
        bounds,flags=fit_regions(a,.1)
        a.update({'bound_'+m:b for m,b in bounds.items()})
        a.update({'empty_'+m:np.asarray(v) for m,v in flags.items()})
        np.savez_compressed(OUT/f'geometry_{family}.npz',**a)
        rows.append(dict(family=family,seed=seed,trial=0,test_index=0,n=30,d=2))
    dump(OUT/'geometry_design.json',rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['simulate','benchmark','geometry','all'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.mode in ['simulate','all']:simulations()
        if args.mode in ['benchmark','all']:benchmark()
        if args.mode in ['geometry','all']:geometry()

if __name__=='__main__':main()
