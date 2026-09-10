"""Complete the original auxiliary-baseline and 2D local-union comparisons."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'envelope_method')]
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from utility.conformal_utils import conformal_quantile, add_jitter
from utility.envelope import envelope_prediction, interval_sup, score_at
from utility.res_rescaled import standardized_prediction
from utility.exps import _function_choice, _stable_hash
from experiments import OUT, dump, generator, metric
from archive_storage import load_archive, source_archive_sha256


def link_vector(q, m, s, n):
    q=np.asarray(q)
    limit=n/np.sqrt(n+1)
    with np.errstate(invalid='ignore',divide='ignore'):
        value=m+(n+1)*s*q/np.sqrt(n*n-(n+1)*q*q)
    return np.where(q>=limit,np.inf,np.where(q<=-limit,-np.inf,value))


def local_2d(cal,test,alpha,exact_ratio=False):
    """Vectorized exhaustive cells; test membership is resolved by its cell index."""
    n,d=cal.shape
    assert d==2
    m,s=cal.mean(axis=0),cal.std(axis=0)
    g=envelope_prediction(cal,alpha).gwc_upper if exact_ratio else standardized_prediction(cal,alpha,method='GWC').upper
    points=[np.r_[0.,np.unique(cal[(cal[:,j]>0)&(cal[:,j]<g[j]),j]),g[j]] for j in range(2)]
    coordinates=[]
    for j in range(2):
        if exact_ratio:
            a=np.array([interval_sup(cal[:,j],m[j],s[j],n,lo,hi) for lo,hi in zip(points[j][:-1],points[j][1:])])
        else:
            center=np.clip(m[j],points[j][:-1],points[j][1:])
            r=np.hypot(s[j],(center-m[j])/np.sqrt(n+1))
            c=min(-float(score_at(0,m[j],s[j],n,0)),-float(score_at(0,m[j],s[j],n,g[j])))
            a=cal[:,j][None,:]/r[:,None]-c
        coordinates.append(a)
    shape=(len(points[0])-1,len(points[1])-1)
    indices=np.indices(shape).reshape(2,-1).T
    bounds=np.empty((len(indices),2))
    jitter=add_jitter(np.zeros(n)) if not exact_ratio else np.zeros(n)
    for offset in range(0,len(indices),128):
        ij=indices[offset:offset+128]
        scores=np.maximum(coordinates[0][ij[:,0]],coordinates[1][ij[:,1]])+jitter
        q=conformal_quantile(scores,alpha,axis=1)
        for j in range(2):
            bounds[offset:offset+len(ij),j]=np.minimum(points[j][ij[:,j]+1],link_vector(q,m[j],s[j],n))
    lower=np.column_stack([points[j][indices[:,j]] for j in range(2)])
    widths=np.maximum(bounds-lower,0)
    nonempty=np.all(bounds>=lower,axis=1)
    volume=float(np.prod(widths,axis=1).sum())
    bbox=np.max(bounds[nonempty],axis=0) if nonempty.any() else np.full(2,-np.inf)
    inside_g=np.all((test>=0)&(test<=g),axis=1)
    ij=np.column_stack([np.clip(np.searchsorted(points[j],test[:,j],side='right')-1,0,shape[j]-1) for j in range(2)])
    b=bounds[np.ravel_multi_index(ij.T,shape)]
    covered=inside_g & np.all(test<=b,axis=1)
    return volume, bbox, covered, int(nonempty.sum())


def run(item, include_full_lwc=False):
    cfg=item['config']
    original=OUT/'absolute'/item['id']
    dest=OUT/'auxiliary'/item['id']
    dest.mkdir(parents=True,exist_ok=True)
    dump(dest/'config.json',item)
    gen=generator(cfg)
    kwargs=dict(n_features=cfg['n_features'],n_informative=cfg['n_features'],n_targets=cfg['d'],
                noise_type=cfg['noise_type'],noise_list=cfg['noise_levels'],**cfg['generator_kwargs'])
    records=[]
    local=include_full_lwc and cfg['d']==2 and cfg['noise_type']=='Laplace' and cfg['n_cal'] in [30,50,100,300,500]
    for trial in range(item['trials']):
        path=dest/f'trial_{trial:03d}.json'
        if path.exists():
            records.extend(json.loads(path.read_text()))
            continue
        ready = original/f'trial_{trial:03d}.json'
        deadline = time.monotonic() + 14400
        while not ready.exists() or json.loads(ready.read_text()).get('version') != 2:
            if time.monotonic() > deadline:
                raise TimeoutError(f'Waiting for fresh primary trial: {ready}')
            time.sleep(2)
        primary_metadata=json.loads(ready.read_text())
        source=original/f'trial_{trial:03d}.npz'
        with load_archive(source,primary_metadata,required_keys=('scores_cal','scores_test','dgp_coef')) as saved:
            cal,test=saved['scores_cal'],saved['scores_test']
            coef=saved['dgp_coef'].copy()
            fitted_coef=saved['model_coef'].copy() if 'model_coef' in saved else None
            fitted_intercept=saved['model_intercept'].copy() if 'model_intercept' in saved else None
        rows,arrays=[],{}
        with threadpool_limits(limits=1),np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            for method in ['TSCP_S','Naive','Bonferroni']:
                start=time.perf_counter()
                reg=_function_choice(cal,cfg['alpha'],method)
                rows.append(metric(method,reg.upper,test,0.,time.perf_counter()-start))
                arrays[method]=reg.upper
            train_seed=_stable_hash(cfg['d'],cfg['n_cal'],trial,'train_test')
            if fitted_coef is None or fitted_intercept is None:
                raise ValueError(f'{source}: retained fitted coefficients are required for the population-oracle comparator; use a full/scores checkpoint. No model was refitted.')
            oracle_seed=_stable_hash(cfg['d'],cfg['n_cal'],trial,'oracle',3)
            xo,yo=gen(n_samples=100000,random_state=oracle_seed,coef=coef,**kwargs)
            oracle=abs(yo-(xo @ fitted_coef.T+fitted_intercept))
            mu,std=oracle.mean(axis=0),oracle.std(axis=0,ddof=1)
            start=time.perf_counter()
            reg=_function_choice(cal,cfg['alpha'],'Population_oracle',mu=mu,std=std)
            rows.append(metric('Population_oracle',reg.upper,test,0.,time.perf_counter()-start))
            arrays.update(Population_oracle=reg.upper,oracle_mean=mu,oracle_scale=std)
            if local:
                for exact_ratio in [False,True]:
                    start=time.perf_counter()
                    volume,bbox,covered,cells=local_2d(cal,test,cfg['alpha'],exact_ratio)
                    name='Exact_ratio_LWC' if exact_ratio else 'Old_LWC_union'
                    row=metric(name,bbox,test,0.,time.perf_counter()-start)
                    row.update(test_coverage=float(covered.mean()),covered_count=int(covered.sum()),
                               outcome_volume=4*volume,mean_log_volume=float(np.log(4*volume)),local_cells=cells,
                               note='union volume and coverage; coordinate lengths are enclosing box')
                    rows.append(row)
                    arrays[name]=bbox
        for row in rows:
            row.update(config_id=item['id'],trial=trial,alpha=cfg['alpha'],n_cal=cfg['n_cal'],n_dim=cfg['d'],
                       redraw_train_test=True,training_seed=train_seed,oracle_seed=oracle_seed,
                       source_archive=str(source.relative_to(ROOT)),
                       source_sha256=source_archive_sha256(primary_metadata))
        dump(path,rows)
        np.savez_compressed(dest/f'trial_{trial:03d}.npz',**arrays)
        records.extend(rows)
    frame=pd.DataFrame(records)
    frame.to_csv(dest/'trials.csv',index=False)
    frame.groupby('method')[['test_coverage','outcome_volume','runtime']].agg(['mean','std','count']).to_csv(dest/'summary.csv')
    dump(dest/'status.json',dict(status='complete',trials=item['trials']))
    return item['id']


def main():
    inventory=json.loads((OUT/'simulation_inventory.json').read_text())
    items=[i for i in inventory if i['config']['kind']=='absolute' and any(s.startswith('syn_exps') for s in i['sources'])]
    print('Fresh auxiliary configurations',len(items),'trials',sum(i['trials'] for i in items),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,item) for item in items]
        for i,future in enumerate(as_completed(futures),1):
            print('COMPLETE auxiliary',i,'/',len(items),future.result(),flush=True)


if __name__=='__main__':
    main()
