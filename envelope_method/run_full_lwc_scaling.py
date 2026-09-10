"""Stream the original n=10 and n=30 full-LWC dimension-scaling benchmarks."""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiments import ROOT, OUT, dump, metric
from run_auxiliary import link_vector
from utility.envelope import score_at, interval_sup, envelope_prediction
from utility.res_rescaled import standardized_prediction
from utility.conformal_utils import conformal_quantile, add_jitter
from archive_storage import load_archive, source_archive_sha256


def local_nd(cal, test, alpha, exact=False):
    n, d = cal.shape
    m, s = cal.mean(0), cal.std(0)
    g = envelope_prediction(cal,alpha).gwc_upper if exact else standardized_prediction(cal,alpha,method='GWC').upper
    points = [np.r_[0.,np.unique(cal[(cal[:,j]>0)&(cal[:,j]<g[j]),j]),g[j]] for j in range(d)]
    coord = []
    for j in range(d):
        if exact:
            values = np.array([interval_sup(cal[:,j],m[j],s[j],n,a,b)
                               for a,b in zip(points[j][:-1],points[j][1:])])
        else:
            center = np.clip(m[j],points[j][:-1],points[j][1:])
            scale = np.hypot(s[j],(center-m[j])/np.sqrt(n+1))
            const = min(-float(score_at(0,m[j],s[j],n,0)), -float(score_at(0,m[j],s[j],n,g[j])))
            values = cal[:,j][None,:]/scale[:,None]-const
        coord.append(values)
    shape = tuple(len(p)-1 for p in points)
    cells = int(np.prod(shape))
    ij = np.column_stack([np.clip(np.searchsorted(points[j],test[:,j],side='right')-1,0,shape[j]-1) for j in range(d)])
    flat_test = np.ravel_multi_index(ij.T,shape)
    inside_g = np.all((test>=0)&(test<=g),axis=1)
    covered = np.zeros(len(test),bool)
    bbox = np.full(d,-np.inf)
    volume, accepted = 0., 0
    jitter = np.zeros(n) if exact else add_jitter(np.zeros(n))
    for start in range(0,cells,2048):
        indices = np.array(np.unravel_index(np.arange(start,min(start+2048,cells)),shape)).T
        transformed = np.full((len(indices),n),-np.inf)
        for j in range(d):
            transformed = np.maximum(transformed,coord[j][indices[:,j]])
        q = conformal_quantile(transformed+jitter,alpha,axis=1)
        lower = np.column_stack([points[j][indices[:,j]] for j in range(d)])
        bound = np.column_stack([np.minimum(points[j][indices[:,j]+1],link_vector(q,m[j],s[j],n)) for j in range(d)])
        valid = np.all(bound>=lower,axis=1)
        widths = np.maximum(bound-lower,0)
        with np.errstate(invalid='ignore'):
            volumes = np.prod(widths,axis=1)
        volumes[np.any(widths==0,axis=1)] = 0
        volume += float(volumes.sum())
        accepted += int(valid.sum())
        if valid.any():
            bbox = np.maximum(bbox,bound[valid].max(0))
        loc = np.flatnonzero(inside_g & (flat_test>=start)&(flat_test<start+len(indices)))
        covered[loc] = np.all(test[loc]<=bound[flat_test[loc]-start],axis=1)
    return volume, bbox, covered, accepted


def run(item, trials=None, output_family='full_lwc_scaling', exact_variants=(False, True)):
    cfg = item['config']
    count = (10 if cfg['n_cal']==10 else 30) if trials is None else int(trials)
    if not 0 < count <= item['trials']:
        raise ValueError('Trial count must be positive and not exceed the saved primary cohort.')
    if not output_family.isidentifier():
        raise ValueError('output_family must be a simple directory name.')
    dest = OUT/output_family/item['id']
    dest.mkdir(parents=True,exist_ok=True)
    dump(dest/'config.json',dict(item,trials=count,exact_variants=list(exact_variants)))
    records = []
    for trial in range(count):
        path = dest/f'trial_{trial:03d}.json'
        if path.exists():
            previous=json.loads(path.read_text())
            wanted={'Exact_ratio_LWC' if exact else 'Old_LWC_union' for exact in exact_variants}
            if wanted <= {r['method'] for r in previous}:
                records.extend(r for r in previous if r['method'] in wanted)
                continue
        with threadpool_limits(limits=1),np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            source = OUT/'absolute'/item['id']/f'trial_{trial:03d}.npz'
            source_metadata=json.loads(source.with_suffix('.json').read_text())
            with load_archive(source,source_metadata,required_keys=('scores_cal','scores_test')) as primary:
                cal=primary['scores_cal'].copy()
                test=primary['scores_test'].copy()
            rows,arrays = [],{}
            for exact in exact_variants:
                start=time.perf_counter()
                volume,bbox,covered,cells=local_nd(cal,test,cfg['alpha'],exact)
                name='Exact_ratio_LWC' if exact else 'Old_LWC_union'
                row=metric(name,bbox,test,0.,time.perf_counter()-start)
                row.update(test_coverage=float(covered.mean()),covered_count=int(covered.sum()),
                           outcome_volume=2**cfg['d']*volume,mean_log_volume=float(np.log(2**cfg['d']*volume)),
                           local_cells=cells,trial=trial,config_id=item['id'],redraw_train_test=True,
                           source_archive=str(source.relative_to(ROOT)),
                           source_sha256=source_archive_sha256(source_metadata))
                rows.append(row)
                arrays[name]=bbox
        dump(path,rows)
        np.savez_compressed(dest/f'trial_{trial:03d}.npz',**arrays)
        records.extend(rows)
    df=pd.DataFrame(records)
    df.to_csv(dest/'trials.csv',index=False)
    dump(dest/'status.json',dict(status='complete',trials=count))
    return item['id']


if __name__=='__main__':
    raise SystemExit('Full LWC is manual-only. Open envelope_method/full_lwc_manual.ipynb.')
