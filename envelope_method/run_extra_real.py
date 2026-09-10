"""Reconstruct the two archived real-data studies without residual caches."""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'envelope_method')]
for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(var,'1')
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from utility.exps import _load_real_experiment_data, _stable_hash
from experiments import evaluate_absolute, dump

OUT = ROOT / 'data/envelope_method/results/extra_real'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['air','crime']:
        dest = OUT / name
        dest.mkdir(exist_ok=True)
        if (dest/'dataset.pkl').exists():
            continue
        X,y,model = _load_real_experiment_data(name)
        model.set_params(n_jobs=1)
        with (dest/'dataset.pkl').open('wb') as file:
            pickle.dump((X,y,model),file)
        dump(dest/'metadata.json',dict(dataset=name, rows=len(X), features=X.shape[1],outputs=y.shape[1],
                                      targets=list(y.columns),trials=200,cal_fraction=.05,test_fraction=.2,
                                      model_params=model.get_params(),
                                      provenance='reconstructed with current repository loader; old raw splits were not cached'))
        print('PREPARED',name,X.shape,y.shape,flush=True)


def one(name, trial):
    dest = OUT / name
    saved = dest/f'trial_{trial:03d}.json'
    if saved.exists():
        return name, trial
    with (dest/'dataset.pkl').open('rb') as file:
        X,y,template = pickle.load(file)
    seed = _stable_hash(trial)
    train,rest = train_test_split(np.arange(len(X)),test_size=.25,random_state=seed)
    cal,test = train_test_split(rest,test_size=.8,random_state=seed)
    model = clone(template)
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        model.fit(X.iloc[train],y.iloc[train])
        ec = abs(y.iloc[cal].to_numpy()-model.predict(X.iloc[cal]))
        et = abs(y.iloc[test].to_numpy()-model.predict(X.iloc[test]))
        fit_seconds=time.perf_counter()-start
        with np.errstate(divide='ignore',invalid='ignore'):
            rows, bounds = evaluate_absolute(ec,et,.1)
    for row in rows:
        row.update(dataset=name,trial=trial,alpha=.1,n_cal=len(cal),n_dim=y.shape[1],fit_seconds=fit_seconds)
    np.savez_compressed(dest/f'trial_{trial:03d}.npz',scores_cal=ec,scores_test=et,
                        train_indices=train,cal_indices=cal,test_indices=test,**bounds)
    dump(saved,rows)
    return name,trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','run'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare()
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(one,name,trial) for name in ['air','crime'] for trial in range(200)]
        for i,job in enumerate(as_completed(jobs),1):
            name,trial=job.result()
            if i%25==0:
                print('COMPLETE extra real',i,'/400',name,trial,flush=True)
    for name in ['air','crime']:
        dest=OUT/name
        rows=[r for path in sorted(dest.glob('trial_*.json')) for r in json.loads(path.read_text())]
        frame=pd.DataFrame(rows)
        frame.to_csv(dest/'trials.csv',index=False)
        frame.groupby('method')[['test_coverage','outcome_volume','runtime']].agg(['mean','std','count']).to_csv(dest/'summary.csv')
        dump(dest/'status.json',dict(status='complete',trials=frame.trial.nunique()))


if __name__=='__main__':
    main()
