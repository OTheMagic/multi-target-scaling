"""Independent saved-artifact checks; no regeneration or tuning of experiments."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/envelope_method/report_revision'
sys.path[:0]=[str(ROOT),str(HERE)]
from utility.envelope import envelope_prediction
from experiment_suite import measures,fit_regions

manifest=json.loads((DATA/'manifest.json').read_text())
trials=pd.read_csv(DATA/'trials.csv')
checks=dict(archive_hashes=0,training_hashes=0,recomputed_trial_metrics=0,
            recomputed_algorithm_pairs=0,translation_invariance=0)
training_hashes=set()
for entry in manifest:
    path=DATA/entry['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    checks['archive_hashes']+=1
    with np.load(path) as a:
        digest=hashlib.sha256(a['X_train'].tobytes()+a['Y_train'].tobytes()).hexdigest()
        assert digest not in training_hashes
        training_hashes.add(digest)
        saved=json.loads(path.with_suffix('.json').read_text())
        for split in ['cal','test']:
            pred=np.column_stack([np.ones(len(a['X_'+split])),a['X_'+split]])@a['model_coef']
            np.testing.assert_allclose(pred,a['pred_'+split],atol=1e-12)
            np.testing.assert_allclose(abs(a['Y_'+split]-pred),a['abs_'+split],atol=1e-12)
            raw=np.maximum(a['lo_'+split]-a['Y_'+split],a['Y_'+split]-a['hi_'+split])
            np.testing.assert_array_equal(raw,a['raw_'+split])
        bounds={m:a['bound_'+m] for m in ['abs_env','abs_old','cap_env','cap_old','signed_env','signed_gwc','base']}
        flags={m:a['empty_'+m] for m in bounds}
        rows,audit=measures(a,bounds,flags)
        for row in rows:
            old=next(r for r in saved['records'] if r['method']==row['method'])
            for field in row:
                if field!='method':np.testing.assert_allclose(row[field],old[field],atol=1e-12,rtol=1e-12)
        checks['recomputed_trial_metrics']+=1
        if entry['trial'] in [0,59,119]:
            fresh,flags2=fit_regions(a,saved['spec']['alpha'])
            for m in bounds:
                np.testing.assert_allclose(fresh[m],bounds[m],rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(flags2[m],flags[m])
            checks['recomputed_algorithm_pairs']+=1
checks['training_hashes']=len(training_hashes)
rng=np.random.default_rng(2026090928)
for _ in range(120):
    cal=rng.lognormal(size=(40,3));shift=rng.uniform(.2,4,size=3)
    original=envelope_prediction(cal,.1)
    shifted=envelope_prediction(cal-shift,.1,lower=-shift)
    np.testing.assert_allclose(shifted.upper+shift,original.upper,atol=1e-10,rtol=1e-10)
    checks['translation_invariance']+=1
assert len(trials)==2400*7
report=dict(status='passed',checks=checks,
    reproduction_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [HERE/'experiment_suite.py',HERE/'plot_report.py',HERE/'build_report.py',Path(__file__)]})
(DATA/'verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
