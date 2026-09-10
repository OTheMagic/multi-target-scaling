"""Verify the evidence retained by full, scores, and compact trial checkpoints."""
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from utility.envelope import envelope_prediction, EnvelopeCalibration
from envelope_method.archive_storage import get_storage_mode, load_archive, validate_records


def audit_trial(archive,metadata,cfg,*,replay=False):
    """Audit available evidence without upgrading compact data or refitting."""
    assert metadata['version']==2
    validate_records(metadata)
    records=metadata['records']
    assert records and all(r['redraw_train_test'] and r['fit_seconds']>0 for r in records)
    mode=get_storage_mode(metadata)
    result=dict(mode=mode,archive_checked=False,raw_data_checked=False,
                formula_replayed=False,raw_fit_replayed=False,signatures={},
                nan_volume_records=sum(np.isnan(r['outcome_volume']) for r in records))
    if mode=='compact':
        result['unavailable']='Raw observations, residual checks and formula replay unavailable: compact storage retains metrics only.'
        return result
    with load_archive(archive,metadata,required_keys=('scores_cal','scores_test')) as data:
        result['archive_checked']=True
        assert data['scores_cal'].shape==(cfg['n_cal'],cfg['d'])
        assert data['scores_test'].shape==(cfg['n_test'],cfg['d'])
        if mode=='full':
            for split,size in [('train',cfg['n_train']),('cal',cfg['n_cal']),('test',cfg['n_test'])]:
                X=data['X_'+split]
                assert X.shape==(size,cfg['n_features'])
                assert data['y_'+split].shape==(size,cfg['d'])
                result['signatures'][split]=hashlib.sha256(X[:2].tobytes()).hexdigest()
            result['raw_data_checked']=True
        else:
            result['unavailable']='Raw draw distinctness and fitted-score reconstruction unavailable: raw observations were not retained.'
        if replay:
            cal=data['scores_cal']
            if cfg['kind']=='absolute':
                if mode=='full':
                    predicted=data['X_cal'] @ data['model_coef'].T+data['model_intercept']
                    np.testing.assert_allclose(abs(data['y_cal']-predicted),cal,rtol=1e-10,atol=1e-9)
                    result['raw_fit_replayed']=True
                reg=envelope_prediction(cal,cfg['alpha'])
                np.testing.assert_allclose(reg.upper,data['Envelope'],rtol=1e-12,atol=1e-12)
            else:
                reg=EnvelopeCalibration(cal,cfg['alpha']).predict(-data['base_lengths_test'][0]/2)
                np.testing.assert_allclose(reg.upper,data['Envelope_signed'][0],rtol=1e-12,atol=1e-12)
            result['formula_replayed']=True
    return result


def main():
    inventory=[]
    for name in ['settings.json','notebook_settings.json','repair_settings.json']:
        path=ROOT/'envelope_method'/name
        if path.exists():
            inventory+=json.loads(path.read_text())
    count=0
    totals=Counter()
    modes=Counter()
    rows=[]
    for index,item in enumerate(inventory):
        cfg=item['config']
        assert cfg['redraw'] is True
        folder=ROOT/'data/envelope_method/results'/cfg['kind']/item['id']
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='complete' and status['completed']==item['trials']
        signatures={split:set() for split in ['train','cal','test']}
        local=Counter()
        for trial in range(item['trials']):
            path=folder/f'trial_{trial:03d}.json'
            metadata=json.loads(path.read_text())
            result=audit_trial(path.with_suffix('.npz'),metadata,cfg,replay=trial==0)
            modes[result['mode']]+=1
            for key in ['archive_checked','raw_data_checked','formula_replayed','raw_fit_replayed','nan_volume_records']:
                local[key]+=int(result[key])
                totals[key]+=int(result[key])
            for split,signature in result['signatures'].items():
                signatures[split].add(signature)
            count+=1
        # Missing arrays are never presented as fresh-data verification.
        assert all(len(s)==local['raw_data_checked'] for s in signatures.values())
        rows.append(dict(config_id=item['id'],trials=item['trials'],
                         distinct_training=len(signatures['train']) if local['raw_data_checked'] else None,
                         distinct_calibration=len(signatures['cal']) if local['raw_data_checked'] else None,
                         distinct_testing=len(signatures['test']) if local['raw_data_checked'] else None,
                         raw_observation_trials_unavailable=item['trials']-local['raw_data_checked'],**local))
        if (index+1)%20==0:
            print('Audited',index+1,'configurations;',count,'trial checkpoints',flush=True)
    versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','scikit-learn','matplotlib']}
    files=['utility/envelope.py','utility/exps.py','utility/res_rescaled.py','utility/data_generator.py',
           'utility/cqhr.py','utility/conformal_utils.py','envelope_method/experiments.py',
           'envelope_method/run_toys.py','envelope_method/archive_storage.py']
    report=dict(status='passed',configurations=len(rows),trials_checked=count,
                trial_archives=totals['archive_checked'],storage_modes=dict(modes),
                checks_performed=dict(totals),raw_observation_trials_unavailable=count-totals['raw_data_checked'],
                configurations_checked=rows,python=sys.version,platform=platform.platform(),packages=versions,
                code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                note='Passed checks applicable to retained evidence. Full: archive, raw shapes/distinctness, first-trial formula and absolute fitted-score replay. Scores: archive, score shapes and first-trial formula replay. Compact: records/protocol metadata only. Missing raw evidence is unavailable, not verified. Full LWC not rerun.')
    # Preserve the historical full-data verification as dated provenance.
    # A retention audit has a different verification scope and its own report.
    (ROOT/'data/envelope_method/results/retained_storage_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS',count,'checkpoints;',totals['archive_checked'],'retained archives; storage',dict(modes),flush=True)


if __name__=='__main__':
    main()
