"""Retention contracts for metrics, audit scope, and old comparator lineage."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'envelope_method')]
from archive_storage import records_sha256,save_trial_archive,sha256_file
from final_audit import audit_trial
from summarize import read_trial_frame,summarize_frame
from verify_comparator_backfills import verify_sidecar
from utility.envelope import envelope_prediction
from run_cqr_baselines import complete_trial


def checkpoint(path,arrays,records,mode,source_hash=None):
    storage=save_trial_archive(path,arrays,mode)
    storage['records_sha256']=records_sha256(records)
    if source_hash:
        storage['source_archive_sha256']=source_hash
    payload=dict(version=2,records=records,storage=storage,archive_sha256=storage['archive_sha256'])
    path.with_suffix('.json').write_text(json.dumps(payload),encoding='utf-8')
    return payload


def test_legacy_real_split_json_rebuild_excludes_coordinate_records(tmp_path):
    records=[dict(dataset='energy',trial=0,method='Envelope',alpha=.1,
                  test_coverage=.92,outcome_volume=27.4,runtime=.01)]
    payload=dict(source='cache/energy/split_000.npz',source_sha256='legacy-source',
                 trials=records,coordinates=[dict(trial=0,coordinate=0,length=3.)])
    (tmp_path/'split_000.json').write_text(json.dumps(payload),encoding='utf-8')
    pd.DataFrame([dict(method='stale')]).to_csv(tmp_path/'trials.csv',index=False)
    pd.testing.assert_frame_equal(read_trial_frame(tmp_path,from_json=True),
                                  pd.DataFrame(records))
    (tmp_path/'trials.csv').unlink()
    pd.testing.assert_frame_equal(read_trial_frame(tmp_path),pd.DataFrame(records))


def test_compact_json_rebuild_and_explicit_nonfinite_statistics(tmp_path):
    records=[dict(method='Envelope',trial=i,test_coverage=.8+i*.01,
                  outcome_volume=v,runtime=.01,empty_rate=0) for i,v in enumerate([1.,3.,np.inf])]
    checkpoint(tmp_path/'trial_000.npz',{},records,'compact')
    frame=read_trial_frame(tmp_path)
    row=summarize_frame(frame,['method']).iloc[0]
    assert np.isposinf(row.outcome_volume_mean)
    assert np.isnan(row.outcome_volume_se)
    assert row.outcome_volume_median==3
    assert row.outcome_volume_finite_median==2
    assert row.outcome_volume_finite_count==2
    assert row.outcome_volume_posinf_count==1
    assert row.infinite_volume_fraction==pytest.approx(1/3)
    assert row.test_coverage_mean==np.mean(frame.test_coverage)
    assert row.test_coverage_se==np.std(frame.test_coverage,ddof=1)/np.sqrt(3)
    payload=json.loads((tmp_path/'trial_000.json').read_text())
    payload['records'][0]['test_coverage']=0
    (tmp_path/'trial_000.json').write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError,match='measurements'):
        read_trial_frame(tmp_path)


@pytest.mark.parametrize('mode',['full','scores','compact'])
def test_audit_reports_only_available_evidence(tmp_path,mode):
    rng=np.random.default_rng(19)
    cfg=dict(n_train=8,n_cal=12,n_test=5,n_features=2,d=2,kind='absolute',alpha=.1)
    arrays=dict(model_coef=np.zeros((2,2)),model_intercept=np.zeros(2))
    for split,n in [('train',8),('cal',12),('test',5)]:
        arrays['X_'+split]=rng.normal(size=(n,2))
        arrays['y_'+split]=abs(rng.normal(size=(n,2)))
    arrays['scores_cal']=arrays['y_cal']
    arrays['scores_test']=arrays['y_test']
    arrays['Envelope']=envelope_prediction(arrays['scores_cal'],.1).upper
    records=[dict(method='Envelope',redraw_train_test=True,fit_seconds=.1,outcome_volume=1.)]
    path=tmp_path/'trial_000.npz'
    metadata=checkpoint(path,arrays,records,mode)
    report=audit_trial(path,metadata,cfg,replay=True)
    assert report['archive_checked']==(mode!='compact')
    assert report['raw_data_checked']==(mode=='full')
    assert report['raw_fit_replayed']==(mode=='full')
    assert report['formula_replayed']==(mode!='compact')
    if mode!='full':
        assert 'unavailable' in report


@pytest.mark.parametrize('mode',['scores','compact'])
@pytest.mark.parametrize('relocated',[False,True])
def test_old_sidecar_hash_survives_retention_without_faking_metric_checks(tmp_path,mode,relocated):
    source=tmp_path/'source'/'trial_000.npz'
    source.parent.mkdir()
    score=np.array([[.1,.2],[.2,.3],[.7,.8]])
    base=np.full((3,2),2.)
    arrays=dict(scores_cal=score,scores_test=score,base_lengths_test=base,X_train=np.ones((4,2)))
    primary_records=[dict(method='Base',trial=0,redraw_train_test=True,fit_seconds=.1,
                          test_coverage=0.,outcome_volume=4.,covered_count=0,n_test=3)]
    original=checkpoint(source,arrays,primary_records,'full')
    source_hash=original['archive_sha256']
    # Work in temporary fixtures only, modeling a committed migration.
    source.unlink()
    checkpoint(source,arrays,primary_records,mode,source_hash=source_hash)
    if relocated:
        moved=tmp_path/'data'/'source'/source.name
        moved.parent.mkdir(parents=True)
        if source.exists():
            source.rename(moved)
        source.with_suffix('.json').rename(moved.with_suffix('.json'))
        source=moved
    sidecar=tmp_path/'baseline'/'trial_000.npz'
    sidecar.parent.mkdir()
    adjustment=np.array([.3,.4])
    rows=[dict(method='Unscaled',redraw_train_test=True,source_archive='source/trial_000.npz',
               source_sha256=source_hash,test_coverage=2/3,outcome_volume=2.6*2.8)]
    metadata=checkpoint(sidecar,dict(Unscaled=adjustment),rows,'full')
    metadata['source_sha256']=source_hash
    sidecar.with_suffix('.json').write_text(json.dumps(metadata),encoding='utf-8')
    result=verify_sidecar(sidecar.with_suffix('.json'),root=tmp_path)
    assert result['metrics_recomputed']==(1 if mode=='scores' else 0)
    assert bool(result.get('unavailable'))==(mode=='compact')
    if mode=='scores':
        with source.open('ab') as stream:
            stream.write(b'corruption')
        with pytest.raises(ValueError,match='checksum'):
            verify_sidecar(sidecar.with_suffix('.json'),root=tmp_path)


def test_compact_primary_comparators_are_reused_as_records(tmp_path):
    records=[dict(method=name,trial=0,redraw_train_test=True,fit_seconds=.1)
             for name in ['Envelope_signed','Unscaled','Empirical_copula']]
    source=tmp_path/'trial_000.npz'
    checkpoint(source,{},records,'compact')
    returned=complete_trial(source,tmp_path/'comparators',0,.1)
    assert returned==records[1:]
    assert not source.exists()


def test_compact_missing_comparator_does_not_refit(tmp_path):
    records=[dict(method='Envelope_signed',trial=0,redraw_train_test=True,fit_seconds=.1)]
    source=tmp_path/'trial_000.npz'
    checkpoint(source,{},records,'compact')
    with pytest.raises(FileNotFoundError,match='compact checkpoint'):
        complete_trial(source,tmp_path/'comparators',0,.1)
