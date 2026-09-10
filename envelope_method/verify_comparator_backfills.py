"""Check retained comparator sidecars and recompute metrics when arrays exist."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utility.project_paths import resolve_artifact
from envelope_method.archive_storage import (archive_hash_matches,get_storage_mode,
    load_archive,sha256_file,validate_archive,validate_records)
OUT=ROOT/'data/envelope_method/results'


def verify_sidecar(path,root=ROOT):
    metadata=json.loads(path.read_text())
    assert metadata['version']==2
    validate_records(metadata)
    source=resolve_artifact(metadata['records'][0]['source_archive'],root=root)
    original=json.loads(source.with_suffix('.json').read_text())
    validate_records(original)
    expected=metadata['source_sha256']
    # Old sidecars retain original-hash lineage; active bytes are checked too.
    linked=archive_hash_matches(original,expected)
    if not linked and get_storage_mode(original)=='full' and source.exists():
        linked=sha256_file(source)==expected
    assert linked,f'Comparator source provenance mismatch: {path}'
    for row in metadata['records']:
        assert row['redraw_train_test'] and row['source_sha256']==expected
    compact=get_storage_mode(original)=='compact' or get_storage_mode(metadata)=='compact'
    if get_storage_mode(original)!='compact':
        validate_archive(source,original)
    if get_storage_mode(metadata)!='compact':
        validate_archive(path.with_suffix('.npz'),metadata)
    if compact:
        return dict(records=len(metadata['records']),metrics_recomputed=0,
                    unavailable='Independent metric recomputation unavailable: source or sidecar is compact.')
    with load_archive(source,original,required_keys=('scores_test','base_lengths_test')) as data, \
            load_archive(path.with_suffix('.npz'),metadata) as results:
        score,base=data['scores_test'],data['base_lengths_test']
        for row in metadata['records']:
            adjustment=results[row['method']]
            coverage=np.all(score<=adjustment,axis=1).mean()
            widths=np.maximum(base+2*adjustment,0)
            with np.errstate(invalid='ignore'):
                volume=np.prod(widths,axis=1)
            volume[np.any(widths==0,axis=1)]=0
            np.testing.assert_allclose(coverage,row['test_coverage'],rtol=0,atol=1e-14)
            np.testing.assert_allclose(volume.mean(),row['outcome_volume'],rtol=1e-12,atol=1e-12)
            if row['method']=='Envelope_capped_0':
                assert np.all(adjustment>=0)
    return dict(records=len(metadata['records']),metrics_recomputed=len(metadata['records']))


def main():
    trials=records=recomputed=unavailable=0
    for family in ['cqr_baselines','toy_baselines']:
        for path in (OUT/family).glob('*/trial_*.json'):
            result=verify_sidecar(path)
            trials+=1
            records+=result['records']
            recomputed+=result['metrics_recomputed']
            unavailable+=int(bool(result.get('unavailable')))
    report=dict(status='passed',paired_trial_sidecars=trials,method_records=records,
                method_metrics_recomputed=recomputed,compact_sidecars_metrics_unavailable=unavailable,
                checks=['original source-hash lineage and active retained checksums','fresh protocol metadata',
                        'independent coverage and volume where arrays retained','accepted capped zero where arrays retained'],
                note='Compact metrics retain their saved values; absent arrays are explicitly unavailable, not independently verified.')
    (OUT/'comparator_retained_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
