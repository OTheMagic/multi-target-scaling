"""Add omitted comparators to already-fresh paired trials, without rerunning full LWC."""
import hashlib
import json
import time
import numpy as np
import pandas as pd
from experiments import ROOT, OUT, dump, evaluate_raw_cqr_baselines, metric
from utility.envelope import envelope_prediction
from utility.res_rescaled import standardized_prediction
from archive_storage import (archive_hash_matches, get_storage_mode, load_archive,
                             source_archive_sha256, validate_archive, validate_records)


def complete_trial(source, dest, trial, alpha, toy=False):
    metadata = json.loads(source.with_suffix('.json').read_text())
    validate_records(metadata)
    assert metadata['version'] == (3 if toy else 2)
    assert all(r['redraw_train_test'] for r in metadata['records'])
    checksum = source_archive_sha256(metadata)
    if checksum is None and source.exists():
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if not toy:
        assert all(r['fit_seconds'] > 0 for r in metadata['records'])
    dest.mkdir(parents=True, exist_ok=True)
    target = dest/source.with_suffix('.json').name
    wanted = {'Unscaled', 'Empirical_copula'}
    if toy:
        wanted |= {'Envelope_capped_0', 'TSCP_R_capped_0', 'TSCP_GWC_capped_0'}
    if target.exists():
        prior = json.loads(target.read_text())
        validate_records(prior)
        matches = archive_hash_matches(metadata,prior.get('source_sha256')) or (checksum is not None and prior.get('source_sha256') == checksum)
        if prior.get('version') == 2 and matches and wanted <= {r['method'] for r in prior['records']}:
            if get_storage_mode(metadata) != 'compact':
                validate_archive(source,metadata)
            if get_storage_mode(prior) != 'compact':
                validate_archive(target.with_suffix('.npz'),prior)
            return prior['records']
    if get_storage_mode(metadata) == 'compact' and wanted <= {r['method'] for r in metadata['records']}:
        # New compact primary trials already evaluate these comparators. Reuse
        # their saved measurements; do not pretend arrays were reconstructed.
        return [dict(r) for r in metadata['records'] if r['method'] in wanted]
    raw_fit_check = 'unavailable: raw observations not retained'
    with load_archive(source,metadata,required_keys=('scores_cal','scores_test','base_lengths_test')) as saved:
        if get_storage_mode(metadata) == 'full':
            for split in ['train', 'cal', 'test']:
                assert len(saved['X_'+split]) == len(saved['y_'+split]) > 0
            raw_fit_check = 'raw split dimensions checked'
        if toy and get_storage_mode(metadata) == 'full':
            fitted = saved['X_cal'] @ saved['model_coef'].T + saved['model_intercept']
            np.testing.assert_allclose(abs(saved['y_cal']-fitted)-saved['base_lengths_cal']/2,
                                       saved['scores_cal'], rtol=1e-11, atol=1e-11)
            raw_fit_check = 'raw fitted calibration scores recomputed'
        cal, test, lengths = saved['scores_cal'], saved['scores_test'], saved['base_lengths_test']
        records, arrays = evaluate_raw_cqr_baselines(cal, test, lengths, alpha)
        if toy:
            capped = np.maximum(cal, 0)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                for method in ['Envelope', 'TSCP_R', 'TSCP_GWC']:
                    start = time.perf_counter()
                    region = (envelope_prediction(capped, alpha) if method == 'Envelope' else
                              standardized_prediction(capped, alpha, method='GWC' if method == 'TSCP_GWC' else 'LWC'))
                    key = method+'_capped_0'
                    arrays[key] = region.upper
                    records.append(metric(key, region.upper, test, lengths, time.perf_counter()-start))
                    records[-1]['fallback'] = getattr(region, 'fallback', '')
    for row in records:
        row.update(trial=trial, redraw_train_test=True,
                   source_archive=str(source.relative_to(ROOT)), source_sha256=checksum,
                   protocol='paired comparison on fresh-per-trial training/calibration/test fit')
        if toy:
            row.update(study=source.parent.name, n_train=1000)
        else:
            primary = metadata['records'][0]
            for key in ['config_id', 'alpha', 'n_cal', 'n_dim', 'fit_seconds', 'n_train',
                        'training_seed', 'calibration_seed', 'split_seed']:
                row[key] = primary[key]
    archive = target.with_suffix('.npz')
    np.savez_compressed(archive, **arrays)
    dump(target, dict(version=2, source_sha256=checksum, records=records,
                      source_storage_mode=get_storage_mode(metadata),raw_fit_check=raw_fit_check,
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    return records


def main():
    inventory = json.loads((ROOT/'envelope_method/settings.json').read_text())
    inventory += json.loads((ROOT/'envelope_method/notebook_settings.json').read_text())
    summary = []
    for item in inventory:
        if item['config']['kind'] != 'cqr':
            continue
        rows = []
        for trial in range(item['trials']):
            rows += complete_trial(OUT/'cqr'/item['id']/f'trial_{trial:03d}.npz',
                                   OUT/'cqr_baselines'/item['id'], trial, item['config']['alpha'])
        folder = OUT/'cqr_baselines'/item['id']
        pd.DataFrame(rows).to_csv(folder/'trials.csv', index=False)
        dump(folder/'status.json', dict(status='complete', trials=item['trials']))
        summary.append(dict(config_id=item['id'], trials=item['trials'], family='cqr_baselines'))
        print('CQR comparators complete', item['id'], item['trials'], flush=True)
    for folder in sorted((OUT/'toys').iterdir()):
        if not (folder/'config.json').exists():
            continue
        config = json.loads((folder/'config.json').read_text())
        if 'base_alpha' not in config:
            continue
        rows = []
        for trial in range(config['trials']):
            rows += complete_trial(folder/f'trial_{trial:04d}.npz', OUT/'toy_baselines'/folder.name,
                                   trial, config.get('alpha', .1), toy=True)
        dest = OUT/'toy_baselines'/folder.name
        pd.DataFrame(rows).to_csv(dest/'trials.csv', index=False)
        dump(dest/'status.json', dict(status='complete', trials=config['trials']))
        summary.append(dict(study=folder.name, trials=config['trials'], family='toy_baselines'))
        print('Toy comparators complete', folder.name, config['trials'], flush=True)
    dump(OUT/'cqr_baseline_completion.json', dict(status='complete', workloads=summary,
        note='Missing methods evaluated on each previously generated fresh trial and its fitted model. No fixed observations across trials; no full LWC run.'))


if __name__ == '__main__':
    main()
