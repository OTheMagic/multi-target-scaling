"""Refresh capped-envelope results affected by exact-zero inverse cancellation."""
import hashlib
import json
import shutil
import time
import numpy as np
import pandas as pd
from experiments import ROOT, OUT, dump, metric
from utility.envelope import envelope_prediction
from archive_storage import require_storage


def main():
    inventory = json.loads((ROOT/'envelope_method/settings.json').read_text())
    inventory += json.loads((ROOT/'envelope_method/notebook_settings.json').read_text())
    review = ROOT/'quarantine/obsolete_synthetic_2026-09-08/numerical_boundary_2026-09-09'
    changes = []
    count = 0
    for item in inventory:
        if item['config']['kind'] != 'cqr':
            continue
        folder = OUT/'cqr'/item['id']
        records = []
        for trial in range(item['trials']):
            metadata_path = folder/f'trial_{trial:03d}.json'
            archive = metadata_path.with_suffix('.npz')
            metadata = json.loads(metadata_path.read_text())
            # This historical, scientific-value-changing repair predates tiered
            # storage. Never rewrite a retained checkpoint with stale lineage.
            require_storage(metadata,'full')
            if 'storage' in metadata:
                raise ValueError('This historical one-time boundary repair does not rewrite tiered checkpoints. Use a separately versioned repair preserving archive and record lineage.')
            rows = metadata['records']
            with np.load(archive) as saved:
                cal, test, lengths = saved['scores_cal'], saved['scores_test'], saved['base_lengths_test']
                start = time.perf_counter()
                reg = envelope_prediction(np.maximum(cal, 0), item['config']['alpha'])
                seconds = time.perf_counter()-start
                old = saved['Envelope_capped_0']
                if not np.allclose(reg.upper, old, rtol=1e-12, atol=1e-12, equal_nan=True):
                    # Keep the pre-correction numerical record in the same review
                    # folder; it is fresh-protocol data, not mislabeled pool reuse.
                    for path in [metadata_path, archive]:
                        destination = review/path.relative_to(ROOT)
                        assert destination.resolve().is_relative_to(review.resolve())
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if destination.exists():
                            raise FileExistsError(destination)
                        shutil.copy2(path, destination)
                    arrays = {name: saved[name] for name in saved.files}
                    arrays['Envelope_capped_0'] = reg.upper
                    replacement = metric('Envelope_capped_0', reg.upper, test, lengths, seconds,
                                         np.any(lengths + 2*reg.upper < 0, axis=1))
                    replacement['fallback'] = reg.fallback
                    for row in rows:
                        if row['method'] == 'Envelope_capped_0':
                            row.update(replacement)
                    old_hash = metadata['archive_sha256']
                    changes.append(dict(archive=str(archive.relative_to(ROOT)), old_sha256=old_hash,
                                        old_upper=old.tolist(), corrected_upper=reg.upper.tolist(),
                                        reason='Exact accepted zero lost through inverse-link cancellation. This is not reused-pool data.'))
                else:
                    arrays = None
            if arrays is not None:
                np.savez_compressed(archive, **arrays)
                metadata.update(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                boundary_correction=dict(date='2026-09-09', previous_archive_sha256=old_hash))
                dump(metadata_path, metadata)
                changes[-1]['new_sha256'] = metadata['archive_sha256']
            records += rows
            count += 1
        frame = pd.DataFrame(records)
        frame.to_csv(folder/'trials.csv', index=False)
        frame.groupby('method')[['test_coverage','outcome_volume','mean_log_volume','runtime','empty_rate']].agg(
            ['mean','std','count']).to_csv(folder/'summary.csv')
    report = dict(status='complete', checked_cqr_trials=count, changed_trials=len(changes), changes=changes,
                  note='Only materially changed capped-envelope bounds replaced; all raw observations and all other method outputs preserved.')
    dump(OUT/'zero_boundary_corrections.json', report)
    if changes:
        dump(review/'manifest.json', report)
    print('Checked', count, 'CQR trials; corrected', len(changes), flush=True)


if __name__ == '__main__':
    main()
