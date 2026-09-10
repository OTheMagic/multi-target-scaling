"""Audit direct rank localization against backward search on saved formula cases."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from utility.envelope import EnvelopeCalibration


def main():
    records = []
    for path in (ROOT / 'data/envelope_method/results/toys/boundary').glob('*.npz'):
        with np.load(path) as data:
            backwards, ranked, examples = 0, 0, 0
            for cal, lower in zip(data['calibration'], data['lower']):
                model = EnvelopeCalibration(cal, .1)
                b, r = model.predict(lower), model.predict(lower, search='rank')
                np.testing.assert_allclose(b.upper, r.upper, rtol=1e-12, atol=1e-12)
                assert b.empty == r.empty
                backwards += sum(b.evaluations)
                ranked += sum(r.evaluations)
                examples += 1
        records.append(dict(study=path.stem, samples=examples,
                            backward_surface_quantiles=backwards, rank_surface_quantiles=ranked))
    report = dict(status='passed', studies=records, samples=sum(r['samples'] for r in records),
                  note='Rank localization also uses one auxiliary order statistic per coordinate; counts are not runtime.')
    (ROOT / 'data/envelope_method/rank_search_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
