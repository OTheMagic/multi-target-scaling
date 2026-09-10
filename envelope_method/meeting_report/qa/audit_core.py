"""Small independent algebra/containment audit; no full-LWC enumeration.

Uses only the bundled code and writes its evidence beside this script.
Run: python qa/audit_core.py from the report directory (or any cwd).
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'code'))
from report_paths import artifact, data_path
from utility.envelope import EnvelopeCalibration, interval_sup, link, score_at


def main():
    rng = np.random.default_rng(2026091003)
    counts = dict(augmented_standardization=0, signed_inverse=0,
                  interval_grid_suprema=0, translated_domains=0,
                  candidate_acceptances_checked=0, accepted_candidates=0,
                  search_agreements=0)
    for n in (8, 30, 80):
        for rep in range(8):
            cal = rng.normal(size=(n, 3)) * np.array([.2, 1., 4.])
            m, s = cal.mean(0), cal.std(0)
            z = rng.normal(size=3) * 3
            augmented = np.vstack((cal, z))
            direct = ((cal - augmented.mean(0)) /
                      augmented.std(0, ddof=1))
            np.testing.assert_allclose(score_at(cal, m, s, n, z), direct,
                                       rtol=2e-13, atol=2e-13)
            counts['augmented_standardization'] += 1
            for q in (-.95, -.2, 0., .4, .95):
                target = q * n / np.sqrt(n + 1)
                inv = link(target, m, s, n)
                np.testing.assert_allclose(score_at(inv, m, s, n, inv),
                                           target, rtol=2e-13, atol=2e-13)
                counts['signed_inverse'] += 1
            for j in range(3):
                lo, hi = sorted(rng.normal(size=2) * 3)
                t = cal[:, j]
                exact = interval_sup(t, m[j], s[j], n, lo, hi)
                grid = np.linspace(lo, hi, 2001)
                # Independent direct augmented-statistic expression.
                values = ((t[:, None] - m[j] - (grid - m[j]) / (n+1)) /
                          np.sqrt(s[j]**2 + (grid-m[j])**2 / (n+1)))
                assert np.all(exact + 2e-12 >= values.max(1))
                # Evaluate the derived stationary point as an independent
                # analytic candidate, not merely a dense-grid lower bound.
                best = np.maximum(values[:, 0], values[:, -1])
                for i, ti in enumerate(t):
                    if ti > m[j]:
                        critical = m[j] - s[j]**2 / (ti-m[j])
                        if lo <= critical <= hi:
                            v = ((ti-m[j]-(critical-m[j])/(n+1)) /
                                 np.sqrt(s[j]**2+(critical-m[j])**2/(n+1)))
                            best[i] = max(best[i], v)
                np.testing.assert_allclose(exact, best, rtol=2e-13, atol=2e-13)
                counts['interval_grid_suprema'] += 1
            lower = np.array([-1.5, -2., -6.])
            alpha = (.1, .3, .7, .95)[rep % 4]
            model = EnvelopeCalibration(cal, alpha)
            base = model.predict(lower)
            shift = rng.uniform(-4, 4, size=3)
            moved = EnvelopeCalibration(cal+shift, alpha).predict(lower+shift)
            assert moved.empty == base.empty
            np.testing.assert_allclose(moved.upper-shift, base.upper,
                                       atol=2e-11, rtol=2e-11)
            counts['translated_domains'] += 1
            for mode in ('rank', 'exhaustive'):
                check = model.predict(lower, search=mode)
                assert check.empty == base.empty
                np.testing.assert_allclose(check.upper, base.upper,
                                           atol=2e-11, rtol=2e-11)
                counts['search_agreements'] += 1
            candidates = lower + rng.exponential(4., size=(120, 3))
            rank = int(np.ceil((n+1)*(1-alpha)))
            for candidate in candidates:
                a = np.vstack((cal, candidate))
                standardized = (a-a.mean(0))/a.std(0, ddof=1)
                scores = standardized.max(1)
                threshold = np.inf if rank > n else np.sort(scores[:-1])[rank-1]
                accepted = scores[-1] <= threshold
                counts['candidate_acceptances_checked'] += 1
                if accepted:
                    assert base.contains(candidate)
                    counts['accepted_candidates'] += 1
    # Closed zero-width domains are valid points; impossible domains are empty.
    for qsign in (-1, 1):
        cutoff = qsign * 10 / np.sqrt(11)
        expected = qsign * np.inf
        assert np.all(link(cutoff, np.zeros(2), np.ones(2), 10) == expected)
    report = dict(status='passed', checks=counts,
                  scope='Independent finite algebra, search, and full-conformal inclusion checks; not an exhaustive proof or coverage simulation.')
    (data_path('qa/core_audit.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
