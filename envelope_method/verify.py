"""Independent numerical audits of the formulas and set containments."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import minimize_scalar
from utility.envelope import EnvelopeCalibration, envelope_prediction, interval_sup, link, score_at
from utility.conformal_utils import conformal_quantile
from utility.res_rescaled import standardized_prediction


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(20260908)
    counts = dict(augmentation=0, inverse=0, interval_optimization=0,
                  oracle_candidates=0, accepted_oracle_candidates=0,
                  search_agreement=0, nonnegative_dominance=0,
                  prefix_sequences=0, prefix_violations=0)
    witness = None
    for _ in range(150):
        n = int(rng.integers(5, 60))
        t = rng.normal(size=n) * np.exp(rng.normal()) + rng.normal()
        m, s = t.mean(), t.std()
        z = rng.normal() * 10
        aug = np.r_[t, z]
        np.testing.assert_allclose(score_at(t, m, s, n, z),
                                   (t - aug.mean()) / aug.std(ddof=1), rtol=1e-12, atol=1e-12)
        counts['augmentation'] += 1
        q = score_at(z, m, s, n, z)
        np.testing.assert_allclose(link(q, m, s, n), z, rtol=2e-10, atol=2e-10)
        counts['inverse'] += 1
        for _ in range(8):
            a, b = np.sort(rng.normal(size=2) * 5)
            v = float(rng.choice(t))
            fit = minimize_scalar(lambda zz: -float(score_at(v, m, s, n, zz)),
                                  bounds=(a, b), method='bounded')
            numeric = max(float(score_at(v, m, s, n, a)),
                          float(score_at(v, m, s, n, b)), -fit.fun)
            np.testing.assert_allclose(interval_sup(v, m, s, n, a, b), numeric,
                                       rtol=2e-8, atol=2e-8)
            counts['interval_optimization'] += 1
    for case in range(400):
        n, d = int(rng.integers(5, 55)), int(rng.integers(1, 5))
        alpha = float(rng.choice([0.05, 0.1, 0.3, 0.5, 0.7, 0.9]))
        cal = rng.normal(size=(n, d)) * np.exp(rng.normal(size=d)) + rng.normal(size=d)
        if case % 3 == 0:
            cal = np.round(cal, 1)
        lower = np.full(d, -np.inf) if case % 2 else -np.exp(rng.normal(size=d))
        model = EnvelopeCalibration(cal, alpha)
        fast = model.predict(lower)
        full, cells = model.predict(lower, search='exhaustive', return_cells=True)
        ranked = model.predict(lower, search='rank')
        np.testing.assert_allclose(ranked.upper, full.upper, rtol=1e-12, atol=1e-12)
        assert ranked.empty == full.empty
        np.testing.assert_array_equal(fast.upper, full.upper)
        assert fast.empty == full.empty
        counts['search_agreement'] += 1
        for seq in cells:
            flags = np.array([c['survives'] for c in seq])
            violation = bool(np.any(np.diff(flags.astype(int)) > 0))
            counts['prefix_sequences'] += 1
            counts['prefix_violations'] += int(violation)
            if violation and witness is None:
                witness = dict(cal=cal.tolist(), alpha=alpha, lower=lower.tolist(), sequence=seq)
        candidates = rng.normal(size=(120, d)) * 4
        candidates = candidates[np.all(candidates >= lower, axis=1)]
        for candidate in candidates:
            aug = np.vstack([cal, candidate])
            sigma = aug.std(axis=0, ddof=1)
            sigma = np.where(sigma == 0, 1.0, sigma)
            scores = ((aug - aug.mean(axis=0)) / sigma).max(axis=1)
            accepted = scores[-1] <= conformal_quantile(scores[:-1], alpha)
            counts['oracle_candidates'] += 1
            if accepted:
                counts['accepted_oracle_candidates'] += 1
                assert fast.contains(candidate), (cal, alpha, candidate, fast)
    max_excess = -np.inf
    for case in range(600):
        n, d = int(rng.integers(10, 90)), int(rng.integers(1, 9))
        alpha = float(rng.choice([0.05, 0.1, 0.3, 0.5, 0.9]))
        cal = rng.lognormal(0, rng.uniform(0.1, 2.5), size=(n, d)) * np.exp(rng.normal(size=d))
        if case % 4 == 0:
            cal = np.maximum(cal - 0.5, 0)
        env = envelope_prediction(cal, alpha)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            old = standardized_prediction(cal, alpha)
        if not env.empty and not env.fallback:
            finite = np.isfinite(old.upper)
            excess = (env.upper[finite] - old.upper[finite]) / (1 + np.abs(old.upper[finite]))
            if excess.size:
                max_excess = max(max_excess, float(excess.max()))
            assert np.all(env.upper <= old.upper + 1e-8 * (1 + np.abs(old.upper))), (cal, alpha, env.upper, old.upper)
        counts['nonnegative_dominance'] += 1
    ties = np.array([[0., 1.], [0., 2.], [1., 2.], [1., 3.], [2., 3.]])
    assert np.isposinf(envelope_prediction(ties, 0.01).upper).all()
    assert envelope_prediction(np.ones((20, 2))).fallback == 'zero_calibration_scale'
    # A high-miscoverage negative domain can legitimately yield an empty set.
    empty = envelope_prediction(np.arange(20.)[:, None] - 100, .9, lower=0)
    assert empty.empty and empty.volume() == 0 and not empty.contains([0])
    report = dict(status='passed', checks=counts, max_scaled_excess_over_old=max_excess,
                  seconds=time.perf_counter()-start, seed=20260908,
                  prefix_counterexample=witness,
                  note='Finite numerical audits support the proofs; they are not proofs themselves.')
    (ROOT / 'data/envelope_method/verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'prefix_counterexample'}, indent=2))


if __name__ == '__main__':
    main()
