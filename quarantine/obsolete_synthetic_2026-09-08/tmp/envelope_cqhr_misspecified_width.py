import json
import math
import sys

import numpy as np

sys.path.insert(0, "tmp")
from envelope_cqhr_toy import cqhr_bounds, envelope_signed_bounds  # noqa: E402


def draw_sample(rng, n, d, sd=0.9, tail_scales=None):
    if tail_scales is None:
        tail_scales = np.ones(d)
    tail_scales = np.asarray(tail_scales, dtype=float)
    central_width = np.exp(sd * rng.normal(size=(n, d)) - 0.5 * sd * sd)
    base_lengths = 2.0 * central_width
    inside = rng.random((n, d)) < 0.9
    negative_scores = -central_width * rng.random((n, d))
    positive_scores = rng.exponential(scale=tail_scales, size=(n, d))
    scores = np.where(inside, negative_scores, positive_scores)
    return scores, base_lengths


def conformal_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "se": float(np.std(values, ddof=1) / math.sqrt(len(values))),
    }


def run(seed=2026091601, n_splits=80, n_cal=80, n_test=160, d=2):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_splits):
        cal_scores, cal_lengths = draw_sample(rng, n_cal, d, tail_scales=[0.8, 1.4][:d])
        test_scores, test_lengths = draw_sample(rng, n_test, d, tail_scales=[0.8, 1.4][:d])

        cqhr_adj, _ = cqhr_bounds(cal_scores, cal_lengths, test_lengths, 0.1)
        env_bounds = np.empty_like(test_scores)
        gwc_bounds = np.empty_like(test_scores)
        for i in range(n_test):
            env, gwc, _ = envelope_signed_bounds(cal_scores, test_lengths[i], 0.1)
            env_bounds[i] = env
            gwc_bounds[i] = gwc

        volumes = {}
        for name, adj in [("cqhr", cqhr_adj), ("env", env_bounds), ("gwc", gwc_bounds)]:
            lengths = np.maximum(test_lengths + 2.0 * adj, 0.0)
            volumes[name] = np.prod(lengths, axis=1)

        records.append(
            {
                "base_coverage": float(np.mean(np.all(test_scores <= 0.0, axis=1))),
                "cqhr_coverage": float(np.mean(np.all(test_scores <= cqhr_adj, axis=1))),
                "env_coverage": float(np.mean(np.all(test_scores <= env_bounds, axis=1))),
                "gwc_coverage": float(np.mean(np.all(test_scores <= gwc_bounds, axis=1))),
                "base_volume": float(np.mean(np.prod(test_lengths, axis=1))),
                "cqhr_volume": float(np.mean(volumes["cqhr"])),
                "env_volume": float(np.mean(volumes["env"])),
                "gwc_volume": float(np.mean(volumes["gwc"])),
                "env_vs_cqhr_volume_ratio": float(
                    np.mean(volumes["env"]) / np.mean(volumes["cqhr"])
                ),
                "env_better_fraction": float(np.mean(volumes["env"] < volumes["cqhr"])),
            }
        )
    return {key: conformal_summary([r[key] for r in records]) for key in records[0]}


def main():
    result = run()
    print(json.dumps(result, indent=2))
    with open(
        "tmp/envelope_cqhr_misspecified_width_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
