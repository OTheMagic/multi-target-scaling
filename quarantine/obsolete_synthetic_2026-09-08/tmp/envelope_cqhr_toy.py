import json
import math
from time import perf_counter

import numpy as np


ALPHA = 0.1
Z90_NORMAL = 1.6448536269514722


def conformal_rank(n, alpha):
    return math.ceil((1.0 - alpha) * (n + 1))


def conformal_quantile(values, alpha):
    values = np.asarray(values, dtype=float).reshape(-1)
    rank = conformal_rank(values.size, alpha)
    if rank > values.size:
        return np.inf
    return np.partition(values, rank - 1)[rank - 1]


def score_at_z(t, mean, std, n, z):
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    out_shape = np.broadcast_shapes(t.shape, z.shape)
    if np.all(np.isposinf(z)):
        return np.full(out_shape, -1.0 / math.sqrt(n + 1.0))
    mu = (n * mean + z) / (n + 1.0)
    sigma = np.sqrt(((z - mean) ** 2) / (n + 1.0) + std**2)
    return np.broadcast_to((t - mu) / sigma, out_shape)


def sup_score_interval(t, mean, std, n, lo, hi):
    t = np.asarray(t, dtype=float)
    vals = score_at_z(t, mean, std, n, lo)
    vals = np.maximum(vals, score_at_z(t, mean, std, n, hi))

    above_mean = t > mean
    if np.any(above_mean):
        z_star = mean - std**2 / (t[above_mean] - mean)
        in_interval = z_star >= lo
        if not np.isposinf(hi):
            in_interval &= z_star <= hi
        if np.any(in_interval):
            idx = np.flatnonzero(above_mean)[in_interval]
            vals[idx] = np.maximum(
                vals[idx],
                score_at_z(t[idx], mean, std, n, z_star[in_interval]),
            )
    return vals


def omega(c, mean, std, n):
    if np.isposinf(c):
        return np.inf
    limit = n / math.sqrt(n + 1.0)
    if c >= limit:
        return np.inf
    if c <= -limit:
        return -np.inf
    denom = n * n - (n + 1.0) * c * c
    return mean + std * c * (n + 1.0) / math.sqrt(max(denom, 1e-300))


def gwc_signed(cal_scores, lower, alpha):
    n, d = cal_scores.shape
    means = cal_scores.mean(axis=0)
    stds = cal_scores.std(axis=0)
    coord_scores = [
        sup_score_interval(cal_scores[:, j], means[j], stds[j], n, lower[j], np.inf)
        for j in range(d)
    ]
    global_scores = np.max(np.column_stack(coord_scores), axis=1)
    q = conformal_quantile(global_scores, alpha)
    upper = np.array([omega(q, means[j], stds[j], n) for j in range(d)])
    return q, upper, means, stds


def envelope_signed_bounds(cal_scores, base_lengths_test_row, alpha, return_sequences=False):
    n, d = cal_scores.shape
    lower = -0.5 * np.asarray(base_lengths_test_row, dtype=float)
    q_gwc, upper_gwc, means, stds = gwc_signed(cal_scores, lower, alpha)
    upper_gwc = np.maximum(upper_gwc, lower)

    bounds = lower.copy()
    sequences = []
    gwc_coord_scores = [
        sup_score_interval(cal_scores[:, j], means[j], stds[j], n, lower[j], upper_gwc[j])
        for j in range(d)
    ]

    for j in range(d):
        if d == 1:
            off_scores = np.full(n, -np.inf)
        else:
            off_scores = np.max(
                np.column_stack([gwc_coord_scores[ell] for ell in range(d) if ell != j]),
                axis=1,
            )

        finite_upper = upper_gwc[j]
        points = [lower[j]]
        if np.isfinite(finite_upper):
            inside = cal_scores[:, j][
                (cal_scores[:, j] > lower[j]) & (cal_scores[:, j] < finite_upper)
            ]
            points.extend(np.unique(inside))
            points.append(finite_upper)
        else:
            inside = cal_scores[:, j][cal_scores[:, j] > lower[j]]
            points.extend(np.unique(inside))
            points.append(np.inf)
        points = np.array(sorted(points), dtype=float)

        seq = []
        for lo, hi in zip(points[:-1], points[1:]):
            if not hi > lo:
                continue
            active_scores = sup_score_interval(cal_scores[:, j], means[j], stds[j], n, lo, hi)
            surface_scores = np.maximum(active_scores, off_scores)
            q_surface = conformal_quantile(surface_scores, alpha)
            linked = omega(q_surface, means[j], stds[j], n)
            if linked > lo:
                candidate = min(hi, linked)
                bounds[j] = max(bounds[j], candidate)
            else:
                candidate = lower[j]
            seq.append(
                {
                    "lo": float(lo),
                    "hi": float(hi) if np.isfinite(hi) else "inf",
                    "q": float(q_surface) if np.isfinite(q_surface) else "inf",
                    "linked": float(linked) if np.isfinite(linked) else "inf",
                    "B": float(candidate) if np.isfinite(candidate) else "inf",
                    "nonzero": bool(linked > lo),
                }
            )
        sequences.append(seq)

    if return_sequences:
        return bounds, upper_gwc, q_gwc, sequences
    return bounds, upper_gwc, q_gwc


def cqhr_bounds(cal_scores, base_lengths_cal, base_lengths_test, alpha, reference_dim=0):
    safe_cal_lengths = np.maximum(base_lengths_cal, 1e-12)
    safe_test_lengths = np.maximum(base_lengths_test, 1e-12)
    reference_cal = safe_cal_lengths[:, [reference_dim]]
    converted = cal_scores * reference_cal / safe_cal_lengths
    calibration_scores = np.max(converted, axis=1)
    q = conformal_quantile(calibration_scores, alpha)
    reference_test = safe_test_lengths[:, [reference_dim]]
    return q * safe_test_lengths / reference_test, q


def equicorrelated_normals(rng, n, d, rho):
    if abs(rho) < 1e-15:
        return rng.normal(size=(n, d))
    common = rng.normal(size=(n, 1))
    independent = rng.normal(size=(n, d))
    return math.sqrt(rho) * common + math.sqrt(1.0 - rho) * independent


def draw_scales(rng, n, d, scenario):
    base = np.asarray(scenario.get("base_scales", [1.0] * d), dtype=float)
    if scenario["scale_kind"] == "constant":
        return np.tile(base, (n, 1))
    if scenario["scale_kind"] == "lognormal_independent":
        sd = scenario.get("scale_sd", 0.6)
        z = rng.normal(size=(n, d))
        return base * np.exp(sd * z - 0.5 * sd * sd)
    if scenario["scale_kind"] == "lognormal_common":
        sd = scenario.get("scale_sd", 0.6)
        z = rng.normal(size=(n, 1))
        return base * np.exp(sd * z - 0.5 * sd * sd)
    raise ValueError(f"unknown scale_kind {scenario['scale_kind']}")


def draw_residual_sample(rng, n, d, scenario):
    scales = draw_scales(rng, n, d, scenario)
    eps = equicorrelated_normals(rng, n, d, scenario.get("rho", 0.0))
    residuals = scales * (np.abs(eps) - Z90_NORMAL)
    base_lengths = 2.0 * Z90_NORMAL * scales
    return residuals, base_lengths


def summarize(x):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "se": float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0,
    }


def boundary_diagnostics(sequences):
    total = 0
    zero_suffix_ok = 0
    nondecreasing_ok = 0
    violations = []
    for split_idx, split_sequences in enumerate(sequences):
        for j, seq in enumerate(split_sequences):
            if not seq:
                continue
            total += 1
            nonzero = np.array([item["nonzero"] for item in seq], dtype=bool)
            b = np.array([item["B"] for item in seq], dtype=float)
            first_zero = np.flatnonzero(~nonzero)
            suffix_ok = True
            if first_zero.size:
                suffix_ok = not np.any(nonzero[first_zero[0] :])
            zero_suffix_ok += int(suffix_ok)
            mono_ok = bool(np.all(np.diff(b) >= -1e-10))
            nondecreasing_ok += int(mono_ok)
            if (not suffix_ok or not mono_ok) and len(violations) < 5:
                violations.append(
                    {
                        "split": split_idx,
                        "coord": j,
                        "suffix_ok": bool(suffix_ok),
                        "nondecreasing_ok": bool(mono_ok),
                        "B_prefix": b[:12].round(6).tolist(),
                        "nonzero_prefix": nonzero[:12].tolist(),
                    }
                )
    return {
        "coordinate_sequences": total,
        "zero_suffix_fraction": zero_suffix_ok / total if total else None,
        "nondecreasing_fraction": nondecreasing_ok / total if total else None,
        "violations": violations,
    }


def run_scenario(scenario, n_cal=80, n_test=120, n_splits=50, alpha=ALPHA):
    n_cal = scenario.get("n_cal", n_cal)
    n_test = scenario.get("n_test", n_test)
    n_splits = scenario.get("n_splits", n_splits)
    rng = np.random.default_rng(scenario["seed"])
    d = scenario["d"]
    records = []
    diagnostic_sequences = []
    t0 = perf_counter()

    for split in range(n_splits):
        cal_scores, cal_lengths = draw_residual_sample(rng, n_cal, d, scenario)
        test_scores, test_lengths = draw_residual_sample(rng, n_test, d, scenario)

        cqhr_adj, cqhr_q = cqhr_bounds(cal_scores, cal_lengths, test_lengths, alpha)
        cqhr_cover = np.all(test_scores <= cqhr_adj, axis=1)
        cqhr_lengths = np.maximum(test_lengths + 2.0 * cqhr_adj, 0.0)
        cqhr_volume = np.prod(cqhr_lengths, axis=1)

        env_bounds = np.empty_like(test_scores)
        gwc_bounds = np.empty_like(test_scores)
        split_sequences = None
        envelope_cache = {}
        for i in range(n_test):
            want_seq = split < scenario.get("diagnostic_splits", 20) and i == 0
            cache_key = tuple(np.round(test_lengths[i], 10))
            if want_seq:
                env, gwc, _, seq = envelope_signed_bounds(
                    cal_scores, test_lengths[i], alpha, return_sequences=True
                )
                split_sequences = seq
                envelope_cache[cache_key] = (env, gwc)
            elif cache_key in envelope_cache:
                env, gwc = envelope_cache[cache_key]
            else:
                env, gwc, _ = envelope_signed_bounds(cal_scores, test_lengths[i], alpha)
                envelope_cache[cache_key] = (env, gwc)
            env_bounds[i] = env
            gwc_bounds[i] = gwc
        if split_sequences is not None:
            diagnostic_sequences.append(split_sequences)

        env_cover = np.all(test_scores <= env_bounds, axis=1)
        env_lengths = np.maximum(test_lengths + 2.0 * env_bounds, 0.0)
        env_volume = np.prod(env_lengths, axis=1)

        gwc_cover = np.all(test_scores <= gwc_bounds, axis=1)
        gwc_lengths = np.maximum(test_lengths + 2.0 * gwc_bounds, 0.0)
        gwc_volume = np.prod(gwc_lengths, axis=1)

        base_cover = np.all(test_scores <= 0.0, axis=1)
        base_volume = np.prod(test_lengths, axis=1)

        records.append(
            {
                "base_coverage": float(np.mean(base_cover)),
                "base_volume": float(np.mean(base_volume)),
                "cqhr_coverage": float(np.mean(cqhr_cover)),
                "cqhr_volume": float(np.mean(cqhr_volume)),
                "cqhr_q": float(cqhr_q),
                "env_coverage": float(np.mean(env_cover)),
                "env_volume": float(np.mean(env_volume)),
                "gwc_coverage": float(np.mean(gwc_cover)),
                "gwc_volume": float(np.mean(gwc_volume)),
                "env_vs_cqhr_volume_ratio": float(np.mean(env_volume) / np.mean(cqhr_volume)),
                "env_better_fraction": float(np.mean(env_volume < cqhr_volume)),
                "env_contained_in_cqhr_fraction": float(np.mean(np.all(env_bounds <= cqhr_adj, axis=1))),
                "cqhr_contained_in_env_fraction": float(np.mean(np.all(cqhr_adj <= env_bounds, axis=1))),
                "env_mean_adjustment": float(np.mean(env_bounds)),
                "cqhr_mean_adjustment": float(np.mean(cqhr_adj)),
            }
        )

    keys = records[0].keys()
    summary = {key: summarize([record[key] for record in records]) for key in keys}
    summary["elapsed_seconds"] = perf_counter() - t0
    summary["boundary_diagnostics"] = boundary_diagnostics(diagnostic_sequences)
    return summary


def main():
    scenarios = [
        {
            "name": "2d_homoskedastic_independent",
            "d": 2,
            "scale_kind": "constant",
            "base_scales": [1.0, 1.0],
            "rho": 0.0,
            "seed": 2026090401,
        },
        {
            "name": "2d_constant_heterogeneous",
            "d": 2,
            "scale_kind": "constant",
            "base_scales": [1.0, 3.0],
            "rho": 0.0,
            "seed": 2026090402,
        },
        {
            "name": "2d_lognormal_independent_scales",
            "d": 2,
            "scale_kind": "lognormal_independent",
            "base_scales": [1.0, 3.0],
            "scale_sd": 0.75,
            "rho": 0.0,
            "seed": 2026090403,
        },
        {
            "name": "2d_lognormal_common_scale",
            "d": 2,
            "scale_kind": "lognormal_common",
            "base_scales": [1.0, 3.0],
            "scale_sd": 0.75,
            "rho": 0.0,
            "seed": 2026090404,
        },
        {
            "name": "3d_constant_heterogeneous",
            "d": 3,
            "scale_kind": "constant",
            "base_scales": [1.0, 2.0, 4.0],
            "rho": 0.0,
            "seed": 2026090405,
        },
        {
            "name": "3d_correlated_homoskedastic",
            "d": 3,
            "scale_kind": "constant",
            "base_scales": [1.0, 1.0, 1.0],
            "rho": 0.75,
            "seed": 2026090406,
        },
    ]

    all_results = {}
    for scenario in scenarios:
        print(f"running {scenario['name']} ...", flush=True)
        result = run_scenario(scenario)
        all_results[scenario["name"]] = result
        compact = {
            key: result[key]
            for key in [
                "base_coverage",
                "cqhr_coverage",
                "env_coverage",
                "gwc_coverage",
                "base_volume",
                "cqhr_volume",
                "env_volume",
                "gwc_volume",
                "env_vs_cqhr_volume_ratio",
                "env_better_fraction",
                "env_contained_in_cqhr_fraction",
                "cqhr_contained_in_env_fraction",
            ]
        }
        print(json.dumps(compact, indent=2), flush=True)

    with open("tmp/envelope_cqhr_toy_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("wrote tmp/envelope_cqhr_toy_summary.json")


if __name__ == "__main__":
    main()
