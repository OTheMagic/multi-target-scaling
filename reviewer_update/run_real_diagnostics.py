"""Reproduce real-data splits, cache residuals, and audit TSCP search branches.

From the repository root:
  python reviewer_update/run_real_diagnostics.py prepare
  python reviewer_update/run_real_diagnostics.py fit --trials 200 --workers 4
  python reviewer_update/run_real_diagnostics.py evaluate --trials 200

Fitting and construction timing are separate phases to avoid timing under the
load of concurrent model fits. Cached residuals permit exact diagnostic replay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from utility.exps import _load_real_experiment_data, _stable_hash, _function_choice
from utility.res_rescaled import standardized_prediction

DATASETS = ["stock", "rf2", "scm1d", "scm20d", "energy", "student"]
METHODS = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
EXPECTED_SHAPES = {"stock": (15, 6), "rf2": (384, 8), "scm1d": (490, 16),
                   "scm20d": (448, 16), "energy": (38, 2), "student": (32, 3)}
OUT = ROOT / "reviewer_update/real_diagnostics"
CACHE = OUT / "cache"
TABLES = OUT / "data"
ALPHA = 0.1
TIMING_REPEATS = 7
STRESS_ALPHA = [0.1, 0.3, 0.5, 0.7, 0.9]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def versions():
    return {name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "ucimlrepo"]}


def fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def prepare(datasets):
    CACHE.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for name in datasets:
        path = CACHE / f"{name}.pkl"
        if path.exists():
            print(f"Prepared data already cached: {name}", flush=True)
            continue
        start = time.perf_counter()
        X, y, model = _load_real_experiment_data(name)
        assert X.index.equals(y.index)
        assert np.isfinite(y.to_numpy(dtype=float)).all()
        metadata = dict(dataset=name, rows=len(X), features=X.shape[1], outputs=y.shape[1],
                        target_names=list(y.columns), feature_hash=fingerprint(X),
                        target_hash=fingerprint(y), model_params=model.get_params(deep=True),
                        package_versions=versions())
        with path.open("wb") as handle:
            pickle.dump((X, y, model, metadata), handle, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(CACHE / f"{name}_metadata.json", metadata)
        print(f"Prepared {name}: X={X.shape}, y={y.shape}; {time.perf_counter()-start:.2f}s", flush=True)


def fit_one(name, trial):
    destination = CACHE / name / f"split_{trial:03d}.npz"
    if destination.exists():
        return name, trial, 0.0, True
    with (CACHE / f"{name}.pkl").open("rb") as handle:
        X, y, template, metadata = pickle.load(handle)
    seed = _stable_hash(trial)
    train, rest = train_test_split(np.arange(len(X)), test_size=0.25, random_state=seed)
    cal, test = train_test_split(rest, test_size=0.8, random_state=seed)
    assert (len(cal), y.shape[1]) == EXPECTED_SHAPES[name]
    assert not (set(train) & set(cal) or set(cal) & set(test) or set(train) & set(test))
    model = clone(template)
    # Parallelize independent fits; individual fits/predictions stay deterministic.
    model.set_params(**{key: 1 for key in model.get_params() if key.endswith("n_jobs")})
    start = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(X.iloc[train], y.iloc[train])
        scores_cal = np.abs(model.predict(X.iloc[cal]) - y.iloc[cal].to_numpy())
        scores_test = np.abs(model.predict(X.iloc[test]) - y.iloc[test].to_numpy())
    elapsed = time.perf_counter() - start
    assert np.isfinite(scores_cal).all() and np.isfinite(scores_test).all()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, scores_cal=scores_cal, scores_test=scores_test,
                            train_indices=train, cal_indices=cal, test_indices=test,
                            trial=trial, split_seed=seed, fit_seconds=elapsed,
                            target_names=np.asarray(metadata["target_names"], dtype=str),
                            warnings_json=json.dumps([str(w.message) for w in captured]))
    os.replace(temporary, destination)
    return name, trial, elapsed, False


def fit(datasets, trials, workers):
    jobs = [(name, trial) for trial in range(trials) for name in datasets]
    start = time.perf_counter()
    completed = 0
    stream = Parallel(n_jobs=workers, return_as="generator_unordered", pre_dispatch=workers)(
        delayed(fit_one)(name, trial) for name, trial in jobs
    )
    for name, trial, seconds, cached in stream:
        completed += 1
        if not cached or completed == len(jobs):
            print(f"Fit {completed}/{len(jobs)}: {name} split={trial} {seconds:.2f}s"
                  f"{' (cached)' if cached else ''}; elapsed={time.perf_counter()-start:.1f}s", flush=True)


def evaluate_one(scores, alpha, timing_repeats=TIMING_REPEATS):
    diagnostic = {}
    traced = standardized_prediction(scores, alpha, diagnostics=diagnostic)
    assert not np.isnan(traced.upper).any() and np.all(traced.upper >= traced.lower)
    plain = standardized_prediction(scores, alpha)
    np.testing.assert_array_equal(traced.info(), plain.info())
    elapsed = []
    for _ in range(timing_repeats):
        start = time.perf_counter_ns()
        repeated = standardized_prediction(scores, alpha)
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
        np.testing.assert_array_equal(repeated.info(), plain.info())
    branches = diagnostic["coordinate_branch"]
    backward = branches.count("backward")
    category = "fallback" if diagnostic["fallback"] else ("any_backward" if backward else "binary_only")
    record = dict(alpha=alpha, n_cal=len(scores), n_dim=scores.shape[1],
                  fallback=int(diagnostic["fallback"]), any_backward=int(backward > 0),
                  backward_coordinates=backward, binary_coordinates=branches.count("binary"),
                  binary_candidates=sum(diagnostic["binary_evaluations"]),
                  backward_candidates=sum(diagnostic["backward_evaluations"]),
                  regime=category, runtime_ms=float(np.median(elapsed)),
                  zero_scale_coordinates=int(np.count_nonzero(np.std(scores, axis=0) == 0)))
    return traced, record, diagnostic, elapsed


def evaluate(datasets, trials):
    TABLES.mkdir(parents=True, exist_ok=True)
    joint_rows, coordinate_rows, branch_rows, branch_coordinate_rows = [], [], [], []
    stress_rows, timing_rows = [], []
    for name in datasets:
        start = time.perf_counter()
        for trial in range(trials):
            with np.load(CACHE / name / f"split_{trial:03d}.npz", allow_pickle=False) as cached:
                cal, test = cached["scores_cal"], cached["scores_test"]
                targets = cached["target_names"]
                fit_seconds = float(cached["fit_seconds"])
            with threadpool_limits(limits=1), np.errstate(divide="ignore", invalid="ignore"):
                tscp, branch, diagnostic, timings = evaluate_one(cal, ALPHA)
                common = dict(dataset=name, trial=trial, n_cal=len(cal), n_test=len(test), n_dim=cal.shape[1])
                branch_rows.append({**common, **branch, "fit_seconds": fit_seconds})
                for j, target in enumerate(targets):
                    branch_coordinate_rows.append({**common, "coordinate": j+1, "target": target,
                        "branch": diagnostic["coordinate_branch"][j],
                        "binary_candidates": diagnostic["binary_evaluations"][j],
                        "backward_candidates": diagnostic["backward_evaluations"][j]})
                timing_rows.extend({**common, "alpha": ALPHA, "repeat": r,
                                    "runtime_ms": value, "regime": branch["regime"]}
                                   for r, value in enumerate(timings))
                for method in METHODS:
                    region = tscp if method == "TSCP_R" else _function_choice(cal, ALPHA, method)
                    assert not np.isnan(region.upper).any() and np.all(region.lower == 0)
                    inside = test <= region.upper
                    joint = np.all(inside, axis=1)
                    lengths = 2 * region.upper
                    joint_rows.append({**common, "method": method, "covered_count": int(joint.sum()),
                                       "joint_coverage": float(joint.mean()), "residual_volume": region.volume(),
                                       "infinite_region": int(np.isinf(lengths).any())})
                    for j, target in enumerate(targets):
                        coordinate_rows.append({**common, "method": method, "coordinate": j+1,
                            "target": target, "full_length": lengths[j],
                            "marginal_covered_count": int(inside[:, j].sum()),
                            "marginal_coverage": float(inside[:, j].mean())})
                stress_rows.append({**common, **branch})
                for alpha in STRESS_ALPHA[1:]:
                    _, stress, _, _ = evaluate_one(cal, alpha, timing_repeats=3)
                    stress_rows.append({**common, **stress})
            if trial % 25 == 24 or trial + 1 == trials:
                print(f"Evaluated {name}: {trial+1}/{trials}, {time.perf_counter()-start:.1f}s", flush=True)
        # Checkpoint complete datasets; never label partial data as a full run.
        frames = {"real_joint_trials": joint_rows, "real_coordinate_trials": coordinate_rows,
                  "real_search_trials": branch_rows, "real_search_coordinate_trials": branch_coordinate_rows,
                  "real_search_alpha_trials": stress_rows, "real_timing_repeats": timing_rows}
        for stem, rows in frames.items():
            pd.DataFrame(rows).to_csv(TABLES / f"{stem}.csv", index=False)
    manifest = dict(datasets=datasets, trials_per_dataset=trials, alpha=ALPHA,
                    train_cal_test_fractions=[0.75, 0.05, 0.20],
                    split_seed="utility.exps._stable_hash(trial), trial=0,...,R-1; same seed in both splits",
                    fitting="original model hyperparameters; all estimator/transformer n_jobs set to one; independent splits fitted in parallel",
                    timing="serial, no concurrent fits; one warm-up; median of seven uninstrumented calls",
                    stress_timing="median of three uninstrumented calls; same cached residuals",
                    stress_alpha=STRESS_ALPHA, interval_lengths="full outcome-space length, 2 * residual half-width",
                    inference_scope="random splits conditional on each fixed dataset, not independent new datasets",
                    package_versions=versions(), python=sys.version, processor=platform.processor(),
                    logical_cpus=os.cpu_count(), platform=platform.platform(),
                    source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [Path(__file__), ROOT / "utility/res_rescaled.py", ROOT / "utility/exps.py"]})
    write_json(OUT / "run_manifest.json", manifest)
    print("Completed all requested datasets and saved the run manifest.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "fit", "evaluate"])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    assert args.trials > 0 and args.workers > 0
    os.chdir(ROOT)
    if args.phase == "prepare":
        prepare(args.datasets)
    elif args.phase == "fit":
        fit(args.datasets, args.trials, args.workers)
    else:
        evaluate(args.datasets, args.trials)


if __name__ == "__main__":
    main()
