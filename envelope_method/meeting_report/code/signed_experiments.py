"""Paired signed-score comparisons for the meeting report.

Run without arguments to recompute from the compact local score archives.
Use --import-source PATH once to extract the fixed, previously fitted trials
from report_revision. No model or comparator is selected on calibration/test
performance. The source fits, designs and hashes are retained as provenance.
"""
from pathlib import Path
from report_paths import artifact, data_path
import argparse
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLCONFIGDIR", str(REPORT / "qa" / "mpl"))
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
from utility.envelope import EnvelopeCalibration
from utility.res_rescaled import standardized_prediction
from utility.conformal_utils import conformal_rank

DATA = data_path('data/signed')
FIGURES = REPORT / "figures"
CONFIGS = [18, 19, 7]
LABELS = {18: "Gaussian, 2 outcomes\n90% marginal base",
          19: "Gaussian, 2 outcomes\n98% marginal base",
          7: "Laplace, 6 outcomes\n90% marginal base"}
METHODS = ["signed_env", "cap_old", "constant_old", "width_old"]
NAMES = dict(signed_env="Envelope signed", cap_old="Old capped",
             constant_old="Old constant shift", width_old="Old width shift")
COLORS = dict(signed_env="#087F8C", cap_old="#D17A31",
              constant_old="#7963AF", width_old="#657384")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def old_bound(scores, alpha):
    if conformal_rank(len(scores), alpha) > len(scores) or np.any(scores.std(0) == 0):
        return np.full(scores.shape[1], np.inf)
    assert scores.min() >= 0
    with np.errstate(all="ignore"):
        return standardized_prediction(scores, alpha).upper


def import_scores(source):
    """Extract only score data needed to reproduce the paired comparison."""
    source = artifact(source.resolve())
    manifest = []
    for config in CONFIGS:
        for trial in range(120):
            origin = source / "data" / f"config_{config:02d}" / f"trial_{trial:03d}.npz"
            sidecar = json.loads(origin.with_suffix(".json").read_text())
            digest = hashlib.sha256(origin.read_bytes()).hexdigest()
            assert digest == sidecar["sha256"]
            with np.load(origin) as original:
                keys = ["raw_cal", "raw_test", "width_cal", "width_test",
                        "group_cal", "group_test", "fitted_error_quantiles", "scales"]
                a = {key: original[key] for key in keys}
                a["reference_signed_upper"] = original["bound_signed_env"]
                a["reference_signed_empty"] = original["empty_signed_env"]
                a["reference_cap_upper"] = original["bound_cap_old"]
            a.update(seed=np.array(sidecar["seed"], dtype=np.int64),
                     source_sha256=np.array(digest),
                     source_config=np.array(config), source_trial=np.array(trial))
            path = DATA / f"config_{config:02d}" / f"trial_{trial:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **a)
            meta = dict(config=config, trial=trial, seed=sidecar["seed"],
                        spec=sidecar["spec"], source_sha256=digest,
                        source_archive=f"report_revision/data/config_{config:02d}/trial_{trial:03d}.npz")
            dump(path.with_suffix(".json"), meta)
            manifest.append(dict(path=path.relative_to(data_path('.')).as_posix(), **meta))
    dump(DATA / "source_manifest.json", manifest)


def fit(a, alpha):
    half = (a["fitted_error_quantiles"][:, 1] - a["fitted_error_quantiles"][:, 0]) / 2
    constant = half.max(axis=0)
    raw = a["raw_cal"]
    shifted = raw + constant
    width_shifted = raw + half[a["group_cal"]]
    assert shifted.min() >= -1e-12 and width_shifted.min() >= -1e-12
    # A zero below roundoff is corrected only after the mathematically exact shift.
    shifted = np.maximum(shifted, 0)
    width_shifted = np.maximum(width_shifted, 0)
    model = EnvelopeCalibration(raw, alpha)
    regions = [model.predict(-h) for h in half]
    upper = {"signed_env": np.array([region.upper for region in regions]),
             "cap_old": np.tile(old_bound(np.maximum(raw, 0), alpha), (2, 1)),
             "constant_old": np.tile(old_bound(shifted, alpha) - constant, (2, 1)),
             "width_old": old_bound(width_shifted, alpha)[None, :] - half}
    empty = {m: np.any(b < -half, axis=1) for m, b in upper.items()}
    empty["signed_env"] = np.array([region.empty for region in regions])
    np.testing.assert_allclose(upper["signed_env"], a["reference_signed_upper"], atol=1e-11, rtol=1e-11)
    np.testing.assert_array_equal(empty["signed_env"], a["reference_signed_empty"])
    np.testing.assert_allclose(upper["cap_old"][0], a["reference_cap_upper"], atol=1e-11, rtol=1e-11)
    return upper, empty, constant


def measure(a, upper, empty):
    rows = []
    for method in METHODS:
        b = upper[method][a["group_test"]]
        e = empty[method][a["group_test"]]
        hits = (a["raw_test"] <= b).all(axis=1) & ~e
        lengths = np.maximum(a["width_test"] + 2 * b, 0)
        lengths[e] = 0
        with np.errstate(invalid="ignore"):
            volumes = lengths.prod(axis=1)
        volumes[np.any(lengths == 0, axis=1)] = 0
        row = dict(method=method, coverage=hits.mean(), volume=volumes.mean(),
                   infinite=np.isinf(volumes).mean(), empty=e.mean(),
                   negative_adjustment=(b < 0).mean())
        for j in range(lengths.shape[1]):
            row[f"length_{j + 1}"] = lengths[:, j].mean()
        rows.append(row)
    return rows


def simulate():
    manifest = json.loads((DATA / "source_manifest.json").read_text())
    rows = []
    audit = dict(trials=0, source_reference_matches=0,
                 constant_shift_nonnegative=0, width_shift_nonnegative=0,
                 signed_contained_in_constant_old=0, finite_signed_trials=0,
                 zero_scale_capped_trials=0, zero_scale_constant_shift_trials=0,
                 zero_scale_width_shift_trials=0)
    final_manifest = []
    for entry in manifest:
        path = artifact(entry["path"])
        with np.load(path) as archive:
            a = {k: archive[k] for k in archive.files if not k.startswith(("upper_", "empty_"))}
        upper, empty, constant = fit(a, entry["spec"]["alpha"])
        a.update({"upper_" + method: b for method, b in upper.items()})
        a.update({"empty_" + method: b for method, b in empty.items()})
        a["constant_shift"] = constant
        np.savez_compressed(path, **a)
        for row in measure(a, upper, empty):
            rows.append(dict(config=entry["config"], trial=entry["trial"], seed=entry["seed"],
                             family=entry["spec"]["family"], n=entry["spec"]["n"],
                             d=entry["spec"]["d"], alpha=entry["spec"]["alpha"],
                             base_alpha=entry["spec"]["base_alpha"], **row))
        audit["trials"] += 1
        audit["source_reference_matches"] += 1
        audit["constant_shift_nonnegative"] += 1
        audit["width_shift_nonnegative"] += 1
        audit["zero_scale_capped_trials"] += int(np.any(np.maximum(a["raw_cal"], 0).std(0) == 0))
        audit["zero_scale_constant_shift_trials"] += int(np.any((a["raw_cal"] + constant).std(0) == 0))
        audit["zero_scale_width_shift_trials"] += int(np.any((a["raw_cal"] + a["width_cal"] / 2).std(0) == 0))
        # This is a diagnostic only; the report's main theorem uses equal scores.
        for group in range(2):
            assert empty["signed_env"][group] or np.all(
                upper["signed_env"][group] <= upper["constant_old"][group] +
                1e-8 * (1 + np.abs(upper["constant_old"][group])))
        audit["signed_contained_in_constant_old"] += 1
        audit["finite_signed_trials"] += int(np.isfinite(upper["signed_env"]).all())
        final_manifest.append(dict(path=entry["path"], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   source_sha256=entry["source_sha256"], seed=entry["seed"]))
        if (entry["trial"] + 1) % 120 == 0:
            print(f"Completed paired signed config {entry['config']}: 120 trials", flush=True)
    pd.DataFrame(rows).to_csv(DATA / "trials.csv", index=False)
    dump(DATA / "audit.json", audit)
    dump(DATA / "manifest.json", final_manifest)
    dump(DATA / "design.json", dict(configs=CONFIGS, trials_per_config=120, train=800, test=1200,
         alpha=.1, n_calibration=80, scores="raw signed CQR", methods=METHODS,
         constant_shift="max over fitted scale groups of fitted base half-width, coordinatewise",
         width_shift="fitted base half-width at each observation; equivalent to absolute deviation from base midpoint",
         fitting="OLS plus training-error quantiles within two covariate scale groups",
         selection="No method selection; smallest average old volume is a descriptive comparison only",
         geometry=dict(configs=[18, 19], trial=0, test_index=0, predetermined=True)))


def moments(values):
    a = np.asarray(values)
    if not np.isfinite(a).all():
        return float(a.mean()), np.nan, np.nan
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    return float(a.mean()), se, float(student_t.ppf(.975, len(a) - 1) * se)


def summarize():
    df = pd.read_csv(DATA / "trials.csv")
    summaries, comparisons, best = [], [], []
    for (config, method), g in df.groupby(["config", "method"]):
        row = dict(config=int(config), method=method, replicates=len(g))
        for metric in ["coverage", "volume", "infinite", "empty", "negative_adjustment"]:
            mu, se, ci = moments(g[metric])
            row.update({metric + "_mean": mu, metric + "_mcse": se, metric + "_ci_halfwidth": ci})
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(DATA / "summary.csv", index=False)
    for config in CONFIGS:
        g = df[df.config == config]
        w = g.pivot(index="trial", columns="method", values="volume")
        for old in METHODS[1:]:
            good = np.isfinite(w.signed_env) & np.isfinite(w[old]) & (w[old] > 0)
            ratios = w.loc[good, "signed_env"] / w.loc[good, old]
            mu, se, ci = moments(ratios)
            comparisons.append(dict(config=config, baseline=old, mean_paired_volume_ratio=mu,
                                    ratio_mcse=se, ratio_ci_halfwidth=ci, finite_pairs=int(good.sum()),
                                    excluded_pairs=int((~good).sum()), reduction_percent=100 * (1 - mu),
                                    reduction_ci_halfwidth=100 * ci))
        ss = summary[(summary.config == config) & (summary.method != "signed_env")]
        selected = ss.sort_values("volume_mean").iloc[0]
        envelope = summary[(summary.config == config) & (summary.method == "signed_env")].iloc[0]
        best.append(dict(config=config, best_average_old=selected.method,
                         envelope_mean_volume=float(envelope.volume_mean),
                         best_old_mean_volume=float(selected.volume_mean),
                         ratio_of_mean_volumes=float(envelope.volume_mean / selected.volume_mean)))
    pd.DataFrame(comparisons).to_csv(DATA / "paired_comparisons.csv", index=False)
    dump(DATA / "findings.json", dict(best_average_old=best, summaries=summaries, comparisons=comparisons))
    lines = ["# Signed residual comparison", "", "All results use 120 paired fitted trials, n=80 and alpha=0.1. "
             "Intervals and MCSEs are across trials. The strongest old baseline by average size is "
             "a retrospective descriptive label, not a test-time method-selection procedure.", ""]
    for setting in best:
        config = setting["config"]
        lines.append(f"## {LABELS[config].replace(chr(10), ': ')}")
        for row in summaries:
            if row["config"] == config:
                lines.append(f"- {NAMES[row['method']]}: coverage {row['coverage_mean']:.5f} "
                             f"(MCSE {row['coverage_mcse']:.5f}), mean volume {row['volume_mean']:.5g}, "
                             f"infinity {row['infinite_mean']:.3f}, negative adjustments {row['negative_adjustment_mean']:.3f}.")
        lines.append(f"Smallest average old volume: {NAMES[setting['best_average_old']]}; "
                     f"envelope / old ratio of mean volumes = {setting['ratio_of_mean_volumes']:.5f}.\n")
    (REPORT / "data/signed/findings.md").write_text("\n".join(lines), encoding="utf-8")
    return df, summary, pd.DataFrame(comparisons)


def plots(df, summary, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#AAB2BA",
        "axes.titleweight": "normal", "grid.color": "#E4E7EA",
        "grid.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42})
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5), layout="constrained")
    for col, config in enumerate(CONFIGS):
        ss = summary[summary.config == config].set_index("method")
        ax = axes[0, col]
        for x, method in enumerate(METHODS):
            row = ss.loc[method]
            ax.errorbar(x, row.coverage_mean, yerr=row.coverage_ci_halfwidth,
                        fmt="o", color=COLORS[method], capsize=3, markersize=4)
        ax.axhline(.9, color="#77828D", linestyle="--", linewidth=1)
        ax.set(title=LABELS[config], xticks=range(4), xticklabels=["Env.", "Cap", "Const.", "Width"],
               ylim=(.87, 1.005))
        if col == 0:
            ax.set_ylabel("Joint coverage")
        ax.grid(axis="y")
        ax = axes[1, col]
        pp = paired[paired.config == config].set_index("baseline")
        for x, method in enumerate(METHODS[1:]):
            row = pp.loc[method]
            ax.errorbar(x, row.mean_paired_volume_ratio, yerr=row.ratio_ci_halfwidth,
                        fmt="o", color=COLORS[method], capsize=3, markersize=4)
            if row.excluded_pairs:
                ax.annotate(f"{int(row.finite_pairs)} finite pairs", (x, row.mean_paired_volume_ratio),
                            xytext=(0, -18), textcoords="offset points", ha="center", fontsize=6.5)
        ax.axhline(1, color="#77828D", linestyle="--", linewidth=1)
        ax.set(xticks=range(3), xticklabels=["Capped", "Constant\nshift", "Width\nshift"],
               xlabel="Old comparator", ylim=(.30, 1.22))
        if col == 0:
            ax.set_ylabel("Envelope / old volume")
        ax.grid(axis="y")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=COLORS[m], linestyle="", label=NAMES[m])
                        for m in METHODS], loc="outside lower center", ncol=4, frameon=False)
    for extension in ["pdf", "png"]:
        fig.savefig(FIGURES / ("signed_comparison." + extension), bbox_inches="tight", dpi=190)
    plt.close(fig)
    geometry = []
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85), layout="constrained")
    for ax, config in zip(axes, [18, 19]):
        with np.load(DATA / f"config_{config:02d}" / "trial_000.npz") as a:
            group = int(a["group_test"][0])
            half = a["width_test"][0] / 2
            scales = a["scales"]
            radii = {m: (half + a["upper_" + m][group]) / scales for m in METHODS}
            base = half / scales
            finite = [r for r in radii.values() if np.isfinite(r).all()]
            limit = np.maximum(base, np.max(finite, axis=0)) * 1.12
            ax.add_patch(Rectangle(-base, 2 * base[0], 2 * base[1],
                                   facecolor="#EEF1F4", edgecolor="#ABB3BC", linewidth=1))
            for m in ["cap_old", "constant_old", "width_old", "signed_env"]:
                r = radii[m]
                if np.isfinite(r).all():
                    ax.add_patch(Rectangle(-r, 2 * r[0], 2 * r[1], fill=False,
                        edgecolor=COLORS[m], linewidth=1.8, linestyle="-" if m == "signed_env" else "--"))
                else:
                    ax.text(.03, .96, f"{NAMES[m]} is infinite", transform=ax.transAxes,
                            va="top", fontsize=8, color=COLORS[m],
                            bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
            ax.plot(0, 0, "+", color="#29333D", markersize=8)
            ax.set(xlim=(-limit[0], limit[0]), ylim=(-limit[1], limit[1]),
                   title=LABELS[config].replace("\n", ": "),
                   xlabel="Outcome 1 deviation / noise scale")
            ax.set_aspect("equal", adjustable="box")
            if config == 18:
                ax.set_ylabel("Outcome 2 deviation / noise scale")
            geometry.append(dict(config=config, trial=0, test_index=0, group=group,
                                 source_seed=int(a["seed"]), base_radii=base.tolist(),
                                 radii={m: r.tolist() for m, r in radii.items()}))
    fig.legend(handles=[Line2D([0], [0], color=COLORS[m], linestyle="-" if m == "signed_env" else "--", label=NAMES[m])
                        for m in METHODS], loc="outside lower center", ncol=4, frameon=False)
    for extension in ["pdf", "png"]:
        fig.savefig(FIGURES / ("signed_regions." + extension), bbox_inches="tight", dpi=190)
    plt.close(fig)
    dump(DATA / "geometry.json", geometry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-source", type=Path,
                        help="directory containing the original report_revision archives")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.import_source:
        import_scores(args.import_source.resolve())
    with threadpool_limits(limits=1):
        if not args.plots_only:
            simulate()
        df, summary, paired = summarize()
        plots(df, summary, paired)
    print("Signed comparisons, compact archives, audits and figures completed.", flush=True)


if __name__ == "__main__":
    main()
