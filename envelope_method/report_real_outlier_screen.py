"""Audit and report the fixed-rule Crime sensitivity experiment and cohort screen."""

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl"))
from envelope_method.crime_outlier_control import evaluate

OUT = ROOT / "data/envelope_method/results/real_outlier_screen"
REPORT_OUT = ROOT / 'envelope_method/results/real_outlier_screen'
REPORT_OUT.mkdir(parents=True, exist_ok=True)
CRIME = OUT / "crime_control"
SOURCE = ROOT / "data/envelope_method/results/extra_real/crime"
RF2 = ROOT / "data/envelope_method/results/rf2_standardized_outlier_control"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_archives(frame, metadata):
    """Recompute both methods from residuals, common-test metrics, and controls."""
    assert digest(SOURCE / "dataset.pkl") == metadata["source_sha256"]
    row = metadata["removed_position"]
    records, test_controls, hashes = 0, 0, {}
    for trial in range(200):
        original_path = SOURCE / f"trial_{trial:03d}.npz"
        hashes[str(original_path.relative_to(ROOT))] = digest(original_path)
        with np.load(original_path) as z:
            original = {k: z[k].copy() for k in z.files}
        for scaled in [False, True]:
            retained_bounds = None
            for removed in [False, True]:
                if not scaled and not removed:
                    archive = original
                else:
                    path = CRIME / (
                        f"trial_{trial:03d}_{'removed' if removed else 'retained'}_"
                        f"{'scaled' if scaled else 'raw'}.npz"
                    )
                    hashes[str(path.relative_to(ROOT))] = digest(path)
                    with np.load(path) as z:
                        archive = {k: z[k].copy() for k in z.files}
                bounds = evaluate(archive["scores_cal"], archive["scores_test"])
                common = archive["test_indices"] != row
                for method, upper in bounds.items():
                    assert np.isfinite(upper).all() and (upper > 0).all()
                    # Original archives preserve the pre-correction CHR rank.
                    # Their residuals are reused; today's table uses recomputed bounds.
                    historical_chr = not scaled and not removed and method == "Point_CHR"
                    if method in archive and not historical_chr:
                        np.testing.assert_allclose(
                            upper, archive[method], rtol=1e-12,
                            err_msg=f"trial={trial}, scaled={scaled}, removed={removed}, method={method}",
                        )
                    result = frame[
                        (frame.trial == trial)
                        & (frame.scaled == scaled)
                        & (frame.removed == removed)
                        & (frame.method == method)
                    ]
                    assert len(result) == 1
                    r = result.iloc[0]
                    inside = (archive["scores_test"] <= upper).all(axis=1)
                    np.testing.assert_allclose(
                        [inside.mean(), inside[common].mean(), np.prod(2 * upper)],
                        [r.coverage, r.common_test_coverage, r.volume],
                        rtol=1e-12,
                    )
                    records += 1
                    if removed and row in original["test_indices"]:
                        np.testing.assert_array_equal(upper, retained_bounds[method])
                        test_controls += 1
                if not removed:
                    retained_bounds = bounds
    return dict(
        status="passed",
        method_records_recomputed_from_calibration_residuals=records,
        identical_bounds_for_test_only_deletion=test_controls,
        original_chr_bounds_recomputed_using_corrected_rank=200,
        hashes=hashes,
    )


def diagnostics(frame):
    effects, tails, intervals = [], [], []
    for (scaled, method), g in frame.groupby(["scaled", "method"]):
        volume = g.pivot(index="trial", columns="removed", values="volume")
        coverage = g.pivot(index="trial", columns="removed", values="common_test_coverage")
        delta = coverage[True] - coverage[False]
        halfwidth = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
        effects.append(dict(
            scaled=bool(scaled), method=method,
            removed_over_retained_mean_volume=float(volume[True].mean() / volume[False].mean()),
            median_paired_removal_ratio=float((volume[True] / volume[False]).median()),
            common_coverage_change=float(delta.mean()),
            common_coverage_mc95=[float(delta.mean() - halfwidth), float(delta.mean() + halfwidth)],
        ))
    for (scaled, removed, method), g in frame.groupby(["scaled", "removed", "method"]):
        v = g.sort_values("volume", ascending=False)
        tails.append(dict(
            scaled=bool(scaled), removed=bool(removed), method=method,
            top_one_volume_share=float(v.volume.head(1).sum() / v.volume.sum()),
            top_five_volume_share=float(v.volume.head(5).sum() / v.volume.sum()),
            original_calibration_partition_volume_share=float(
                g[g.outlier_partition == "cal"].volume.sum() / g.volume.sum()
            ),
            largest_trials=v[["trial", "volume", "outlier_partition"]].head(5).to_dict("records"),
        ))
    # Descriptive split-level Monte Carlo uncertainty, conditional on this dataset.
    # The same sampled trial indices are used for paired methods and treatments.
    indices = np.random.default_rng(20260909).integers(0, 200, size=(20000, 200))
    for (scaled, removed), g in frame.groupby(["scaled", "removed"]):
        v = g.pivot(index="trial", columns="method", values="volume")
        env, chr_ = v.Envelope.to_numpy(), v.Point_CHR.to_numpy()
        ratio = env[indices].mean(axis=1) / chr_[indices].mean(axis=1)
        intervals.append(dict(
            scaled=bool(scaled), removed=bool(removed), replicates=20000,
            ratio_mean_volume_mc95=np.quantile(ratio, [.025, .975]).tolist(),
            geometric_mean_paired_ratio=float(np.exp(np.log(env / chr_).mean())),
        ))
    return dict(effects=effects, tail_concentration=tails, split_bootstrap=intervals)


def main():
    frame = pd.read_csv(CRIME / "trials.csv")
    summary = pd.read_csv(CRIME / "summary.csv")
    metadata = json.loads((CRIME / "metadata.json").read_text())
    fit_audit = json.loads((CRIME / "audit.json").read_text())
    screen = json.loads((OUT / "screen.json").read_text())
    extra = diagnostics(frame)
    extra["archive_audit"] = audit_archives(frame, metadata)
    (CRIME / "diagnostics.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    by_partition = frame.groupby(["scaled", "removed", "method", "outlier_partition"]).agg(
        trials=("trial", "size"), volume=("volume", "mean"),
        coverage=("common_test_coverage", "mean"),
    ).reset_index()
    by_partition.to_csv(CRIME / "by_partition.csv", index=False)
    with (SOURCE / "dataset.pkl").open("rb") as handle:
        X, y, _ = pickle.load(handle)
    row = metadata["removed_position"]
    raw = y.iloc[row]
    observations = pd.DataFrame({
        "target": y.columns,
        "removed_value": raw.values,
        "empirical_percentile": [(y[t] <= raw[t]).mean() for t in y],
        "maximum_elsewhere": y.drop(y.index[row]).max().values,
    })
    observations.to_csv(CRIME / "removed_observation.csv", index=False)
    overview = pd.DataFrame([
        {k: r[k] for k in ["dataset", "rows", "outputs", "envelope_coverage", "chr_coverage", "ratio"]}
        for r in screen
    ])
    overview.to_csv(OUT / "screen_summary.csv", index=False)

    lines = [
        "# Outlier sensitivity: standardized rf2 and other existing real cohorts", "",
        "rf2 shows substantial outlier sensitivity even with standardized training, but no reversal against Point CHR. Crime shows a numerical reversal in mean volume after one fixed deletion, including with standardized training. Crime's result is dominated by rare extreme-volume splits; it does not establish a consistent advantage after cleaning or an erroneous record.", "",
        "## Controlled rf2 comparison", "",
        "200 matched splits per treatment, with identical forest settings. Target means and population SDs are fitted on each training partition only, and predictions are converted back to original units before calibration. The retained condition has 200 new fits; the removed condition reuses the previously completed 200 standardized fits. All volumes are products of full interval lengths, at nominal 90% joint coverage.", "",
        "| Outlier | Method | Joint coverage | Mean full volume |", "|---|---|---:|---:|",
    ]
    rf2 = pd.read_csv(RF2 / "summary.csv")
    for treatment in ["Retained", "Removed"]:
        for method in ["Envelope", "Point_CHR"]:
            r = rf2[(rf2.treatment == treatment) & (rf2.method == method)].iloc[0]
            lines.append(f"| {treatment} | {method.replace('_', ' ')} | {r.coverage*100:.3f}% | {r.volume:,.1f} |")
    lines += [
        "", "Removing the NASI2 value 78.9 (next largest 5.62) reduces envelope mean volume by 85.55%, versus 37.92% for Point CHR. Envelope/CHR mean-volume ratio falls from 6.347 to 1.477. The median within-split envelope volume reduction is 46.44%; the larger mean reduction reflects extreme-volume trials. Coverage on the identical remaining test observations changes by +0.0059 percentage points for envelope.", "",
        "Training standardization improves target weighting in the forest but does not make calibration means and SDs resistant to extreme residuals. The six splits with this observation in calibration have envelope mean volumes 1,143,702 retained versus 8,622 removed. In all 36 test-only deletions the bounds are identical.", "",
        "rf2 is still a useful outlier-sensitivity example under this random-split protocol. Temporal dependence limits interpreting its coverage as future-forecast coverage; it does not invalidate this controlled comparison. The value is statistically extreme, but a recording error has not been established.", "",
        "[Detailed rf2 report](../rf2_standardized_outlier_control/REPORT.md) · [rf2 trial data](../rf2_standardized_outlier_control/trials.csv)", "",
        "## Screening the existing cohorts", "",
        "The table below uses the current complete-data, original-training benchmark and the corrected Point CHR recalibration rank. Only rf2 and Crime start with larger mean volumes for envelope. This is a descriptive candidate screen; deletion fits were run for rf2 and Crime, not for every cohort.", "",
        "| Dataset | Envelope / CHR mean volume | Relevance |", "|---|---:|---|",
    ]
    reasons = {
        "stock": "Point CHR is unbounded in all 200 small-calibration splits; not a finite losing baseline.",
        "rf2": "Controlled extreme-observation study above; no reversal.",
        "scm1d": "Envelope already has smaller mean volume.",
        "scm20d": "Envelope already has smaller mean volume.",
        "energy": "Envelope already has smaller mean volume; building simulation outputs.",
        "student": "Envelope already has smaller mean volume; student-level records.",
        "air": "Envelope is slightly smaller after the CHR rank correction; hourly time series.",
        "crime": "Community-level records; fixed largest-population removal tested below.",
    }
    for r in screen:
        ratio = "CHR unbounded" if r["dataset"] == "stock" else f'{r["ratio"]:.3f}'
        lines.append(f'| {r["dataset"]} | {ratio} | {reasons[r["dataset"]]} |')
    lines += [
        "", "Dataset descriptions: [UCI Crime](https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized), [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency), and [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality). Energy consists of simulated building configurations, so it is not a natural measurement-error example.", "",
        "## Crime: a qualified reversal", "",
        f"Removal was fixed once, using the maximum input population before examining deletion outcomes: retained position {row}, original dataframe index {metadata['original_index']}, population {metadata['population']:,.0f} versus {metadata['next_largest_population']:,.0f} next largest. The original 1,902 rows become 1,901. Other train/calibration/test memberships are preserved. All four conditions have 200 splits: 600 new fits plus 200 reused original fits. The same corrected Point CHR formula is used throughout.", "",
        "| Training targets | Largest community | Method | Joint coverage | Mean full volume | Median full volume |",
        "|---|---|---|---:|---:|---:|",
    ]
    for scaled in [False, True]:
        for removed in [False, True]:
            for method in ["Envelope", "Point_CHR"]:
                r = summary[(summary.scaled == scaled) & (summary.removed == removed) & (summary.method == method)].iloc[0]
                lines.append(f"| {'Standardized' if scaled else 'Original'} | {'Removed' if removed else 'Retained'} | {method.replace('_', ' ')} | {r.coverage*100:.3f}% | {r.volume:.4e} | {r.median_volume:.4e} |")
    lines += ["", "| Training targets | Largest community | Ratio of mean volumes | Median paired Env/CHR ratio | Envelope smaller | Split-bootstrap 95% interval for mean ratio |", "|---|---|---:|---:|---:|---:|"]
    for r in fit_audit["method_ratios"]:
        b = next(b for b in extra["split_bootstrap"] if b["scaled"] == r["scaled"] and b["removed"] == r["removed"])
        lo, hi = b["ratio_mean_volume_mc95"]
        lines.append(f"| {'Standardized' if r['scaled'] else 'Original'} | {'Removed' if r['removed'] else 'Retained'} | {r['ratio_of_means']:.4g} | {r['median_paired_ratio']:.3f} | {round(200*r['envelope_smaller_fraction'])}/200 | [{lo:.3g}, {hi:.3g}] |")
    lines += [
        "", "A ratio below one favors envelope. Under standardized training, removing the community reduces envelope's mean volume by 99.917% and CHR's by 68.34%, reversing the mean-volume ratio from 38.615 to 0.101. Envelope common-test coverage changes by -0.063 percentage points. With original training, the mean-volume ratio also reverses, from 2,081.97 to 0.0354, although Point CHR's mean volume increases after the deletion.", "",
        "### What drives the reversal", "",
        "The community occurs in training in 143 splits, calibration in 11, and testing in 46. With standardized training, those 11 calibration splits contribute 99.90% of envelope's total retained-condition volume. Their mean volume falls from 1.285e67 to 1.092e60 on deletion. The five largest retained-envelope trials contribute 99.28% of its total volume. These are empirical signs of rare calibration-tail inflation, amplified by multiplying 18 interval widths.", "",
        "The advantage is less persuasive for typical splits: after standardized deletion, envelope is smaller in exactly 100/200 splits, the median paired Env/CHR ratio is 0.932, and the geometric mean paired ratio is 1.084. Envelope's median paired volume change from deletion is only a 0.274% reduction. Both methods continue to have very skewed volume distributions.", "",
        "The 20,000 paired split-bootstrap replicates give a 95% interval [0.0113, 2.58] for the standardized removed-condition ratio of mean volumes. It crosses one. These are descriptive Monte Carlo intervals conditional on this dataset and split protocol, not population confidence intervals; rare unobserved extreme splits remain a limitation. A mean-volume reversal in these 200 splits is observed, but a reliable general superiority claim is not established.", "",
        "### Is the community an outlier?", "",
        "It is an influential upper-tail observation in population and several crime counts. Its maximum robust z-score after log1p target transformation is 6.696, compared with 6.405 and 6.120 for the next two observations. This is less isolated than rf2's NASI2 value. The UCI dataset mixes eight counts and ten rates across 18 outcomes; exceptionally large communities can legitimately have exceptionally large counts. The counts and population make a size effect plausible, but this analysis does not verify the record's accuracy or prove a data-entry error. Label this a largest-community sensitivity analysis, not a corrected dataset.", "",
        "## Interpretation for the experiments", "",
        "Crime supplies the requested numerical loss-to-win reversal in mean volume, with an explicit qualification that it is driven by rare tails and an influential community, not a confirmed erroneous observation. rf2 supplies clearer evidence of an isolated statistical outlier and strong sensitivity after standardized training, but no reversal. The current experiments therefore support a careful sensitivity argument more strongly than a general claim that removing outliers makes envelope outperform Point CHR.", "",
        "## Verification and saved data", "",
        "The fit audit checks all 600 new Crime archives, deletion/index alignment, and training-only transformations. The independent reporting audit recalculates both calibration bounds and every coverage/volume record for all 800 fit conditions (1,600 method records). Original archives preserve historical Point CHR bounds from before the rank correction; their residuals are reused to recompute the corrected comparator throughout this report. All 184 method comparisons across the 46 test-only deletion splits and two training choices reproduce exactly. Dataset and archive SHA-256 hashes are recorded. No source observations or primary benchmark results were overwritten.", "",
        "[Crime trial data](crime_control/trials.csv) · [Crime summary](crime_control/summary.csv) · [Crime diagnostics and archive hashes](crime_control/diagnostics.json) · [Removed observation](crime_control/removed_observation.csv) · [Screen summary](screen_summary.csv)", "",
        "Reproduction: run `rf2_standardized_outlier_control.py`, `report_rf2_standardized_control.py`, `screen_real_outlier_examples.py`, `crime_outlier_control.py`, then `report_real_outlier_screen.py` from the project root with the configured Python dependencies. The fit scripts reuse existing completed checkpoints.",
    ]
    (REPORT_OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure(rf2, summary, extra)
    print(json.dumps({k: v for k, v in extra["archive_audit"].items() if k != "hashes"}, indent=2))
    print("Report, tables, and figure saved.")


def figure(rf2, crime, extra):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.6), layout="constrained")
    conditions = [("rf2\nStandardized training", rf2, None),
                  ("Crime\nOriginal training", crime, False),
                  ("Crime\nStandardized training", crime, True)]
    for ax, (title, data, scaled) in zip(axes, conditions):
        ratios = []
        for removed in [False, True]:
            if scaled is None:
                subset = data[data.treatment == ("Removed" if removed else "Retained")]
            else:
                subset = data[(data.scaled == scaled) & (data.removed == removed)]
            v = subset.set_index("method").volume
            ratios.append(v.Envelope / v.Point_CHR)
        ax.bar([0, 1], ratios, width=.56, color=["#a55a40", "#167c80"])
        ax.set_yscale("log")
        ax.axhline(1, color="#444444", linestyle="--", linewidth=1.1)
        for i, r in enumerate(ratios):
            ax.annotate(f"{r:,.3g}", (i, r), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=11)
        ax.set_xticks([0, 1], ["Retained", "Removed"])
        ax.set_title(title, pad=14)
        ax.set_ylim(min(.5, min(ratios) / 3), max(ratios) * 5)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Envelope / Point CHR mean full volume (log scale)")
    fig.suptitle("Single-observation sensitivity | 200 matched splits per condition", fontsize=14)
    fig.supxlabel("Below 1 favors envelope. Crime means are dominated by rare extreme-volume splits.", fontsize=10)
    fig.savefig(REPORT_OUT / "comparison.png", dpi=180)
    fig.savefig(REPORT_OUT / "comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
