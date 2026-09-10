# Where the synthetic and real results come from

> September 10 layout update: numerical files referenced here now live under `data/` with the same repository-relative suffix. See [data layout](DATA_LAYOUT.md). The author removed `deletable/` before this reorganization; historical rollback copies are no longer local.

Verified 2026-09-09. These folders are not interchangeable with an unused archive.

| Location | What it contains | Relationship to richer results | Treatment |
|---|---|---|---|
| `syn_exps/` | 96 current aggregate CSV exports, 186,356 bytes | Derived from the same retained synthetic trials in `envelope_method/results/absolute/`, `auxiliary/` and `full_lwc_scaling/` | Keep as the small notebook/figure interface |
| `real_exps/*.csv` | Eight historical aggregate tables, 12,246 bytes | Earlier runs; newer real cohorts are reruns, not recovery of these original trials | Keep and label historical; use the audited current table for paper comparisons |
| `real_exps/data/*.arff` | Four raw datasets, 63,988,927 bytes | Current loader inputs, not experiment output | Keep |
| `envelope_method/results/real_comparison_audited.csv` | Corrected current eight-dataset comparison | Six cached reviewer cohorts plus reconstructed Air/Crime cohorts | Current real comparison entry point |

## Synthetic exports: the same experiments at different levels of detail

`export_fresh_tables.py` produces `syn_exps` from retained trial tables. All **1,016 exported rows** reconcile with their source data: **97 unique configurations, 994 distinct configuration/method summaries, and 15,480 checked summary/metadata cells, with zero discrepancies**. The underlying source rows also match the consolidated `simulation_trials.csv` (176,520 method/trial rows checked).

The exports cover Cauchy, Gamma, Gaussian, unit Gaussian, Laplace, mixed noise and Student-t families. Their methods include auxiliary comparators; they do not all come from `absolute/` alone. Some full-LWC and exact-ratio entries require `auxiliary/` or `full_lwc_scaling/`.

There are **22 repeated configuration/method summaries** in broad Laplace and specialized calibration-size views. Those are repeated presentations of the same trials, not additional independent replications. No two complete CSV files are byte-identical. There is no numerical evidence available only in `syn_exps`.

Two unit conversions are intentional: synthetic exports use residual half-width volume, equal to full outcome-space volume divided by `2^d`, and half of the full maximum side length. Comparing files without converting units would suggest false discrepancies.

`exps.ipynb` and `reviewer_update/unify_experiment_figures.py` are wired to these exports. They are cheap, useful derived products; removing them would break existing readers until regeneration. Numerical consistency does not establish that every saved manuscript figure has been regenerated.

Full-LWC coverage is narrower than the main method sweep: broad Laplace has dimension 2 at calibration sizes 30/50/100/300/500 (200 trials each); the calibration-size-30 view adds dimensions 3 and 4 (30 trials each); calibration size 10 covers dimensions 2–6 (10 trials each). The exports do not imply complete high-dimensional full-LWC comparisons.

See the [per-family audit](../output/results_provenance_2026-09-09/syn_exps_audit.md) and [machine-readable checks](../data/output/results_provenance_2026-09-09/syn_exps_audit.json).

## Real results: historical summaries and newer paired evaluations

The original CSVs cover Air, Crime, Energy, RF2, SCM1D, SCM20D, Stock and Student. Each has eight methods: Scaled Shortcut/Split, Standardized Shortcut/Split, Point CHR, Empirical copula, Unscaled and Bonferroni. They contain aggregates, not saved individual trials, splits or fitted models. Several original baselines are absent from the current six-method comparison, so these files also document older comparison scope.

The later data lineage is:

1. **Reviewer diagnostic reruns:** six datasets, 200 splits each, intended original model/split protocol. The [diagnostic README](../reviewer_update/real_diagnostics/README.md) explicitly identifies new reruns, not recovery of the original fitted models or timings. These save dataset fingerprints, split indices and residuals.
2. **Current envelope evaluations:** `run_real.py` evaluates methods on those same cached reviewer residuals. This extends the same modern fitted cohort without another fitting run; source hashes connect the records.
3. **Air/Crime reconstruction:** `run_extra_real.py` reconstructs splits and refits because original residual caches were unavailable.
4. **Audited comparison:** [REAL_COMPARISON.md](../envelope_method/results/REAL_COMPARISON.md) and [real_comparison_audited.csv](../data/envelope_method/results/real_comparison_audited.csv) assemble the eight datasets with corrected Point CHR ranks and full outcome-space volume. Generic older summary files are not interchangeable with this audited table.

For example, RF2's original standardized-shortcut mean coverage is **0.901402995**, versus **0.901943359** in the current audited results. Even after multiplying the old half-width volume by `2^d`, its mean volume is **138,778.905**, versus **124,472.547** now. Thus these are not exact numerical duplicates with merely more detail attached. Matching some other aggregates would not prove identical original fits.

The eight historical CSVs have no byte-identical copies elsewhere in the repository. Six still feed the historical runtime figure builder and coverage audit. Their total storage is only about 12 KB. The four ARFF datasets (RF1, RF2, SCM1D and SCM20D) are active supported loader inputs; RF2 also supports the outlier/source checks. RF1 has no corresponding saved result CSV here.

## Recommendation and checks

Keep these small interfaces and inputs at their current paths. Treat `envelope_method/results/` as the richer current evidence and label the original real summaries as historical. For missing real baselines, evaluate them on the current cached residuals to obtain new paired comparisons; this does not recover the original trials or historical timings.

No experiment results were moved, deleted or regenerated during this provenance check. A legacy JSON-reader mismatch discovered during inspection was corrected: the summary reader now accepts the real split checkpoint's `trials` field alongside modern `records` checkpoints. Nine storage-consumer tests pass; all 18,000 method/alpha/trial rows in the six current real cohorts reconstruct consistently with their CSVs after normal CSV missing-value parsing.
