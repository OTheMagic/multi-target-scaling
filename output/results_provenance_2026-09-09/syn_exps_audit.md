# Current `syn_exps` provenance audit

Checked on 2026-09-09. All research data was read only. No exporter, experiment, notebook, or cleanup operation was run.

`syn_exps` is a small, current collection of notebook-compatible summary exports. It is not the archive of obsolete experiments. Every current row can be reproduced from the newer `envelope_method/results` trial tables. `exps.ipynb` still contains readers for these paths (cells 3, 5, 7, 8, 10), although its saved execution counts and outputs are cleared. The larger archive provides the owning experiment records, configurations, and retained numerical arrays; these CSVs supply convenient plotting views.

## Exhaustive checks

- **96 files, all CSV; 186,356 bytes** (about 182 KiB).
- **1,016 exported summary rows**, representing **994 distinct configuration–method summaries**, **97 unique configurations**, and **17,430 shared primary trials**. Different methods on a shared trial are not independent experiments.
- All **15,480 summary and metadata cells** checked against the owning `absolute`, `auxiliary`, and `full_lwc_scaling` trial CSVs: **zero mismatches** at relative tolerance 1e-11 / absolute tolerance 1e-12, with matching NaN/infinity semantics.
- All **176,520 distinct method–trial rows** used by the exports also match `simulation_trials.csv`: trial IDs, coverage, outcome volume, runtime, and **1,072,420 coordinate values** checked, with zero mismatches.
- **No exact duplicate whole files.** There are **22 repeated configuration–method rows** across the broad Laplace and calibration-size-30 views. They use the same saved trials, not fresh independent repetitions.
- **No current CSV, row, or method was found only in `syn_exps`.** No unmapped legacy file remains in this folder.

## What the current tables contain

| View | Files | Rows | Configurations | Dimensions | Calibration sizes |
|---|---:|---:|---:|---|---|
| Cauchy | 10 | 120 | 12 | 2, 10 | 30, 50, 100, 300, 500, 1000 |
| Gamma | 10 | 100 | 10 | 2, 10 | 30, 50, 100, 300, 500 |
| Gaussian | 10 | 100 | 10 | 2, 10 | 30, 50, 100, 300, 500 |
| Unit Gaussian | 10 | 100 | 10 | 2, 10 | 30, 50, 100, 300, 500 |
| Broad Laplace | 12 | 110 | 10 | 2, 10 | 30, 50, 100, 300, 500 |
| Laplace, calibration size 30 | 12 | 86 | 8 | 2, 3, 4, 5, 10, 20, 30, 50 | 30 |
| Laplace, calibration size 10 | 12 | 60 | 5 | 2, 3, 4, 5, 6 | 10 |
| Mixed | 10 | 100 | 10 | 2, 10 | 30, 50, 100, 300, 500 |
| Student-t | 10 | 240 | 24 | 2, 10 | 30, 500 |

Student-t degrees of freedom are 1.5, 2, 3, 10, 30, and 100. All exported settings have alpha 0.1. The configuration counts above overlap by two; summing them gives 99 appearances, not 99 distinct configurations.

## Derived views, fresh evidence, and exceptions

| Category | Current export files / rows | Owning evidence | Interpretation |
|---|---|---|---|
| Envelope, TSCP-R, TSCP-GWC, Unscaled, Point CHR, Empirical copula | 54 / 594 | `results/absolute/<config>/trials.csv` | Six methods on the shared fresh fitted trials; current CSVs are derived summaries. |
| TSCP-S, Naive, Bonferroni, Population oracle | 36 / 396 | `results/auxiliary/<config>/trials.csv` | Additional comparator calculations on the same primary trials. These are absent from the six-method primary table, but present in the wider archive and consolidated trial table. |
| Old full LWC and exact-ratio LWC | 6 / 26 | `results/auxiliary` for the five broad 2D Laplace cohorts; `results/full_lwc_scaling` for the additional dimension studies | Distinct method evidence, fully preserved in the wider archive. Each method has 1,110 saved trials across 12 distinct configurations. These are not equivalent to TSCP-GWC or Envelope. |
| Historical fixed-pool results | 0 current files | Historical migration archive, outside current `syn_exps` | The fresh-protocol migration reran experiments with training and test redraw/refitting. Current summary exports should not be described as byte-identical duplicates of those historical results. |

The six full-LWC files are `tscp_lwc_laplace.csv`, `tscp_lwc_laplace_30sample.csv`, `tscp_lwc_laplace_10sample.csv`, and the three corresponding `exact_ratio_lwc_*` files. Coverage is intentionally limited: broad Laplace has full LWC only at d=2 and n_cal=30,50,100,300,500 (200 trials each); the n_cal=30 dimension view adds d=3,4 (30 trials each); the n_cal=10 view has d=2,3,4,5,6 (10 trials each). No full-LWC rows exist for d=5,10,20,30,50 at n_cal=30 or for broad d=10. The exporter emits only methods actually available, so the presence of twelve files in a view does not imply a complete method-by-configuration grid.

The two repeated configurations are `4e11385a5254cc2a` (Laplace d=2, n_cal=30: all twelve methods appear in both views) and `336c0699d8acb5d9` (Laplace d=10, n_cal=30: ten methods appear in both). These produce the 22 repeated summaries.

## Units and summary conventions

`export_fresh_tables.py` lines 39–64 merge the three owning source families and retain the first record for each method and trial. The synthetic CSV convention is **outcome volume divided by 2^d**, and **maximum coordinate length divided by 2**. Means, sample standard deviations (ddof=1), medians of trial-wise maximum lengths, runtime means, trial counts, and all configuration metadata were independently reconciled. A numerical difference by a factor 2^d between these CSVs and the main archive is therefore a unit conversion, not a separate result. For full LWC, reported volume and coverage refer to the union; reported coordinate lengths describe its enclosing box, as recorded by `run_auxiliary.py` lines 119–128.

The current exports contain 200 repetitions per broad-noise configuration, 30 or 200 repetitions in the n_cal=30 dimension view, and 10 repetitions in the n_cal=10 dimension view. The file suffix describes calibration sample size, not a uniform repetition count.

The JSON companion contains every checked filename, size, configuration ID, source family, view mapping, overlap, and comparison count. Retaining this 182-KiB convenience folder is inexpensive and keeps the current plotting notebook paths working; archiving it would require deliberately updating those readers.
