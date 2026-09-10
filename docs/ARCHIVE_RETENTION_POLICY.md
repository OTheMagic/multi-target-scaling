# Archive retention policy — September 9, 2026

> September 10 layout update: numerical files referenced here now live under `data/` with the same repository-relative suffix. See [data layout](DATA_LAYOUT.md). The author removed `deletable/` before this reorganization; historical rollback copies are no longer local.

This policy implements the authorized cleanup of existing primary absolute-score and CQR archives. It specifies which full per-trial NPZ files retain their original observations; it does not select which experimental results enter the paper. Every trial retains its saved scores, bounds, existing non-X/y arrays, and compact JSON/CSV numerical results. The migration is performed separately by the cleanup runner.

Machine-readable policy: [cleanup/retention_policy.json](cleanup/retention_policy.json). Entries are expanded explicitly for all 225 current configuration IDs.

## Scope and retained data

The default for `envelope_method/results/absolute/` and `envelope_method/results/cqr/` is **scores mode**. For non-exception trials, remove only `X_train`, `y_train`, `X_cal`, `y_cal`, `X_test`, and `y_test` from NPZ files. Preserve every other member, including calibration/test residuals, all method bounds, CQR base lengths, model/DGP coefficients where present, and any future or unlisted member. All per-trial JSONs, configuration maps and CSV summaries remain.

Scores remain for **all trials**, including configurations needed for unfinished TSCP-S or other comparator work. This policy does not discard experiments, repetitions, extreme outcomes, empty regions, or infinite-volume trials.

Keep all toys, all real-data and outlier/control data, CQR10 pilots, the separate September 9 report-revision data, auxiliary baselines, comparator sidecars, and completed full-LWC results unchanged. These locations are outside this migration scope.

## Full-cohort exceptions

Selection uses design and an existing documented demonstration, not measured gains or coverage. All trials of each chosen cohort are kept. The signed-witness cohort is retained in full rather than selecting only a favorable or exceptional trial.

| Kind / config ID | Retained trials | Design and purpose | Original full-NPZ bytes |
|---|---:|---|---:|
| `absolute/898faf279901d34c` | all 200 | Cauchy, d=2, n_cal=30, n_train=6400, n_test=1600 | 153,896,321 |
| `absolute/c7ab3684ea5e6952` | all 200 | Gaussian, d=2, n_cal=30, n_train=6400, n_test=1600 | 153,843,437 |
| `absolute/4e11385a5254cc2a` | all 200 | Laplace, d=2, n_cal=30, n_train=6400, n_test=1600 | 153,878,041 |
| `cqr/1e011016d30adf10` | all 100 | Gaussian, d=3, n_cal=100, n_train=2400, n_test=600, base/target alpha=0.1/0.1 | 23,929,577 |
| `cqr/d62d892adec04062` | all 100 | Gaussian, d=3, n_cal=30, n_train=2400, n_test=600, base/target alpha=0.1/0.1 | 23,219,131 |
| `cqr/928d1456b39f8c8f` | all 100 | Gaussian, d=3, n_cal=50, n_train=2400, n_test=600, base/target alpha=0.1/0.1 | 23,394,803 |
| `cqr/04a924ad4b07dcaf` | all 100 | Gaussian, d=3, n_cal=200, n_train=2400, n_test=600, base/target alpha=0.1/0.1 | 25,008,904 |
| `cqr/8527e223b376ca56` | all 100 | Gaussian, d=3, n_cal=30, n_train=1600, n_test=400, base/target alpha=0.1/0.1 | 17,133,617 |
| `absolute/3d3354a844bc2064` | all 3 | Gaussian, d=2, n_cal=12, n_train=80, n_test=20 | 22,867 |

The three main absolute cohorts form a complete same-size clean Gaussian / Laplace / heavy-tail Cauchy comparison at d=2 and n_cal=30. The Laplace cohort also supplies paired inputs for the already saved small full-LWC comparison. The three-trial n_cal=12 Gaussian smoke replication is retained because its small declared design is useful for end-to-end reproduction; it remains a labeled replication, not an exact replay of undocumented historical fit parameters.

The four primary CQR cohorts retain the full n_cal=30/50/100/200 comparison under a common fitted-model setup and base/target alpha=0.1. The additional CQR cohort `8527e223b376ca56` preserves the already documented mixed-sign example in `envelope_method/sign_witnesses.json`: trial 66, test index 0. That witness was found retrospectively; retaining all 100 cohort trials does not turn it into a randomly selected demonstration or a performance guarantee.

For **every other configuration**, retain fixed **trial 0 in full**, selected without reading its outcomes. This preserves at least one raw-input/generator/model audit example at every current dimension, calibration size, noise setting and score-model setup. It also preserves the first-trial formula-replay inputs used by the old final audit. Trial 0 is not a statistically representative replacement for all raw trial inputs.

## Size and coverage of the retention selection

The selection keeps **1,319 full NPZ files** (778 absolute and 541 CQR), totaling **796,388,868 bytes (0.742 GiB)** before migration. Nine whole cohorts are retained; the other 216 configurations retain trial 0. These are original full-file sizes, not an estimate of bytes removed by the migration. All 35,463 primary fitted trials keep their method-evaluation scores and already-computed results.

All untouched toy, real/control, report-revision and CQR10 data are additional to that full-NPZ subtotal. They are retained regardless of the selected primary-cohort list.

## What can and cannot be checked after compaction

Scores and saved bounds permit recalculation of existing score-based methods and coverage/volume summaries, including paired comparator additions that use those scores. The existing CQR base lengths permit the saved outcome-length/volume calculations. Removing original X/y observations prevents direct rechecking of those observations and the predictor fit for compacted trials without regeneration. It also prevents experiments requiring new feature-dependent predictors, a new input-based score map, or new outcome transformations from being performed solely from the compact archive.

This is a deliberate change in archival reproducibility, not a claim that full raw-data verification remains possible for every compacted trial. The prior full-data audit stays a historical record; the new validator must distinguish full and scores modes. `envelope_method/final_audit.py:37` expects X/y arrays and cannot run unchanged on scores archives. `envelope_method/experiments.py:172` currently resumes from JSON alone, so resume validation must be updated to detect archive mode/completeness rather than silently assuming raw data remain.

Before replacing each archive, record old and new checksums, the six removed members and their shape/dtype/hash metadata, and confirm exact equality of every retained NPZ member. Preserve generator source and environment metadata. Member hashes establish what was removed, but do not reconstruct a deleted array. Original complete-file hashes must not be compared to new compact-file bytes as if unchanged.

## Generator and design caveats

Retention records existing data faithfully; it is not scientific sign-off on every intended simulation design. In the current `utility/data_generator.py:46`–`:63`, the Cauchy and Student-t noise branches draw standard distributions without using the passed `scale`, and Gamma uses the zero-based target index as its shape (the first coordinate has shape zero). Configuration labels such as `noise_levels=[2,1]` must therefore not be read as proof of the actual Cauchy/t scale heterogeneity. Full retention of the complete Cauchy cohort and trial 0 of every other generator configuration preserves concrete audit inputs; broader corrections/reruns remain a separate research decision.

Training/test/calibration observations are redrawn and models refitted within each trial, but primary data seeds depend on dimension, calibration size and trial rather than every configuration field (`experiments.py:178`–`:206`). Some configurations consequently share identical raw arrays. This can support paired comparisons across configurations; it does not make configurations independently sampled studies, nor justify deleting their distinct method outcomes.

Seeds plus source/environment records can support regeneration. They do not alone establish byte-identical regenerated observations, fits, runtimes, or original NPZ checksums under a changed environment. The policy therefore keeps coherent full cohorts and fixed-index raw audit examples while honestly labeling the rest as scores archives.

## Non-selection rules

- Never use observed volume, coverage, gain, influence, runtime or a successful assertion to choose which trials retain full data.
- Do not trim scores or numerical trial records while stripping X/y.
- Do not alter the outlier datasets, report-revision demonstrations, toys or CQR10 pilots in this pass.
- Do not relabel current full-data audit results as validations of a different compact format.
- Do not silently rebuild or rerun experiments as part of archive retention.
