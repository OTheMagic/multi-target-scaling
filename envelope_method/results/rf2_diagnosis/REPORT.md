# Why envelope TSCP loses to Point CHR on rf2

The evidence points to an interaction between unusual residual tails in rf2 and TSCP's mean/standard-deviation shape. Tightening the envelope approximation alone is unlikely to close the gap. These are post-hoc diagnostics on the same 200 saved fitted splits, not new fitted experiments.

## Magnitude and reproducibility

| Method | Mean joint coverage | Mean full outcome volume |
|---|---:|---:|
| Envelope TSCP | 0.901917 | 124,336 |
| Point CHR | 0.901497 | 17,881 |
| Old shortcut | 0.901943 | 124,473 |
| GWC | 0.902308 | 125,701 |

Envelope/CHR ratio of means is 6.954, median within-split ratio 2.764, and geometric mean within-split ratio 3.620. CHR is smaller in 166/200 splits. The paired coverage difference is +0.042 percentage points, with a descriptive Monte Carlo 95% interval of [-0.222, +0.306] percentage points conditional on this fixed dataset. Means accentuate extreme-volume splits, but the median and win frequency show that the gap is not only a few extreme means.

Point CHR's bounds, coverage and volume were reproduced on all 200 splits. Its two halves contain 192 residual vectors each, so its use of the first-half conformal rank for the second half is correct here. An algebraically equivalent implementation also matched: let b_j be the first-half marginal conformal 90% residual quantile; then u_j=b_j Q_0.9(max_k r_k/b_k) using the second half. The reference coordinate cancels when b_j are positive. Five inner split seeds give mean CHR volumes 17,881–23,754 and coverage 0.90139–0.90352. The advantage is not confined to seed 42.

## Dataset evidence

The fitted cohort has 7,679 complete rows, 576 features, eight outputs, and 384 calibration rows per split, using a shared multi-output random forest fit within each trial.

NASI2's target median is 3.3, 99th percentile 5.48, and maximum 78.9. The maximum occurs at complete-case positional row 3660 (original retained dataframe index 4782). Adjacent retained target values are 3.56, 3.53, 3.53, **78.9**, 3.54, 3.51, 3.51. Its current NASI2 feature is 3.66. This warrants checking original provenance; it is not proof of a data error and no observations were removed or changed.

| Location of that row | Splits | Mean envelope volume | Mean CHR volume | Ratio of means |
|---|---:|---:|---:|---:|
| Calibration | 6 | 1,168,685 | 4,636 | 252.10 |
| Training | 158 | 108,035 | 19,169 | 5.64 |
| Test | 36 | 21,820 | 14,434 | 1.51 |

When that row is in calibration, its absolute residual is about 75.36; mean NASI2 residual standard deviation is 3.840 while its mean 90th percentile stays 0.0409. When the row is in test, those calibration statistics are 0.0276 and 0.0415. These placement groups are observational diagnostics, not a controlled removal/refit experiment. Training-set contamination may also affect predictions: nearby retained rows with targets near 3.5 have calibration errors as large as 19.08 in some fits. Establishing causation requires controlled refitting.

## Coordinate evidence

| Coordinate | Mean envelope length | Mean CHR length | Geometric mean paired length ratio |
|---|---:|---:|---:|
| NASI2 | 1.687 | 0.2035 | 2.664 |
| NAPM7 | 4.488 | 2.595 | 1.565 |
| SCLM7 | 6.510 | 4.814 | 1.329 |
| CHSI2 | 8.855 | 7.901 | 1.110 |
| EADM7 | 9.216 | 9.558 | 0.962 |
| DLDI4 | 3.644 | 4.088 | 0.893 |
| CLKM7 | 6.219 | 7.435 | 0.841 |
| VALI2 | 1.650 | 2.037 | 0.815 |

For NASI2, the largest four calibration residual deviations (roughly 1%) account for 70.85% of squared deviations on average. Its mean marginal coverage is 98.89% for envelope and 98.21% for CHR. Envelope spends extra width on the most tail-sensitive coordinates while being narrower elsewhere. Products across eight coordinates magnify these shape differences. The product of geometric paired coordinate ratios equals the geometric paired volume ratio, not the ratio of mean volumes.

## Controlled shape comparisons

These methods use the same first/second calibration halves and the same held-out test data. Only the fitted score shape changes.

| Shape | Mean coverage | Mean volume |
|---|---:|---:|
| Split mean + SD times calibrated threshold | 0.901481 | 140,897 |
| Split SD times calibrated threshold, no centering | 0.901543 | 233,595 |
| Split median-residual proportions | 0.901995 | 41,752 |
| Split 90th-percentile proportions (Point CHR) | 0.901497 | 17,881 |

This supports a shape explanation even when splitting is held fixed. Merely removing mean centering does not solve it. A pooled mean/SD plug-in diagnostic, without envelope correction, still has volume 115,137 and coverage 0.900124; it is not asserted to be a valid conformal method. It shows that removing this correction does not by itself explain the roughly sevenfold gap. Envelope improves on GWC by 1.09% and on the old shortcut by 0.11% in mean volume.

## Implications

1. Investigate the provenance of NASI2=78.9 before deciding whether any data correction is justified. A controlled refit with a clearly labeled sensitivity dataset could separate training influence from calibration influence; it must not replace the primary unchanged benchmark silently.
2. Test a robust, independently fitted residual shape. Simply replacing the mean/SD in the existing candidate-updated envelope formula does not inherit its proof: the formula was derived for those statistics. A split-trained quantile scale gives a clean initial comparator; an envelope extension needs its own mathematical justification.
3. Keep joint coverage, paired volume ratios, volume medians and coordinate lengths in the report. Coverage guarantees alone do not optimize rectangle volume or promise dominance over CHR.
4. The feature/target names contain lags and 48H horizons. Random splitting is the current protocol. Whether chronological evaluation is required should be established from dataset provenance; dependence or leakage was not tested here and does not explain the paired shape ablation by itself.

Sources: `diagnosis.json`, `raw_target_check.json`, `paired_trials.csv`, `coordinate_trials.csv`, `chr_seed_trials.csv`; original cache `reviewer_update/real_diagnostics/cache/rf2/`; implementations `utility/envelope.py` and `utility/data_splitting.py`. Scripts: `envelope_method/diagnose_rf2.py`, `envelope_method/rf2_raw_check.py`.
