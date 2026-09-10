# Outlier sensitivity: standardized rf2 and other existing real cohorts

rf2 shows substantial outlier sensitivity even with standardized training, but no reversal against Point CHR. Crime shows a numerical reversal in mean volume after one fixed deletion, including with standardized training. Crime's result is dominated by rare extreme-volume splits; it does not establish a consistent advantage after cleaning or an erroneous record.

## Controlled rf2 comparison

200 matched splits per treatment, with identical forest settings. Target means and population SDs are fitted on each training partition only, and predictions are converted back to original units before calibration. The retained condition has 200 new fits; the removed condition reuses the previously completed 200 standardized fits. All volumes are products of full interval lengths, at nominal 90% joint coverage.

| Outlier | Method | Joint coverage | Mean full volume |
|---|---|---:|---:|
| Retained | Envelope | 90.052% | 101,671.0 |
| Retained | Point CHR | 90.091% | 16,019.1 |
| Removed | Envelope | 90.068% | 14,691.5 |
| Removed | Point CHR | 90.017% | 9,944.2 |

Removing the NASI2 value 78.9 (next largest 5.62) reduces envelope mean volume by 85.55%, versus 37.92% for Point CHR. Envelope/CHR mean-volume ratio falls from 6.347 to 1.477. The median within-split envelope volume reduction is 46.44%; the larger mean reduction reflects extreme-volume trials. Coverage on the identical remaining test observations changes by +0.0059 percentage points for envelope.

Training standardization improves target weighting in the forest but does not make calibration means and SDs resistant to extreme residuals. The six splits with this observation in calibration have envelope mean volumes 1,143,702 retained versus 8,622 removed. In all 36 test-only deletions the bounds are identical.

rf2 is still a useful outlier-sensitivity example under this random-split protocol. Temporal dependence limits interpreting its coverage as future-forecast coverage; it does not invalidate this controlled comparison. The value is statistically extreme, but a recording error has not been established.

[Detailed rf2 report](../rf2_standardized_outlier_control/REPORT.md) · [rf2 trial data](../../../data/envelope_method/results/rf2_standardized_outlier_control/trials.csv)

## Screening the existing cohorts

The table below uses the current complete-data, original-training benchmark and the corrected Point CHR recalibration rank. Only rf2 and Crime start with larger mean volumes for envelope. This is a descriptive candidate screen; deletion fits were run for rf2 and Crime, not for every cohort.

| Dataset | Envelope / CHR mean volume | Relevance |
|---|---:|---|
| stock | CHR unbounded | Point CHR is unbounded in all 200 small-calibration splits; not a finite losing baseline. |
| rf2 | 6.954 | Controlled extreme-observation study above; no reversal. |
| scm1d | 0.535 | Envelope already has smaller mean volume. |
| scm20d | 0.508 | Envelope already has smaller mean volume. |
| energy | 0.776 | Envelope already has smaller mean volume; building simulation outputs. |
| student | 0.282 | Envelope already has smaller mean volume; student-level records. |
| air | 0.983 | Envelope is slightly smaller after the CHR rank correction; hourly time series. |
| crime | 2081.972 | Community-level records; fixed largest-population removal tested below. |

Dataset descriptions: [UCI Crime](https://archive.ics.uci.edu/dataset/211/communities+and+crime+unnormalized), [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), [UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency), and [UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality). Energy consists of simulated building configurations, so it is not a natural measurement-error example.

## Crime: a qualified reversal

Removal was fixed once, using the maximum input population before examining deletion outcomes: retained position 16, original dataframe index 21, population 7,322,564 versus 3,485,398 next largest. The original 1,902 rows become 1,901. Other train/calibration/test memberships are preserved. All four conditions have 200 splits: 600 new fits plus 200 reused original fits. The same corrected Point CHR formula is used throughout.

| Training targets | Largest community | Method | Joint coverage | Mean full volume | Median full volume |
|---|---|---|---:|---:|---:|
| Original | Retained | Envelope | 90.765% | 1.5207e+66 | 1.3787e+57 |
| Original | Retained | Point CHR | 91.706% | 7.3039e+62 | 1.2466e+57 |
| Original | Removed | Envelope | 90.803% | 5.4056e+62 | 8.1057e+56 |
| Original | Removed | Point CHR | 91.665% | 1.5291e+64 | 5.1778e+56 |
| Standardized | Retained | Envelope | 90.655% | 7.0729e+65 | 1.0860e+57 |
| Standardized | Retained | Point CHR | 91.470% | 1.8317e+64 | 8.2653e+56 |
| Standardized | Removed | Envelope | 90.647% | 5.8670e+62 | 4.7951e+56 |
| Standardized | Removed | Point CHR | 91.562% | 5.7996e+63 | 5.3287e+56 |

| Training targets | Largest community | Ratio of mean volumes | Median paired Env/CHR ratio | Envelope smaller | Split-bootstrap 95% interval for mean ratio |
|---|---|---:|---:|---:|---:|
| Original | Retained | 2082 | 1.104 | 100/200 | [14.3, 1.14e+04] |
| Original | Removed | 0.03535 | 0.906 | 102/200 | [0.00516, 4.29] |
| Standardized | Retained | 38.61 | 1.793 | 94/200 | [0.538, 2.85e+04] |
| Standardized | Removed | 0.1012 | 0.932 | 100/200 | [0.0113, 2.58] |

A ratio below one favors envelope. Under standardized training, removing the community reduces envelope's mean volume by 99.917% and CHR's by 68.34%, reversing the mean-volume ratio from 38.615 to 0.101. Envelope common-test coverage changes by -0.063 percentage points. With original training, the mean-volume ratio also reverses, from 2,081.97 to 0.0354, although Point CHR's mean volume increases after the deletion.

### What drives the reversal

The community occurs in training in 143 splits, calibration in 11, and testing in 46. With standardized training, those 11 calibration splits contribute 99.90% of envelope's total retained-condition volume. Their mean volume falls from 1.285e67 to 1.092e60 on deletion. The five largest retained-envelope trials contribute 99.28% of its total volume. These are empirical signs of rare calibration-tail inflation, amplified by multiplying 18 interval widths.

The advantage is less persuasive for typical splits: after standardized deletion, envelope is smaller in exactly 100/200 splits, the median paired Env/CHR ratio is 0.932, and the geometric mean paired ratio is 1.084. Envelope's median paired volume change from deletion is only a 0.274% reduction. Both methods continue to have very skewed volume distributions.

The 20,000 paired split-bootstrap replicates give a 95% interval [0.0113, 2.58] for the standardized removed-condition ratio of mean volumes. It crosses one. These are descriptive Monte Carlo intervals conditional on this dataset and split protocol, not population confidence intervals; rare unobserved extreme splits remain a limitation. A mean-volume reversal in these 200 splits is observed, but a reliable general superiority claim is not established.

### Is the community an outlier?

It is an influential upper-tail observation in population and several crime counts. Its maximum robust z-score after log1p target transformation is 6.696, compared with 6.405 and 6.120 for the next two observations. This is less isolated than rf2's NASI2 value. The UCI dataset mixes eight counts and ten rates across 18 outcomes; exceptionally large communities can legitimately have exceptionally large counts. The counts and population make a size effect plausible, but this analysis does not verify the record's accuracy or prove a data-entry error. Label this a largest-community sensitivity analysis, not a corrected dataset.

## Interpretation for the experiments

Crime supplies the requested numerical loss-to-win reversal in mean volume, with an explicit qualification that it is driven by rare tails and an influential community, not a confirmed erroneous observation. rf2 supplies clearer evidence of an isolated statistical outlier and strong sensitivity after standardized training, but no reversal. The current experiments therefore support a careful sensitivity argument more strongly than a general claim that removing outliers makes envelope outperform Point CHR.

## Verification and saved data

The fit audit checks all 600 new Crime archives, deletion/index alignment, and training-only transformations. The independent reporting audit recalculates both calibration bounds and every coverage/volume record for all 800 fit conditions (1,600 method records). Original archives preserve historical Point CHR bounds from before the rank correction; their residuals are reused to recompute the corrected comparator throughout this report. All 184 method comparisons across the 46 test-only deletion splits and two training choices reproduce exactly. Dataset and archive SHA-256 hashes are recorded. No source observations or primary benchmark results were overwritten.

[Crime trial data](../../../data/envelope_method/results/real_outlier_screen/crime_control/trials.csv) · [Crime summary](../../../data/envelope_method/results/real_outlier_screen/crime_control/summary.csv) · [Crime diagnostics and archive hashes](../../../data/envelope_method/results/real_outlier_screen/crime_control/diagnostics.json) · [Removed observation](../../../data/envelope_method/results/real_outlier_screen/crime_control/removed_observation.csv) · [Screen summary](../../../data/envelope_method/results/real_outlier_screen/screen_summary.csv)

Reproduction: run `rf2_standardized_outlier_control.py`, `report_rf2_standardized_control.py`, `screen_real_outlier_examples.py`, `crime_outlier_control.py`, then `report_real_outlier_screen.py` from the project root with the configured Python dependencies. The fit scripts reuse existing completed checkpoints.
