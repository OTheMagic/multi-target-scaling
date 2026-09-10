# rf2: outlier impact with standardized training in both conditions

200 matched splits per condition. Retained-outlier fits are new; removed-outlier standardized fits reuse the previously completed 200 trials. Training target means/SDs are estimated separately from each training partition; the forest settings and random seed are identical. Predictions are transformed back to original outcome units before calibration.

| Treatment | Method | Coverage | Coverage SD | Mean full volume | Volume SD | Median full volume |
|---|---|---:|---:|---:|---:|---:|
| Retained | Envelope | 90.052% | 1.715 pp | 101,671.0 | 246,082.3 | 24,189.5 |
| Retained | Point_CHR | 90.091% | 2.366 pp | 16,019.1 | 23,978.3 | 8,790.5 |
| Removed | Envelope | 90.068% | 1.679 pp | 14,691.5 | 16,512.7 | 10,197.1 |
| Removed | Point_CHR | 90.017% | 2.402 pp | 9,944.2 | 11,837.5 | 5,654.3 |

## Paired effects

| Method | Reduction in mean volume | Median within-split reduction | Coverage change on identical test observations |
|---|---:|---:|---:|
| Envelope | 85.55% | 46.44% | +0.0059 pp |
| Point_CHR | 37.92% | 17.16% | -0.0846 pp |

The envelope/CHR ratio of mean volumes is 6.3469 with the outlier and 1.4774 after removal. Corresponding median within-split ratios are 2.5934 and 1.6272. Envelope is smaller in 45/200 intact splits and 52/200 deleted splits. This is strong outlier sensitivity but not a ranking reversal.

The mean-volume reduction is larger than the median paired reduction because extreme-volume trials receive more weight in the former statistic. Neither is a universal gain. On common test observations, the envelope coverage change has a descriptive 95% Monte Carlo interval of [-0.1305,+0.1422] percentage points. This is variability across splits conditional on the dataset, not an independent population-sampling interval.

## Why training standardization does not eliminate the outlier effect

| Outlier originally in | Trials | Mean envelope volume retained | Mean envelope volume removed |
|---|---:|---:|---:|
| train | 158 | 81,887.1 | 14,890.8 |
| cal | 6 | 1,143,702.3 | 8,621.9 |
| test | 36 | 14,828.6 | 14,828.6 |

Training standardization changes the forest's target weighting. It does not robustify the mean/SD estimated from calibration residuals. If the extreme observation is in calibration, it remains capable of inflating the envelope even when the location model was trained with standardized targets. The test-only deletion leaves training/calibration unchanged, and the before/after bounds reproduce identically in all 36 such splits.

## Verification and scope

All original split-source hashes match. The removed split indices equal the retained indices after excluding row 3660 (original dataframe index 4782). Training transformations were checked against the exact training rows in both conditions. Every new saved method metric was recomputed from binary bounds/test residuals; removed-cohort metrics were independently recomputed from their reused archives. Those source archives are linked by SHA-256 in audit.json.

rf2 remains useful as an outlier-sensitivity example under the stated random-split protocol. Because of its temporal dependence, these results do not establish future-forecast coverage. The separate ordered-split pilot does not invalidate this within-protocol comparison.

The original observation is statistically extreme; removal is an explicitly labeled sensitivity study, not proof that its recorded value is wrong or that data deletion is justified for the primary benchmark.

Sources: trials.csv, summary.csv, audit.json, retained trial JSON/NPZ files, and the reused archives under ../rf2_remaining/model_time/.