# rf2: one-row deletion with 200 refitted comparisons

The specified observation was removed from every split, wherever it occurred; all 200 random forests were refitted. Original data/results are unchanged. The remaining split assignments, model configuration, and model seeds are preserved. Each method shares the reduced fit/calibration/test data. Target joint coverage is 90%.

| Dataset | Method | Mean coverage | Coverage SD | Mean full volume | Median full volume |
|---|---|---:|---:|---:|---:|
| Original | Envelope | 0.901917 | 0.016578 | 124,335.91 | 33,391.59 |
| Original | Point_CHR | 0.901497 | 0.021664 | 17,880.99 | 10,637.90 |
| One row removed | Envelope | 0.900831 | 0.016285 | 20,339.73 | 15,656.44 |
| One row removed | Point_CHR | 0.901199 | 0.021749 | 12,495.02 | 8,328.96 |

After deletion: ratio of mean volumes = 1.6278; median paired volume ratio = 1.5884; envelope smaller in 53/200 splits.

## Is it an outlier?

NASI2 target=78.9 at retained position 3660, original dataframe index 4782, original ARFF line 5371. The next largest target is 5.62, median 3.3, and MAD 0.07. Its modified z-score, 0.67449*(value-median)/MAD, is 728.45. The upper outer IQR fence is 4.53; 314 observations exceed that fence, so a boxplot flag alone is not unique evidence. The exceptional separation from the next largest observation provides additional evidence. Nearby original rows have NASI2 targets 3.56, 3.53, 3.53, 78.9, 3.54, 3.51, 3.51. Source verification confirms the value is present in the original ARFF, not introduced by the cache. It is a clear statistical outlier, but a recording error cannot be established from these data alone.

No exact 78.9 values occur in NASI2-prefixed input features. Equal numbers in other target-scale feature families are not evidence of copies of this observation.

## Remaining coordinate shape differences

| Target | Mean envelope length | Mean CHR length | Geometric paired width ratio | Mean largest-four variance share |
|---|---:|---:|---:|---:|
| CHSI2_48H__0 | 9.0639 | 7.7258 | 1.1600 | 55.8% |
| CLKM7_48H__0 | 6.3504 | 7.2328 | 0.8809 | 34.8% |
| DLDI4_48H__0 | 3.7091 | 3.9875 | 0.9296 | 40.2% |
| EADM7_48H__0 | 9.4515 | 9.3626 | 1.0052 | 45.8% |
| NAPM7_48H__0 | 4.5928 | 2.5812 | 1.6141 | 66.8% |
| NASI2_48H__0 | 0.1819 | 0.1908 | 0.9116 | 37.8% |
| SCLM7_48H__0 | 6.6811 | 4.7165 | 1.3912 | 62.3% |
| VALI2_48H__0 | 1.6841 | 1.9730 | 0.8562 | 32.2% |

These remaining differences measure residual shape after removing the one specified row. They do not prove that remaining extreme residuals are errors or justify further deletions.

## Verification and interpretation

All 200 original-source hashes and reduced split exclusions verified; all 400 saved method metrics independently recomputed from binary bounds/test residuals.

Calibration has 383 observations in the six trials where the removed row was in calibration, otherwise 384. Point CHR uses the correct conformal order statistic for each half separately (191/192 for those six cases). This avoids the existing helper's assumption that the halves have equal sizes. Training size is reduced by one in 158 trials and test size by one in 36 trials. These are matched deletion comparisons rather than a new random repartition of all remaining rows.

Coverage Monte Carlo intervals and paired before/after differences are saved in summary.json. They describe variability over random splits conditional on this modified dataset. Removing a value identified after inspecting outcomes makes this a sensitivity analysis; it does not establish coverage for the original population or verify that the observation should be removed in the primary benchmark.

Sources: metadata.json, source_verification.json, summary.json, trials.csv, and 200 trial JSON/NPZ pairs in this directory.