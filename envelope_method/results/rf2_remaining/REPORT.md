# rf2: further investigation after one-row removal

Completed 200 target-normalized location-model refits, 200 adaptive-scale fits, 10 matched-size ordered/shuffled model fits, 10 additional row-order pilots, five inner-split seeds for four static score shapes on all 200 cleaned cohorts, and implementation/metric checks. All primary comparisons here use the one-row-deleted dataset from the previous sensitivity study. No further rows were deleted.

## 1. Remaining large errors are spread over many rows

The top-four calibration residuals span 207 distinct rows for NAPM7, 220 for SCLM7, and 246 for CHSI2 across the 200 splits. These are repeated appearances in calibration, not 800 independent outliers. Raw target maxima lie in multi-row plateaus: NAPM7 reaches 128 with neighbors 127–128; SCLM7 reaches 187 with neighbors 186–187. These do not resemble the isolated NASI2=78.9 spike. Large residuals are not sufficient evidence of erroneous observations.

The four largest squared deviations account for about 67% of NAPM7 calibration variance and 62% of SCLM7 variance on average. The residual tail shape continues to distort standard-deviation-based rectangle proportions.

## 2. Robust shape estimation addresses much of the remaining shape problem

Same location-model fits, same 192/192 (or 191/192) calibration split, seed 42. Winsorization caps only the first-half residuals used to estimate mean and SD. The second-half calibration scores and test observations are untouched.

| Shape/method | Coverage | Mean full volume |
|---|---:|---:|
| Envelope on all calibration rows | 90.083% | 20,340 |
| Split mean/SD | 89.979% | 27,428 |
| Split mean/SD with 99th-percentile cap | 90.037% | 18,287 |
| Split mean/SD with 95th-percentile cap | 90.178% | 12,985 |
| Point CHR quantile shape | 90.120% | 12,495 |

Across five fixed inner-split seeds, CHR mean volumes range 12,495–14,948 and 95%-capped mean/SD volumes 12,985–15,091. The remaining gap is not a special seed-42 accident. This robust split-score construction is an exploratory comparator, not a proven drop-in replacement for the candidate-updated envelope statistics. Tuning the cap using these test results would require new held-out validation.

## 3. Multi-output model target weighting also matters

The raw multi-output squared-error objective is sensitive to target units. In the cleaned dataset, CHSI2 and EADM7 together account for 68.5% of total target variance; NASI2 accounts for about 0.0021%. These variance shares are descriptive, not the realized tree-split impurity weights. Standardizing each target using its training-partition mean/SD, fitting the same shared random forest, and inverting the transform improves both methods.

| Location model | Method | Coverage | Mean full volume |
|---|---|---:|---:|
| Original target units | Envelope | 90.083% | 20,340 |
| Original target units | Point_CHR | 90.120% | 12,495 |
| Training-target standardized | Envelope | 90.068% | 14,692 |
| Training-target standardized | Point_CHR | 90.017% | 9,944 |

This is a 200-split paired model ablation. It does not make envelope dominate CHR. Post-fit coordinate rescaling is a different operation: envelope is equivariant to it, and the tested rescalings do not alter regions after conversion back.

## 4. Random splitting does not establish time-forward forecasting performance

In the local complete-case data, all 7,137 available pairs match each target to the corresponding input 48 original rows later. Lag-one input correlations are 0.9965–0.9998. On average about 94% of randomly held-out rows have a training row within one original row index, and effectively all within 48. These are overlapping temporal observations, even though the row IDs themselves are disjoint.

The full ARFF has a sequence discontinuity near row 4108, consistent with the published train/test split boundary. To avoid interpreting a concatenation boundary as continuous time, the matched pilot uses only the second sequence (original index >=4108). It has five expanding-history folds, training sizes 1000/1400/1800/2200/2600, 384 calibration rows, and 1024 test rows. Ordered conditions leave gaps of more than 48 original row indices between partitions. Shuffled conditions use exactly the same per-fold row pools and sizes. Absolute timestamps are unavailable; time order is inferred from the exact lag identities.

| Matched pilot assignment | Envelope coverage | Point CHR coverage |
|---|---:|---:|
| Shuffled | 90.21% | 91.66% |
| Ordered with gaps | 15.78% | 12.64% |

These are averages of five overlapping stress-test folds, not independent estimates of general forecasting accuracy. Nevertheless every ordered fold is far below target. In fold 3, every NAPM7 test target is below the training target range; its calibration 90th-percentile error is 2.11 and test 90th-percentile error 11.49. Both methods have zero NAPM7 marginal coverage there. Distribution shift and the forest's inability to extrapolate beyond its training target range explain this failure more directly than numerical envelope slack. The original random-split benchmark remains a valid description of that interpolation protocol; it must not be presented as demonstrated 48-step-ahead coverage.

## 5. Marginal 90% coverage hides a difficult observable regime

On the training-target-standardized fits, define change risk from the maximum coordinate-standardized difference between current flow inputs and their 48-row lag. Standardization and the 80th-percentile threshold use only the location training partition. Test cases are grouped using their inputs, never their targets.

| Method | All test cases | High-change cases | Ordinary-change cases |
|---|---:|---:|---:|
| Envelope_full | 90.07% | 80.84% | 92.38% |
| Point_CHR | 90.02% | 80.54% | 92.40% |
| Envelope_adaptive | 90.19% | 84.58% | 91.61% |

The adaptive model fits coordinate log-residual scales on only the first calibration half, using a fixed 100-tree ExtraTrees model with depth 4/minimum leaf 20 and the first 64 lagged-flow inputs. Envelope calibration uses the untouched second half. It improves high-change coverage to 84.58% but mean volume increases to 55,413, compared with 20,378 for static envelope using that same second half and 14,692 for the full-calibration static envelope. This first adaptive model is not a useful efficiency improvement. The regime imbalance is not a contradiction of a marginal coverage guarantee.

## 6. A real CHR rank bug, fixed and checked

The helper used the first calibration half's conformal rank for the second half. With an odd calibration count the halves differ by one, so the second rank can be too small. `utility/data_splitting.py` now computes its own rank. All 200 original rf2 results are unchanged, and all 200 previously reported reduced-rf2 bounds match the corrected helper: the deletion runner had already handled this correctly.

The same correction affects Air and Crime, whose calibration sizes are 347 and 95. Their Point CHR coverages change from 89.70% to 90.30%, and 89.45% to 91.71%, respectively; volumes increase. The current `results/REAL_COMPARISON.md` and CSV are refreshed. Historical archives remain unchanged. None of the 225 configurations in settings.json/notebook_settings.json/repair_settings.json has a finite odd-half rank mismatch; this scope check does not rewrite old notebook outputs.

## 7. Missingness changes the evaluated population

The [Mulan catalogue](https://mulan.sourceforge.net/datasets-mtr.html) lists 9,125 rf2 rows. The repository loader drops 1,446 (15.85%) because of missing input features; none have missing targets. Its benchmark therefore evaluates 7,679 complete cases before the one-row sensitivity removal. Missingness handling is another protocol choice worth testing with training-only imputation.

The NASI2=78.9 value also exists as a future input in original row 4830, which is omitted by complete-case filtering because other features are missing. This explains why no copy appeared in the previously inspected retained NASI2 input features. It supports consistency of the stored lag construction but does not establish whether the original measurement was correct.

## Verification and next work

Exhaustive envelope-cell search, coordinate permutation, and coordinate-rescaling checks match on ten cleaned splits each. This is not a full-LWC computation. Saved metrics were independently recomputed for the 210 location-model refits and 200 adaptive-scale fits; the matched blocked pools and worst-coordinate failure metrics were checked. The CHR rank correction passed 400 comparisons against independent saved rf2 outputs.

Priority: establish a time-aware evaluation protocol for forecasting claims; validate robust shape estimation on new held-out trials; investigate training-only imputation and stronger extrapolation-aware location/scale models. Further deleting large residuals would not address the demonstrated distribution shift or score-shape problem.

Evidence: diagnostics.json, shape_summary.csv, tail_row_summary.csv, raw_structure.json, model_time/, blocked_pilot/, conditional/, and chr_rank_fix_verification.json.