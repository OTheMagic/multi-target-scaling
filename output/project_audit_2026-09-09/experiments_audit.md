# Experiments and results audit — 9 September 2026

Read-only review of experiment configuration JSON, actual CSV headers and values, saved audit manifests, runners, and research reports. No experiments were launched, no research files changed, and existing full-archive audits were inspected rather than rerun. This document distinguishes saved evidence from proposed work. Paths are relative to `E:/multi-target-scaling`.

## 1. What exists, without double counting

| Category | Completed evidence | Scientific purpose / qualification |
|---|---|---|
| Primary synthetic + reviewer grids | `settings.json`: 197 unique configurations / 34,230 fresh fitted trials | 175 absolute configurations / 32,800 trials, plus 22 CQR configurations / 1,430 trials. |
| Notebook exploratory configurations | `notebook_settings.json`: 26 configurations / 1,200 trials | Exploratory CQR/coordinate/dependence/CQHR/smoke studies. Shared configurations are mapped to multiple historical study views. |
| Repaired omissions | `repair_settings.json`: 2 configurations / 33 trials | 30 fresh d=50, n_cal=30 Laplace trials and an explicitly specified three-trial smoke replication. |
| Unique combined synthetic corpus | **225 configurations / 35,463 trials** | 179 absolute / 33,233 trials and 46 CQR / 2,230 trials. Counts independently recomputed from the three JSON files and `absolute_paired_summary.csv`. |
| Fitted empirical toys | **13 studies / 1,730 trials** | Seven Gaussian CQR toys ×50, conservative signed toy ×500, misspecified-width toy ×80, four positive-score toys ×200. Separate from the 35,463. |
| September 9 standalone report suite | **20 settings / 2,400 fresh fitted trials** | 1,440 primary six-output trials, 720 target-alpha sensitivity trials, 240 two-output contraction trials. Separate new report cohort. |
| Real-data benchmark | **8 datasets ×200 splits =1,600 fits/splits** | Six cached cohorts also evaluated at alpha=.1/.3/.5/.7/.9; Air/Crime at .1. Additional alpha evaluations are not additional fits. |
| Real search diagnostics | 6 datasets ×200 original diagnostic refits | These are the same six caches reused by the envelope real benchmark; do not add them again to the 1,600. They evaluate the old TSCP search implementation and every coordinate. |
| rf2 follow-ups | Original paired diagnostics; 200 one-row-deleted fits; 200 deleted standardized fits; 200 retained standardized fits; adaptive-scale and ordered-pilot work | Many tables reuse the same fits. Useful causal controls and failure analysis, not independent extra datasets. |
| Crime follow-up | Four conditions ×200 splits, **600 new fits +200 original reused fits** | Original/standardized target training crossed with retained/removed largest community. |
| Ten-output CQR extension | **Six complete pilots /120 quantile models; formal study unrun** | Prepared notebook, tests, saved models, cost projections. Core 22 settings /1,430 trials; optional four settings /260 trials. |
| Full LWC | Some completed small legacy/fresh replications retained; one signed illustration deferred | Manual-only notebook, disabled by default. No further full-LWC work should be treated as a prerequisite by default. |

Evidence: `envelope_method/results/final_audit.json:1`, `results/audit.json:1`, `results/configurations.csv:1`, `results/toy_summary.csv:1`, `report_revision/design.json:1`, `report_revision/verification.json:1`, `cqr10/README.md:7`. `comparator_rerun_audit.json` checks 3,402 method/scenario entries (3,401 complete, one deferred), which include duplicate historical views; this is not 3,402 studies.

## 2. Synthetic experiment families and models

The main absolute-score runner fits OLS afresh in each trial; the CQR runner fits lower/upper coordinatewise gradient-boosted quantile models. All methods within a trial share the same data and fitted predictions. The generator keeps DGP coefficients fixed across trials, while observations and fitted models are redrawn; this is a legitimate fixed-DGP replication design. Some configurations share random seeds/common random numbers across settings, so totals should not be presented as fully independent across configurations.

The main CQR default is 100 trees, depth 5, learning rate .05, minimum leaf size 5. These training settings are materially different from the stronger cqr10 model and from the simple OLS/group-quantile model used in the standalone report. Source: `envelope_method/experiments.py:33`, `:177`, `:185`, `:192`.

| Family | Grid actually present | Typical repetitions / role |
|---|---|---|
| Original noise-law comparisons | Gaussian (homogeneous and heterogeneous), Laplace, Cauchy, Gamma, Student-t, Mixed; usually d=2,10 | Usually 200/config; n_cal=30,50,100,300,500; selected Cauchy n=1,000. Student-t degrees-of-freedom sweep at n=30,500. |
| Dimension/small-calibration stress | Laplace d=2,3,4,5,10,20,30,50 at n_cal=30; d=2,…,6 at n=10 | Mixed 10/30/200-count historical views, with actual counts retained explicitly. Some small full-LWC comparisons separate. |
| Residual heterogeneity | Gaussian d=10, n=100; scale ratio 1,2,5,10,20 | 200/config; a direct test of why coordinate adaptation matters. |
| Dependent errors | Gaussian d=10, five calibration sizes and four dependence settings | 20 configurations /4,000 trials in each relevant source view; some views share configurations. |
| Alpha sensitivity | Gaussian d=10, n=100, alpha=.05,.1,.2 | 200/config. |
| Small-calibration stress | Gaussian d=10, n=10,20,30,50,100 | 200/config; infinite and fallback rates are relevant outcomes. |
| Heavy tails / contamination | Student-t stress; four Gaussian contamination configurations | 200/config; the existing corpus already investigates non-Gaussian tails. |
| Partial heteroskedasticity | Gaussian d=10 at five calibration sizes | 200/config; mean models are OLS, so covariate-dependent scale adaptation remains limited. |
| Shape-template comparator | Gaussian d=2, n=30,50,100,200 | 100/config; already-fresh deterministic paired cohort, external region-design comparator. |
| Primary CQR | d=3; five-input 100-trial grid and ten-input 30-trial grid; n=30,50,100,200 | Training 2,400/test600; base alpha .1,.3,.5,.7,.9; target .1. Capped scores and shifts 100,300,1,000 compared separately. |
| Notebook pilots | d=2,3,4 quantile sweeps and isolated coordinate/dependence checks | 10–200 trials/config; exploratory settings, not substitutes for broad model-quality validation. |

The ordinary comparator family is substantial: Envelope, old TSCP_R shortcut, TSCP_GWC, TSCP_S, Unscaled Max, Empirical Copula, Point CHR, Naive, Bonferroni, Population Oracle; CQR adds Signed GWC, CQHR, raw/capped/shifted representations and the uncalibrated base. Baseline-only omitted rows were explicitly repaired. The archived comparator audit is broad, but does not establish that the set is sufficient against every current published competitor.

## 3. What the synthetic evidence says

**Scaling helps when output error scales differ, but it is not universally efficient.** In the fresh reviewer heterogeneity table (d=10,n_cal=100,200 OLS trials), old TSCP_R has mean residual-box volume 17,959.8 versus Unscaled 14,260.2 at scale ratio1; at ratio20, TSCP_R is 5.747e10 versus Unscaled 3.438e15. Both retain roughly 90% coverage. Exact coverage equality for scale-adaptive methods across this sweep is consistent with scale equivariance/common random numbers, not five independent confirmations of a novel phenomenon. Source: `reviewer_update/data/heterogeneity_sweep_summary.csv`.

**The envelope improvement over the corrected shortcut is mathematically clean but frequently modest.** Across 179 absolute configurations /33,233 trials, no coordinate-containment violations were found. Recomputing `absolute_paired_summary.csv` gives a median configuration mean paired ratio .9866786, equally weighted mean .9620993, smallest .6512913. Thus the median configuration gains about 1.33%; the equal-configuration average gains 3.79%. Neither is a universal gain or a raw pooled-volume ratio. Source: `envelope_method/RESULTS.md:14`.

The new 2,400-trial report makes this clearer on a controlled grid: absolute-score mean paired reductions averaged over four laws are 8.13% at n=30,1.03% at n=80,.27% at n=200. At n=80, capped-score reductions are .19–.46%. Across the primary grid, 98/480 n=30 capped old-wrapper trials are infinite; larger n have none. These infinite fallbacks cannot be converted to finite artificial plotting constants or silently dropped from coverage. Sources: `report_revision/report_main.tex:205`, `:237`; `table_n80.csv`; `evidence.json`.

**Signed scores add useful flexibility but do not dominate capped scores or CQHR.** At base/target alpha=.1, d=3, five features,100 trials/config, signed-envelope mean paired reductions versus old capped shortcut are 4.23%,1.63%,.85%,.11% at n=30,50,100,200. Against CQHR they are -3.82%,-1.89%,+.57%,+.68%. Signed coverage is .9003–.9092. A conservative 98% base-interval toy shows meaningful contraction, but is a different scientific regime from a 90% base. The misspecified-width toy gives a 9.17% mean paired CQHR reduction, whereas its ratio of mean volumes implies17.4%; both can be correct and must not be interchanged. Source: `envelope_method/FOLLOWUP_AUDIT.md:11` and corresponding `signed_gain_by_configuration.csv`/`signed_gain_by_toy.csv`.

The six-output report has larger signed/capped differences in some settings, and explicit counterexamples to signed containment (largest observed signed/capped ratio2.081). These differences should be explained through dimensionality, fitted interval quality, conservatism and zero-variance capping, rather than presented as interchangeable replications of the three-output gradient-boosted grid. Source: `report_revision/report_main.tex:333`.

**Runtime evidence exists and is qualified.** The controlled serial report benchmark uses n=30/80/200/500/1,000, d=2/6/12,15 fresh fitted arrays/cell, randomized method order, one warmup and three timed calls. Absolute backward envelope is faster than old shortcut in all15 tested cells (old/envelope median ratio1.09–2.27); capped ratios .66–1.67 include losses. Region construction excludes model fitting/I/O/test evaluation. Rank search is equivalent but may cost extra when backward search inspects one cell. Uncontrolled concurrent workload timings should not be mixed into this benchmark. Source: `report_revision/README.md:44`, `report_revision/report_main.tex:408`, `runtime_summary.csv`.

## 4. Real data: what we actually have

Rows/features/outcomes below come from saved cache metadata, after loader preprocessing. All eight main cohorts use 200 hash-seeded random 75%/5%/20% splits. Stock uses MultiTaskLasso(alpha=.0001); the other seven use 200-tree random forests, with a training pipeline for student categorical features. rf2/SCM/Air/Crime use depth16,sqrt features,min leaf2; student/energy forests are unrestricted-depth variants.

| Dataset | Rows | Input features | Outcomes | Calibration rows | Envelope 90%-target coverage | Envelope / Point CHR mean full volume |
|---|---:|---:|---:|---:|---:|---:|
| Stock portfolio |315|6|6|15|.95164|CHR infinite in200/200|
| River flow rf2 |7,679|576|8|384|.90192|6.954|
| SCM1d |9,803|280|16|490|.90146|.535|
| SCM20d |8,966|61|16|448|.90110|.508|
| Energy |768|8|2|38|.92117|.776|
| Student |649|30|3|32|.90912|.282|
| Air quality |6,941|8|4|347|.90351|.983|
| Crime |1,902|102|18|95|.90765|2,081.972|

Canonical current values: `envelope_method/results/real_comparison_audited.csv`. These are **ratios of mean full outcome volumes**, not paired mean ratios, and their stability differs dramatically by dataset. The envelope is also worse than Unscaled on Student (1,555.9 versus1,205.5 mean full volume); competing rectangles need not all shrink coordinatewise. Empirical Copula has small volumes accompanied by important undercoverage, e.g.Stock .7269,Student .8611,Crime .8624, and should not be called an equally valid efficiency winner.

Six-cohort search diagnostics exist for all51 coordinates. At alpha=.1 every one of10,200 old-TSCP coordinate searches used its binary branch. More extreme alpha settings exercise backward/fallback branches. This is empirical branch evidence, not proof of a global binary-search predicate, and is not a complexity measurement for the new envelope. Source: `reviewer_update/real_diagnostics/README.md:9`.

## 5. The rf2 and Crime investigations already answer important questions

**rf2 loss is principally a shape/learning/tail issue, not loose envelope approximation.** Original envelope mean full volume124,336 versus Point CHR17,881, with coverage .90192/.90150; tightening GWC saves only about1.09% and tightening old shortcut .11%. Extreme NASI2 residuals inflate empirical SDs and distort eight-dimensional products. Same-split robust shape comparisons support this diagnosis.

The completed controlled target-standardized retained/deleted comparison changes envelope mean volume101,671→14,692 (85.55% reduction), but CHR also improves16,019→9,944. The mean-volume ratio falls6.347→1.477 without a ranking reversal. All36 test-only deletions preserve bounds. The unusual NASI2 value78.9 is not proven to be a recording error. Source: `results/rf2_standardized_outlier_control/REPORT.md:3`.

The ordered-gap stress pilot is much more consequential for application claims: five matched row pools yield shuffled coverage90.21%/91.66% versus ordered15.78%/12.64% (Envelope/CHR). These five overlapping folds are diagnostics, not independent future-forecast estimates, but every ordered fold is badly below target. The data have exact 48-row lag identities and strong temporal dependence. Do not present random-split results as demonstrated48-step-ahead forecasting coverage. Source: `results/rf2_remaining/REPORT.md:38`.

Conditional diagnostics find envelope90.07% overall but80.84% on an input-defined high-change regime. One adaptive-scale attempt raises high-change coverage84.58% while increasing mean volume sharply; it is not yet an efficiency solution. Complete-case filtering discards15.85% of original rf2 rows. Robust shape validation, time-aware evaluation and training-only imputation remain meaningful research directions. Source: `results/rf2_remaining/REPORT.md:51`, `:69`.

**Crime gives a fragile mean-volume reversal, not robust evidence that deletion makes envelope best.** Standardized retained/removed ratio of means38.615→.1012, but11/200 calibration-tail splits contribute99.90% of the retained envelope's volume. After removal, envelope wins100/200 splits, geometric mean ratio1.084, and descriptive split-bootstrap interval[.0113,2.58] crosses one. The maximum-population community is influential and plausible, not a confirmed error. Source: `results/real_outlier_screen/REPORT.md:41`.

## 6. Current versus superseded artifacts: a substantive publication task

1. `quarantine/obsolete_synthetic_2026-09-08/` contains known/suspected obsolete synthetic cohorts, derived tables and notebook displays. It is archival evidence, never current empirical input. The new runners redraw/refit; do not demand that the project rerun all synthetic work again merely because old copies exist.
2. `syn_exps/`, `reviewer_exps/`, `reviewer_update/data/` now contain notebook-compatible fresh exports; their current headers include `redraw_train_test`, training/test sizes, repetitions and often configuration IDs. `envelope_method/results/fresh_exports.json` maps exports. Counts may differ from smaller old views when one shared configuration now exports the full cohort.
3. `reviewer_update/pre_*`, old manuscript tables/figures/PDFs and earlier standalone-report copies remain historical snapshots. A present filename outside quarantine does not by itself establish that its rendered content was regenerated after the migration. The main manuscript requires an explicit results-source integration pass.
4. **Confirmed mismatch:** `envelope_method/results/real_summary.csv:4` retains Air Point CHR coverage .8970122 and `:10` retains Crime .8945144. Corrected canonical `real_comparison_audited.csv:4`/`:10` gives .9029806/.9170604 and larger volumes after the second-half conformal-rank fix. `REAL_COMPARISON.md` is corrected, while generic consolidation still reads preserved historical baseline records. This is a provenance/synchronization problem, not evidence that the corrected result is missing.
5. Volume conventions differ: synthetic notebook tables often use the positive residual-box volume; envelope result tables use full outcome volume, larger by2^d for absolute residuals. Signed CQR uses base length plus twice the adjustment. Trial averaging, mean paired ratios, ratio of means, logarithmic volume and infinite-pair conditioning all answer different questions.

Sources: `envelope_method/README.md:118`, `RESULTS.md:74`, `FOLLOWUP_AUDIT.md:113`, `results/rf2_remaining/REPORT.md:63`, `summarize.py:165`.

## 7. What still needs accomplishing or understanding before a strong submission

**Essential scientific positioning and analysis, rather than indiscriminate more experiments:**

- Decide the paper's central empirical claim: valid finite-sample adaptive rectangles and a provably tighter computable construction are well supported; universal superiority over CHR/CQHR, broad signed shrinkage, and future-forecast performance are not.
- Build one claim-to-evidence map: each theorem/comparison gets exact score representation, learner, data regime, metric, uncertainty unit and figure source. Integrate the fresh cohorts and corrected CHR values into one canonical submission tree.
- Explain the scaling advantage, envelope increment and signed-score increment separately. Large differences against Unscaled in heterogeneous settings do not measure the envelope increment; conservative-base contraction does not establish gains at base alpha .1.
- Audit inductive preprocessing. Air's loader runs `SimpleImputer.fit_transform` on all feature rows before splitting (`utility/exps.py:1791`), using held-out covariates. Crime selects columns by whole-dataset missingness before splitting. No held-out label fitting is evident from these lines, but a clean inductive evaluation should fit imputation/selection using training only, or clearly justify the protocol. Air also removes time columns, so application claims require a temporal protocol decision.
- Make model quality an explicit factor. The cqr10 pilot demonstrates severe legacy quantile underfitting in a diagnostic case; comparisons remain paired but may mostly reflect poor initial intervals. Complete the prepared cqr10 study if high-output signed/CQHR claims are central, and add a focused nonlinear/misspecified learner check rather than simply expanding the same linear Gaussian grid indefinitely.
- **Real signed-CQR/CQHR evaluation is absent from the eight-cohort benchmark.** Its saved methods are absolute-residual Envelope/TSCP_R/GWC/Unscaled/Point CHR/Empirical Copula; `run_real.py` and `run_extra_real.py` evaluate absolute residuals. The richer signed/CQHR comparisons are synthetic/toy evidence. If the submission presents signed CQR as an important practical contribution, add a small frozen real-data quantile-regression benchmark with common well-trained base intervals, signed/capped envelope, CQHR and valid comparators. This gap does not invalidate the existing absolute-residual results.
- Validate robustness findings on new held-out trials/conditions. Choosing winsorization thresholds after seeing these test outcomes is exploratory. Replacing candidate-updated means/SDs by robust statistics does not inherit the envelope proof; a split-trained robust comparator is straightforward, a new envelope requires new mathematics.
- Resolve temporal interpretation for rf2 and inspect other temporal data (Air/SCM) before using forecasting language. Existing random-split studies can remain honestly labeled interpolation/protocol benchmarks.
- Show uncertainty over trials/splits, finite/infinite fractions, paired medians/geometric means and coordinate summaries. For heavy-tailed18-dimensional Crime, mean volume alone is unstable. “Target lies within one standard deviation” is descriptive and does not validate coverage or constitute a confidence interval for the mean.
- Freeze a reproducible public artifact with configs, data provenance, hashes, versions, lightweight verification and figure generation. Archive audit passes are strong supporting assets; reproducibility must include a portable clean environment and unambiguous authoritative outputs.

**Optional extensions, not automatically required:** more full LWC, more omnibus sweeps, robust candidate-updated envelope theory, conditional guarantees, broad forecasting adaptation, and adaptive-scale redesign. Each could be a separate research project. The prepared ten-output core study is a concrete bounded extension (estimated11.78 serial hours,5.6GB;9–19hour planning band); it should be selected because it tests the central claim, not because the existing experiment count is small.

A minimal next experimental package, if signed CQR is central, is therefore: finish the frozen cqr10 core; add a small real signed/CQHR study; run one independently specified nonlinear/misspecified model-quality check; and validate any robustness claim on new held-out cases. Clean preprocessing reruns are needed for any main benchmark whose protocol is changed. A full new forecasting study is necessary only for forecasting claims, while the existing failures should already be disclosed. No additional full-LWC run is required merely to complete this package.

