# Paper experiment plan and KEEP list

Date: 2026-09-09. This is a proposed paper selection and completion plan, not an instruction to delete, move, rerun, or overwrite research artifacts. All existing data and code remain in place. The exact configurations below were checked against `envelope_method/settings.json` and current result tables. See `docs/EXPERIMENT_RETENTION.md` for the preservation map.

## The scientific story

The paper's working goal is **simple coordinate standardization for interpretable simultaneous prediction rectangles, using the calibration data without an extra split, with competitive coverage and volume**. The experiments should explain when this is useful, how much calibration data the procedure saves relative to split-based alternatives, and where the mean/SD shape becomes fragile. They should not be reorganized primarily around the incremental envelope-versus-old-shortcut improvement.

Four distinct statements should stay distinct:

1. **Validity:** finite-sample simultaneous marginal coverage follows from exchangeability and a correctly defined, symmetric conformal construction. The newer development says population moments are unnecessary (`envelope_method/signed_envelope.tex`, around lines 529–544); zero-scale, ties, infinite rank and empty-set conventions must accompany the theorem. Heavy-tail simulations illustrate the guarantee, rather than proving it.
2. **Calibration efficiency:** TSCP avoids the additional shape-estimation/recalibration split used by the implemented Point CHR and TSCP-S. This is potentially most useful when the total calibration budget is small. The regression-training split is still present.
3. **Rectangle efficiency and interpretability:** coordinate lengths respond to error scales, often improving volume over a common-width rectangle. This is neither coordinatewise shortening of every interval nor universal dominance over other valid rectangles.
4. **Robustness limits:** finite-population-moment assumptions may be unnecessary for coverage, while useful or stable mean volume can still fail under heavy tails or an influential calibration observation. Mean/SD standardization need not be robust or volume-optimal.

**CQHR is a separate comparison.** With already fitted quantile intervals, the repository's CQHR uses the full calibration sample for one conformal calibration, not the additional 50/50 shape-estimation split used by Point CHR/TSCP-S. A gain against CQHR must be attributed to score/rectangle shape, adaptation and fitted interval quality, not to avoiding a split that CQHR does not use. Confirmed in `utility/data_splitting.py` and `utility/cqhr.py`.

Before the final plots, freeze which implementation the paper calls “TSCP.” Preserve both `TSCP_R` and `Envelope` method keys and make the public name unambiguous. Using the newer construction does not require making its comparison to the old construction the main empirical story. Their same-score comparison belongs in an algorithmic appendix unless the author chooses otherwise.

## Main-paper KEEP selection

Use a small number of grouped displays. The following seven items are scientific questions, not necessarily seven separate full-page figures. Every favorable display retains its unfavorable regimes and coverage context.

| ID | Keep in the main paper | Existing exact source | What the display should establish |
|---|---|---|---|
| M1 | **Homogeneous versus heterogeneous coordinate scales** | `reviewer_update/data/heterogeneity_sweep_{trial,summary,coordinate_trial,coordinate_summary}.csv`; corresponding raw IDs below | Ratio 1 is the honest control; ratios 2/5/10/20 show why a common-width rectangle wastes volume. Pair coverage and volume with one coordinate-length view. |
| M2 | **Small total calibration budget** | `reviewer_update/data/small_calibration_stress_{trial,summary,coordinate_trial,coordinate_summary}.csv`, `small_calibration_stress_infinite_volume_rate.csv`; five raw IDs below | Directly test the central no-extra-split advantage against Point CHR and TSCP-S at fixed total n. The Point CHR infinity at n=10 is a discrete-rank fact; finite n=20/30/50 comparisons must also be shown. TSCP-S backfill is needed on this exact cohort. |
| M3 | **Noise-law and moment stress** | `syn_exps/{gaussian,laplace,gamma,mixed,cauchy,t}/` current fresh CSVs; representative IDs below | Coverage under Gaussian, Laplace, Gamma, Mixed, Student-t and Cauchy. Separate finite-sample validity from efficiency. Use compact d=10, n=30/100/500 summaries, with complete grids in appendix. |
| M4 | **Dependence and partial heteroskedasticity** | `reviewer_update/data/dependent_gaussian_*` at rho=.6 and `partial_heteroskedasticity_*` | Show that marginal coverage and coordinate adaptation survive these settings. State explicitly that global standardization does not estimate conditional variance and does not claim conditional coverage. Keep all dependence strengths in appendix. |
| M5 | **CQR with a serious CQHR comparison** | `envelope_method/results/cqr/` base-alpha .1 five-feature cohort (four IDs below); `cqr_baselines/`; `signed_gain_by_configuration.csv` | Compare on identical fitted base intervals, target/base alpha=.1. Show actual mixed results and learner diagnostics. The completed base-alpha .5/.9 stress examples are sensitivity analyses, not the default evidence of CQHR competitiveness. Prepared cqr10 study supplies a bounded strengthening. |
| M6 | **All eight real datasets in one transparent table** | `envelope_method/results/real_comparison_audited.csv`, `REAL_COMPARISON.md`; caches and corrected-rank sidecars retained | Report coverage, paired efficiency summaries, infinity fraction and dimension, including rf2/Crime losses and Student versus Unscaled. Keep all eight in the overview; qualify random-split temporal data and preprocessing. Full tables/coordinates in appendix. |
| M7 | **One influential-observation limitation** | `envelope_method/results/rf2_standardized_outlier_control/{trials.csv,summary.csv,REPORT.md}`; original/deleted cohorts retained | A concise matched retained/removed sensitivity panel shows that a single observation can greatly inflate the rectangle. Both methods improve; CHR remains smaller. This is a limitation of empirical shape estimation, not a reason to silently clean the primary benchmark. |

Possible page-efficient arrangement: M1+M2 as one central figure; M3+M4 as one robustness figure; M5 as one CQR figure; M6 as a table plus a few representative interval examples; M7 as one small limitation panel. Precise display count can wait until manuscript layout.

## Exact main configuration map

For an absolute configuration ID `ID`, retain **the entire directory** `envelope_method/results/absolute/ID/`, including config, trial JSON/NPZ, status, trials and summary. Keep matching `results/auxiliary/ID/` whenever present. Do not keep only the aggregated CSV: per-trial bounds and paired records are necessary for revised metrics, uncertainty and comparator backfills.

The reviewer absolute cohorts below use d=10, alpha=.1, ten features, 7,200 fresh OLS training and 800 fresh test observations per trial, 200 trials per configuration. The original noise-law grids instead use 6,400 training and 1,600 test observations; **do not pool them as one identical protocol**.

| Main set | Ordered parameter values | Ordered configuration IDs |
|---|---|---|
| M1 scale ratio | 1, 2, 5, 10, 20; n=100 | `45eaa881eb8c2209`, `20f29f9d3c240799`, `9b40adacea0e77c0`, `84d53983e368be01`, `da2a40fa18b278c6` |
| M2 small calibration | n=10,20,30,50,100 | `d87f81a8c98e88c4`, `200d70a07cc79df8`, `7d81a51e343d1fe5`, `98849d48594e6f7b`, `32a6b9d9d00f98f1` |
| M4 dependent Gaussian rho=.6 | n=30,50,100,300,500 | `f4a5ec9302c38c8e`, `5061d053dee84b8b`, `619cf9d39985247a`, `b8209472b686ed1d`, `0cd587ac15230a3d` |
| M4 partial heteroskedasticity | n=30,50,100,300,500; first half of outputs, strength 1.5 | `380c9d403d142300`, `b437ebc4419604b2`, `387f8c372dea3532`, `b6f7b1a50b75a9cd`, `74d35887601b9113` |
| M3 heterogeneous Gaussian | n=30,100,500 | `b682c56d570d4fc4`, `54bff25c4b5f8976`, `3f24bbb092a6548e` |
| M3 homogeneous Gaussian | n=30,100,500 | `089d250dfd930525`, `52be93013ba6fc36`, `ff833dc7a337ed28` |
| M3 Laplace | n=30,100,500 | `336c0699d8acb5d9`, `c48cdf095d54f4f8`, `f9a7e70628c18069` |
| M3 Gamma | n=30,100,500 | `e0b24d936f7b8e3a`, `ebd6c8ab5ffb0696`, `cdad94cb73d7dbac` |
| M3 Mixed | n=30,100,500 | `139172f0f353f58a`, `98edfda1f95cd191`, `067044987bbdd665` |
| M3 Cauchy | n=30,100,500 | `c675a2f311ed1346`, `aa06302f21148403`, `7f238136cc462b22` |
| M3 original unit-scale Student-t | df=1.5,2,3 at n=30 | `92c5ff720401d003`, `03c93ff4a0a731e3`, `b3ad77b36a7f47a2` |
| M3 original unit-scale Student-t | df=1.5,2,3 at n=500 | `479e8557793279ec`, `eadacd9e788dea0f`, `0fdb2103ef5c3c25` |
| M5 CQR, base alpha=.1 | n=30,50,100,200; d=3, five features, 100 trials/config | `d62d892adec04062`, `928d1456b39f8c8f`, `1e011016d30adf10`, `04a924ad4b07dcaf` |

The CQR IDs use `results/cqr/ID/` and matching `results/cqr_baselines/ID/`, training 2,400/test 600, original 100-tree quantile boosting. They are completed, paired results with imperfect original model-quality evidence, not the stronger cqr10 fits. `results/cqr_paired_*`, `signed_gain_by_configuration.csv` and `signed_vs_capped_*` retain different comparison estimands; use their denominators explicitly.

**Actual noise-law audit: do not infer the DGP solely from `noise_levels`.** In `utility/data_generator.py:58-63`, Student-t and Cauchy use `standard_t`/`standard_cauchy` without multiplying by the supplied coordinate scale. The outer generator adds this noise directly. Therefore the original t/Cauchy grids above have **common unit-scale noise**, even where their configuration lists `(10,...,1)`. This is also visible in `results/absolute/92c5ff720401d003/trial_000.npz`: recomputed training-noise coordinate median absolute values range about .849–.901, rather than following 10 through 1. Keep these as valid saved unit-scale experiments, and correct their interpretation; do not silently relabel them as fresh heterogeneous runs.

Gamma uses `shape=target_index` and the configured scale (`data_generator.py:52-53`), so the first coordinate has shape 0 and zero simulated additive noise. The saved first-coordinate noise in `e0b24d936f7b8e3a/trial_000.npz` is zero to floating-point precision. This is a coordinate-dependent-shape experiment with a noiseless outcome, not a common positive-shape Gamma distribution rescaled across outcomes. Preserve it, explicitly state its DGP, and use a new separately labeled nondegenerate Gamma cohort if the paper needs a clean common-shape law comparison. Existing Gaussian and Laplace branches do apply coordinate scale.

For M3, Cauchy is Student-t with df=1 and supplies an existing infinite-first-moment case. The df=1.5 and 2 cases have finite first moment but infinite variance; df=3 has finite variance. These facts apply to the population, while a realized finite calibration sample can still have finite empirical means/SDs. Do not plot a “population mean/SD oracle” as a defined valid benchmark where those population moments do not exist. Keep its diagnostic status rather than manufacture an oracle normalization.

## Appendix KEEP selection

| Appendix family | Exact retained location | Purpose and qualification |
|---|---|---|
| Complete original noise laws/dimensions/calibration grids | All `syn_exps/` fresh tables and matching `results/absolute/` / `auxiliary/` IDs in `settings.json` | Full d=2/10 and calibration curves; dimension stress through d=50. Not every curve needs main-paper space. |
| Homogeneous Student-t | `reviewer_update/data/heavy_tail_stress_*`; IDs `2de59f88339a4504`, `ccb901fa786b999a`, `45c8c2bc743f6372`, `8c18c31879d5f852`, `ef3910ccf31ce8fd`, `ee5dff295f8147f9` | df=1.5/2/3 × n=30/500, 200 trials each; a second unit-scale t cohort with different training/test sizes; the original t generator also ignores the configured scale vector. |
| Higher-df t | `syn_exps/t/` df=10/30/100 rows | Shows transition toward lighter tails; retain all raw IDs mapped by configurations.csv. |
| Dependence strength | `reviewer_update/data/dependent_gaussian_*`; rho=0,.3,.6,.9 | Support cross-output dependence statements beyond the selected rho=.6 example. |
| Target alpha | `reviewer_update/data/alpha_sensitivity_*` | alpha=.05,.1,.2 at n=100. |
| Contamination | `reviewer_update/data/contamination_stress_*`; IDs `ce58cb1d05e133ac`, `d9a64e430c639f07`, `c86c501f006d2b90`, `1b5dae3499baaf54` | Fractions 0/.01/.05/.1 and multiplier 10; useful broad sensitivity, different from a controlled single calibration-point injection. |
| Other CQR bases/shifts | `reviewer_exps/cqr/`, `reviewer_update/data/cqr_*`, all 46 saved CQR configs and comparator sidecars | Bases .3/.5/.7/.9, shifts100/300/1,000, feature-count variants; label model quality and score representation. |
| Gaussian / signed fitted toys | `results/toys/`, `toy_baselines/`, `toy_summary.csv`, `signed_gain_by_toy.csv` | Seven standard Gaussian settings, conservative 98%-base and misspecified-width examples, four positive-score toys. Preserve counterexamples to uniform signed/CQHR superiority. |
| All real coordinates/searches | `reviewer_update/real_diagnostics/`; `results/real/`; corrected full table | Every coordinate, branch counts, alpha stress, uncertainty and runtime. Old-TSCP binary search evidence is not a theorem about global envelope search. |
| rf2 control details | `results/rf2_diagnosis/`, `rf2_remove_one/`, `rf2_remaining/`, `rf2_standardized_outlier_control/` | Per-coordinate tail diagnosis, original vs standardized models, split-seed controls, common-test comparisons and time-order failure. The time-order result must remain disclosed if the main paper uses this temporal dataset. |
| Crime largest-community sensitivity | `results/real_outlier_screen/`, especially `crime_control/` | Mean-ratio reversal is driven by rare tail splits; interval crosses one, 100/200 wins after standardized removal. Supporting limitation, not the leading success case. |
| Shape-template rectangle baseline | `reviewer_update/data/shape_template_{standard,baseline}_*` | Existing low-dimensional comparator; retain setup/runtime/shape-budget constraints. |
| Envelope algorithmic refinement | `envelope_method/report_revision/`, `absolute_paired_*`, verification/rank-search evidence | Same-score containment and serial runtime support a computational refinement. Do not substitute these for the paper's no-extra-split baseline comparisons. |

## Missing work, in priority order

These are planned actions only. Existing protocols, data and outputs stay unchanged during this selection step.

### P0 — Fix the evidence chain before interpreting a final paper

- Freeze one primary TSCP implementation/name, exact comparator definitions, volume units and trial-level estimands. Preserve current method keys so reports can be regenerated without ambiguous relabeling.
- Freeze actual noise laws from the generator and archived observations, not only configuration labels. Correct t/Cauchy scale claims and document Gamma's zero-shape first outcome before assigning main-figure captions. These are interpretation/design gaps; old saved cohorts must remain unchanged.
- Regenerate submission figures/tables from fresh cohorts and corrected Point CHR outputs. `real_comparison_audited.csv` is corrected; generic `real_summary.csv` still has historical Air/Crime CHR numbers. Existing figures are not automatically current merely because their source CSV was refreshed.
- Decide the inductive real-data protocol. Air globally imputes covariates before splitting; Crime globally selects columns by missingness. Fit preprocessing on training rows for a clean inductive benchmark and rerun only affected fits, or explicitly justify the existing transductive covariate protocol. Preserve both versions and label them. Keep temporal data as random-split benchmarks unless a forecasting protocol is added.

### P1 — Directly test the claimed calibration-budget advantage

- **Backfill TSCP-S on the five M2 saved-score cohorts.** No `results/auxiliary/ID/trials.csv` exists for these five IDs as of this audit; the base records contain Envelope, TSCP_R, GWC, Unscaled, Point CHR and Empirical Copula. TSCP-S exists in original synthetic baselines, but those fits are not interchangeable with this 7,200/800 cohort. Evaluate on the exact saved calibration/test residuals, saving source hashes and sidecars. No refitting or fresh simulated data are needed.
- Add a prespecified small set of inner-split seeds to Point CHR/TSCP-S on M2. Compare fixed total calibration n, average over the same trials, and show variation due to the extra split. This isolates the actual statistical tradeoff more directly than an extra collection of noise laws.
- Keep n=10 infinity as a boundary illustration, but center the finite comparison on n=20/30/50/100. A 90% conformal quantile needs at least 9 scores to be finite; splitting can put a method below this rank threshold. The claim should not rest solely on a deliberately infeasible baseline sample size.
- Optional within this bounded ablation: compare TSCP-S split fractions .25/.5/.75 using the same total n, with all fractions fixed before looking at test performance. Report the full grid or a training/validation-selected fraction; never choose the best fraction on each test set.

### P2 — Make CQHR competitiveness convincing with good base models

- Inspect/report the quality of existing d=3 base intervals. The strongest-looking historical CQHR gaps at base alpha=.5 are not a satisfactory default comparison if they are driven by poor fitted intervals. Use the common base alpha=.1 M5 results, including small losses, as the current primary evidence.
- Run the prepared **base-alpha .1 cqr10 subset first**: five inputs, n=30/50/100/200,100 fresh fitted trials each. Exact planned IDs: `296a36117ca3a00c`, `dcc3f95a9377b2b5`, `f59a2c04fb02ceb7`, `8155fb1743a1477c`. Its saved projection is about 3.01 serial hours and 1.54 GB. This is unrun, not completed evidence. Preserve the entire prepared 22-setting/1,430-trial notebook and 6 pilots; full 11.78-hour sweep can follow if base-alpha sensitivity is needed. Do not launch automatically.
- Add a focused real quantile-regression benchmark because the existing eight real cohorts contain absolute residuals, not signed TSCP/CQHR. A concrete candidate panel is Energy (2 outputs), Student (3) and SCM1d (16), selected by dimension/domain rather than expected winner. Use common trained base intervals and fixed splits; audit SCM temporal interpretation. Start with 50 independent model/split replications per dataset and a prespecified precision/stopping rule for extending to 100; never extend only favorable outcomes. If this small study is not done, limit real-data claims to absolute-score TSCP.
- Include one independently specified nonlinear or misspecified DGP with fresh fitted quantile models and training-only diagnostics. A ten-output linear-initialized model on a linear Gaussian DGP is useful but does not by itself establish model-agnostic practical superiority.

### P3 — Sharpen the limitation and distributional coverage story

- Existing Cauchy and t(1.5)/t(2) results already cover infinite population moments. Additional t sweeps are not required merely to illustrate moment-free validity. Missing full homogeneous/heterogeneous crosses for every law are a gap only for claims about efficiency uniformly across that factorial design; the theorem does not require an exhaustive factorial experiment.
- **Genuinely heterogeneous t/Cauchy is missing despite heterogeneous-looking configuration labels.** If homogeneous-versus-heterogeneous infinite-moment behavior is part of the intended paper claim, promote this to P1: add a separately versioned correctly scaled t cohort with d=10, scales 10 through 1, df=1/1.5/2/3 and n=30/500, 200 fresh trials per setting (eight configurations/1,600 trials, training 6,400/test 1,600). Retain the existing unit-scale controls, use the same within-trial fits across methods, and save explicit noise arrays/quantile diagnostics. Never overwrite existing config IDs or infer that rerunning the unchanged old generator tests this question. A clean positive-common-shape Gamma cohort is a separate small design correction if needed for the main noise-law comparison.
- If the claim specifically says **one calibration outlier** can drive the shape problem, add a small controlled injection sensitivity on saved scores: choose the coordinate, row and multipliers in advance, perturb calibration only, and retain the unchanged test distribution. Describe it as off-protocol contamination robustness, not an exchangeable finite-sample coverage experiment. The existing rf2 retained/deleted matched fits already support the broad real-data limitation; no need to delete additional observations.
- Do not tune robust mean/SD caps on current test outcomes. Any claim that a robust replacement fixes the limitation requires new held-out validation. A split-trained robust comparator remains useful scientifically but forfeits the same no-extra-split benefit; a transductively robust extension needs a separate derivation.

## Metrics and reporting rules

- Use simultaneous test coverage and across-trial uncertainty as the first panel/table columns. Conditional-on-dataset random-split uncertainty is not uncertainty over independent datasets. Never pool test points across shared fitted trials as independent replications.
- Choose full outcome-space volume for unified reporting. Convert old absolute residual-box volume by 2^d; CQR uses actual base length plus twice the adjustment. Do not compare raw numerical volumes across datasets with different units/dimensions.
- Primary relative-efficiency summary: paired log-volume differences or geometric paired ratios, plus median and ratio of mean volumes. Include finite-positive-pair counts, zero/empty frequency and infinite frequency. Report both robust typical-case summaries and means in heavy-tail/outlier settings; they answer different questions.
- Put coordinate lengths and marginal coverage beside a small number of examples. “Interpretable” should mean the user sees one interval per outcome, not a claim that every coordinate is narrower or equally covered.
- Report construction time separately from common model fitting. A single representative serial benchmark against Point CHR, TSCP-S, Unscaled and the chosen TSCP implementation is enough; use the exact same score arrays and include setup/fallback. Existing envelope-vs-old timings alone do not establish the central practical baseline comparison.
- Report the homogeneous case, rf2/Crime losses, signed/CQHR counterexamples and temporal limitations. Competitive behavior across a documented range is a defensible target; universal best volume is not the paper's goal.

## Explicitly not required for this paper plan

Further full-LWC enumeration; a universal robust envelope; conditional-coverage guarantees; a complete forecasting method; a giant homogeneous/heterogeneous cross of every noise law; broad neural model sweeps; and additional hand-selected favorable examples are not automatic prerequisites. Keep their existing exploratory artifacts without promoting them to main evidence or deleting them. Stop expanding once the central comparisons, model-quality checks and evidence chain support the stated claim.


