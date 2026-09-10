# Project inventory and JMLR assessment

**Audit date: September 9, 2026, Pacific time.**

The project has a substantial methods paper, a newer theoretical extension, a large body of freshly fitted experiments, and unusually detailed diagnostic evidence. It is **not yet a synchronized submission package**. The most important remaining work is to settle the final contribution, bring the theorem, implementation, results, manuscript and reviewer response into agreement, and explain the regimes in which the method helps or struggles. More experiment volume alone will not resolve those issues.

My recommended scientific center is **transductive coordinate standardization with the surface-envelope construction**, with the original shortcut as a clearly defined predecessor/comparator and signed CQR as an extension. This is a recommendation about the strongest current research story, not a claim about the editor's preferred scope or an acceptance prediction.

## Scope and navigation

Every file in the folder was enumerated, including hidden files, installed dependencies, archives and Git metadata. Content review covered authored source, active and historical manuscript materials, experiment drivers/settings, saved summaries/manifests, diagnostic reports and targeted executable checks. It did not reread every raw trial array, rerun the full experiment corpus, independently prove every theorem, or repeat the earlier page-by-page figure QA. Existing full-archive verification records were inspected as saved evidence and distinguished from checks run for this audit.

Only this audit directory was created. Research code, observations, results, notebooks and manuscript files were left unchanged.

Read this report first, then use:

- [Exact folder census](FOLDER_INVENTORY.md): top-level and second-level counts, sizes and largest files.
- [Theory and manuscript audit](theory_audit.md): theorem scope, corrected versus unresolved issues, current source-line references.
- [Experiment audit](experiments_audit.md): settings, datasets, numerical findings, provenance and remaining empirical questions.
- [Code and reproducibility audit](code_audit.md): implementation map, checks actually run, release risks and file-line evidence.
- [Inventory summary](../../data/output/project_audit_2026-09-09/inventory_summary.json) and [complete file index, compressed](../../data/output/project_audit_2026-09-09/file_inventory.json.gz): machine-readable snapshot. The redundant uncompressed copy was staged during the subsequent cleanup; the gzip retains exactly the same data. The index includes paths, sizes and modification times; it is not a new content-hash audit of every file.
- [Document metadata](../../data/output/project_audit_2026-09-09/document_metadata.json): observed lengths and sizes of the six inspected PDFs.

## 1. What physically exists

Excluding Git internals and this new audit directory, the folder contains **232,697 files totaling 47,469,412,634 bytes: 47.47 GB, or approximately 44.21 GiB**. Git metadata adds 1,375 files and 0.86 GB. There were no file-enumeration errors. Counts are a snapshot, not counts of independent scientific results.

| Category/location | Files | What it contains | How to treat it |
|---|---:|---|---|
| `utility/` | 31 | 12 Python modules plus bytecode; shared methods, residuals, data generators, experiment and graphing helpers | Core implementation; `envelope.py` is the new method |
| Root files | 12 | README, nine notebooks, extracted paper text, Git attributes | Entry points; README describes mainly the older generation |
| `multi_target_scaling_latex/` | 169 | Main JMLR source, proofs, bibliography/style, experiment prose, embedded tables/figures, PDFs, response/cover and audits | Current manuscript working folder, but its evidence and status notes are partly historical |
| `syn_exps/` | 96 | Notebook-compatible CSV exports in Gaussian, Laplace, gamma, Cauchy, mixed and Student-t families | Fresh replacements now exist; this does not refresh embedded manuscript figures automatically |
| `real_exps/` | 12 | Eight historical result CSVs, four local ARFF data files | Historical baseline interface and source datasets; newer corrected comparisons live elsewhere |
| `reviewer_exps/` | 32 | Absolute-residual and CQR trial/coordinate/summary exports | Reviewer-study interface; includes fresh exports |
| `reviewer_update/` | 1,720 | Experiment/build/validation scripts, 1,200 cached real fits, data/figures, QA and six `pre_*` snapshots | Mix of useful reproducibility machinery and preserved revision stages |
| `envelope_method/` | 128,514 | New method note/report, drivers/settings, raw trial archives, summaries, comparator audits, real-data investigations, cqr10 pilots | Main home of current scientific development; 43.09 GB, mostly reproducibility data |
| `output/` before this audit | 61 | Portable envelope and outlier reports, source/data bundles and ZIPs | Convenient reading/sharing copies, not a complete experiment release |
| `quarantine/` | 87,058 | Superseded synthetic evidence and notebook outputs with manifests/reasons | Historical provenance; excluded from fresh runner inputs |
| `tmp/` | 14,978 | Installed scientific packages, TeX compiler/cache, render images, old compile trees, boundary probes | Mostly infrastructure/intermediates; not thousands of additional research contributions |
| Root bytecode/test cache | 14 | Python bytecode and pytest cache | Generated artifacts |

The Git snapshot contains **129 modified tracked files, 28 deleted tracked paths and 215,697 untracked files**, excluding this audit. Much of the new study and archive material therefore has no committed version in this checkout. That does not tell us whether an external backup exists, but it does mean the public reproducibility release cannot simply be assumed to match this working directory.

There are six historical manuscript snapshots under `reviewer_update/pre_*`. The old readiness reports are useful history; their apparent completion/open labels are not a live dashboard.

## 2. The scientific work, in categories

### A. Problem and original contribution

The problem is to construct one interval per output such that **all outputs are covered simultaneously with probability at least 1-alpha**, averaging over the calibration/test randomness. Rectangles make individual intervals readable. The difficulty is unequal error scales and tail behavior across outputs, especially when calibration data are scarce.

The original TSCP contribution uses symmetric candidate augmentation to estimate coordinate location/scale from calibration data without an extra calibration split. It develops a population oracle, an infeasible transductive/full-conformal reference, GWC and local constructions, then an inexpensive rectangular shortcut. The manuscript has substantive coverage proofs, algorithms and search/complexity analysis. This is already a developed methods project.

The original source currently assumes nonnegative, atom-free residual marginals with positive finite population variances, distinct observed coordinate scores and n>=2. Those assumptions must not silently be carried over to capped CQR or infinite-variance experiments. The newer construction changes the theoretical scope.

### B. New surface-envelope method

The extension optimizes the entire standardized ratio over a candidate domain rather than separately combining loose bounds on its numerator and denominator. Its source contains:

- Signed-score augmentation and inverse formulas, including empty and infinite cases.
- Exact interval suprema, signed GWC and coordinate surface bounds.
- Pathwise containment of the full-conformal reference and finite-sample simultaneous coverage.
- A uniform weak containment theorem relative to the corrected old shortcut on **the same nonnegative scores**, with compatible closed-cell conventions and positive scales.
- Treatment of ties and a conservative whole-domain fallback at zero calibration scale.
- Direct CQR expansion or contraction using the attainable lower score bound at the test input.
- Exact backward search and a proved order-statistic localization option. Worst-case dimension dependence improves from the old O(d²n²) expression to O(dn² + dn log n) in the note's computational model.

Finite population moments are unnecessary for the new coverage argument; finite observed calculations and the stated fallback conventions still matter. The region encloses the full-conformal set and can fill gaps. It is not claimed to equal that set or the full-cell LWC union.

**This development is ahead of the manuscript.** `body.tex:724` still describes direct signed scores as future work. The final article should consolidate these developments rather than paste the separate 20-page report onto the older narrative.

### C. Implemented methods and comparators

| Method/family | Role | Main implementation |
|---|---|---|
| Envelope / signed envelope | Current proposed construction | `utility/envelope.py` |
| GWC, original TSCP-R shortcut, LWC machinery | Predecessor, bound and approximation references | `utility/res_rescaled.py` |
| Naive scaling | Illustrates reuse without a validity correction | `utility/data_splitting.py` |
| TSCP-S | Independent scale/calibration split comparator | `utility/data_splitting.py` |
| Population oracle | Idealized distribution-parameter reference, not a deployable baseline | `utility/data_splitting.py` |
| Point CHR | Point-model rectangular comparator; additional calibration allocation | `utility/data_splitting.py` |
| CQHR | Native quantile-model rectangular comparator | `utility/cqhr.py` |
| Unscaled Max and Bonferroni | Simple simultaneous-coverage baselines | `utility/unscaled.py` |
| Empirical copula | Dependence-oriented empirical comparator | `utility/copula.py` |
| Shape templates | Two-dimensional learned-shape comparison | Reviewer experiment builder and saved shape-template results |
| Rectangle/union and score utilities | Geometry, coverage, volume and conformal ranks | `rectangle.py`, `conformal_utils.py` |

Names need an explicit publication dictionary: “TSCP,” `TSCP_R`, original shortcut, signed GWC, envelope, capped envelope and full LWC are not interchangeable. Similarly, signed-versus-capped is a change of score representation; envelope-versus-shortcut on fixed nonnegative scores is an approximation comparison.

## 3. Experiments already completed

### A. Main simulation corpus

The current saved full-archive audit records **225 configurations and 35,463 freshly fitted trials**. Each trial generates fresh training, calibration and test observations, and refits. Methods share data and predictions within a trial. Multiple study views can refer to one configuration; these are not extra independent experiments.

This corpus includes the original distribution benchmarks, calibration-size and dimensionality variations, dependence, heterogeneous/homogeneous scales, partial heteroskedasticity, contamination, heavy tails, CQR base-alpha and calibration-size grids, shift sensitivity, and notebook/smoke replications. There are **179 absolute-score configurations / 33,233 trials** and **46 CQR configurations / 2,230 trials** in that total.

The drivers/settings preserve different sample counts where the original designs differed; a single claimed sample size cannot describe all experiments. Some settings use common random numbers across configurations, and fixed DGP coefficients are intentionally reused. Fresh trials within a setting should not be confused with independence across every setting or study view. Fresh exports replace many old notebook CSV paths. Quarantined evidence is not an input to these reruns.

### B. Additional fitted toys and standalone report

- **1,730 fitted toy trials** use newly trained centers, with fitted interval widths for quantile toys. They study conservative base intervals, dependence, misspecified widths, positive scores and difficult boundaries.
- A **separate 2,400-trial report suite** uses four noise laws (Gaussian, strongly correlated Gaussian, Laplace, centered lognormal), fitted OLS/groupwise quantile models, a six-output primary grid, target sensitivity and two-output contraction studies. It saves full observations and independently recomputed metrics, plus a serial construction benchmark and two fixed geometry examples.
- The comparator follow-up checks **3,402 historical method/scenario entries**: 3,401 covered by fresh reruns or fitted replications, one signed full-LWC illustration deferred. These entries include duplicate historical views. Comparator sidecars generally evaluate existing fresh fits rather than creating additional independent trials.

These three synthetic corpora account for 39,593 fitted trials before additional timing/quality pilots; this aggregate is useful for scale, not as a measure of diversity or statistical strength. Formula tests and geometry grid points are not empirical fitted trials.

### C. Real-data evidence

Eight cohorts have 200 random splits each, using the 75% training / 5% calibration / 20% test allocation. Six original cohorts have cached reruns and five target-alpha levels. Air and Crime extend the cohort list. Counts below describe the processed data actually used, not necessarily the raw dataset population.

| Cohort | Rows | Features | Outputs | Key qualification |
|---|---:|---:|---:|---|
| stock | 315 | 6 | 6 | Very small calibration allocation; Point CHR can be unbounded |
| rf2 | 7,679 | 576 | 8 | Complete cases; overlapping temporal records |
| scm1d | 9,803 | 280 | 16 | High-dimensional output-volume products |
| scm20d | 8,966 | 61 | 16 | High-dimensional output-volume products |
| energy | 768 | 8 | 2 | Simulated building configurations |
| student | 649 | 30 | 3 | Envelope does not beat Unscaled Max in mean volume |
| air | 6,941 | 8 | 4 | Time series; imputation currently uses full-cohort covariates |
| crime | 1,902 | 102 | 18 | Counts and rates; volume means dominated by rare extreme splits |

The current corrected consolidated real table is **`envelope_method/results/real_comparison_audited.csv`**, accompanied by `REAL_COMPARISON.md`. The similarly named `real_summary.csv` still has older Air/Crime Point CHR results. This is a concrete synchronization issue.

### D. Diagnostic research already done

There is much more than headline coverage/volume plots:

- All 51 coordinates in the six-cohort original real diagnostic study, with marginal coverage and paired widths.
- Search branch/fallback frequency, candidate inspections, serial timing and high-alpha stress tests.
- Original-versus-corrected shortcut boundary/tie witnesses and Point CHR rank checks.
- rf2 residual-tail, raw-record, temporal-overlap, outlier deletion, target-standardization, robust split-shape and adaptive-scale investigations.
- Matched ordered/shuffled rf2 pilots and observable high-change subgroup diagnostics.
- An eight-cohort outlier screen, matched standardized rf2 controls, and four-condition Crime deletion/model-standardization controls.
- Source/archive hashes, metric recomputation, boundary and rank-search equivalence checks.

These diagnostics are valuable research insight. They should inform the limitations and explanations, even where only a compact selection belongs in the paper.

## 4. What the evidence says

### Claims that have substantial support

**Same-score envelope refinement works as intended.** Across the 179 absolute-score configurations, saved records report no containment violations relative to the corrected shortcut. The median configuration's mean paired volume ratio is 0.9867, with a smallest configuration ratio of 0.6513. Thus typical gains are modest, while particular regimes can have larger gains. Equal-weight averaging over configurations is descriptive, not a universal efficiency estimate. See `envelope_method/RESULTS.md` and paired result tables.

**Signed scores solve a real representational restriction, but usually do not produce dramatic gains in the principal CQR grid.** At common base/target miscoverage 0.1 in the three-output study, mean paired reductions versus the old capped shortcut are 4.23%, 1.63%, 0.85% and 0.11% at calibration sizes 30, 50, 100 and 200. The corresponding sign-only reductions versus capped envelope are 3.34%, 1.25%, 0.69% and 0.04%. These are paired ratios, not ratios of pooled mean volumes. See `FOLLOWUP_AUDIT.md`.

**There is a mechanism for shrinkage.** Conservative marginal intervals can leave room to contract while meeting the joint target; ordinary 90% marginal intervals in several independent outputs usually need joint expansion. A mixed-sign coordinate witness shows that one envelope coordinate can contract even when all signed-GWC coordinates expand. Neither fact implies the full box is smaller than its base box in every case.

**CQHR remains competitive.** Standard fitted Gaussian toys show similar or sometimes larger envelope volumes. A misspecified-width toy favors envelope, but that does not justify universal CQHR superiority. A conservative-base toy is useful as a mechanism illustration, with the base alpha explicitly different from the main 0.1 comparison.

**The real-data benchmark is mixed.** Using corrected Point CHR, envelope/CHR ratios of mean volumes are approximately 0.535 on scm1d, 0.508 on scm20d, 0.776 on Energy, 0.282 on Student and 0.983 on Air. Point CHR is unbounded on Stock under its allocated calibration budget. Envelope loses on rf2 (about 6.95) and Crime (about 2,082). These are ratios of means and are sensitive to volume tails. Many envelope gains versus the original shortcut are very small on these real cohorts.

### Limitations that should shape the paper

1. **Outlier sensitivity is not solved by the envelope refinement.** With standardized training on rf2, removing one extreme observation reduces the envelope/CHR mean-volume ratio from 6.347 to 1.477; Point CHR still wins. A better approximation to an SD-based method does not make SD a robust scale estimate.
2. **Crime's apparent reversal is unstable across summaries.** Standardized deletion changes the mean-volume ratio from 38.615 to about 0.101, but envelope wins in only 100/200 splits, and the descriptive paired bootstrap interval for the mean ratio includes one. The record is influential; it is not established to be erroneous. Keep the full-data primary benchmark.
3. **Random-split performance is not forecasting validity.** On the five overlapping ordered-gap rf2 diagnostic folds, envelope/CHR coverage averages 15.78%/12.64%, versus 90.21%/91.66% for matched shuffled assignments. This is a diagnostic of a different, difficult protocol, not a new independent population benchmark or a contradiction of the exchangeability-based theorem.
4. **Marginal validity does not give conditional validity.** The rf2 high-change subgroup has roughly 81% coverage even when aggregate coverage is near 90%. Coordinate coverage balance also does not follow from matching means and variances.
5. **Dimension amplifies small width differences and extreme tails.** Report log-volume or geometric-mean side length, paired ratio distributions, medians, win rates and infinity rates alongside arithmetic means. Raw volumes across datasets with different dimensions and units should not be compared directly.
6. **Model quality can obscure the score comparison.** The ten-output pilots found severe underfitting in the previous quantile-model settings on a low-noise output. That does not by itself invalidate paired comparisons or conformal coverage, but it limits conclusions about well-fitted quantile models.
7. **Runtime claims need a clear boundary.** The newer serial benchmark measures construction, excluding fitting/I/O. Signed domains can require test-input-specific work. Better asymptotic dependence on d is not a guarantee of smaller elapsed time at every n,d or for every score.

## 5. Completed, pending and stale are different states

| Component | Assessment | What remains |
|---|---|---|
| Core research question and original derivation | Developed | Simplify and connect to the current method |
| Signed envelope derivation and implementation | Substantially developed, with explicit checks | Independent proof pass and one final theorem/API contract |
| Main fresh synthetic studies | Completed, with saved provenance audit | Freeze canonical versions and regenerate paper outputs |
| Eight-cohort point-prediction real benchmark | Completed | Use corrected CHR tables; delimit temporal/preprocessing protocol |
| Outlier/temporal/conditional diagnostics | Completed exploratory evidence | Select explanatory findings and preserve honest limitations |
| Higher-dimensional, stronger-model CQR | Prepared, not completed | Run or narrow the claim; six pilots are not formal results |
| Real-data signed CQR versus CQHR | No broad completed benchmark located | High-value addition if direct signed-CQR utility is central |
| Full-cell LWC | Small historical completed cases; one illustration deferred | Manual-only; broad completion is not a default requirement |
| Manuscript figures and numerical prose | Partly stale | Rebuild from frozen current tables, then do the planned presentation pass |
| Reviewer-response statuses and readiness files | Historical snapshots | Reconcile every response with final sources and evidence |
| Portable public release | Incomplete | Environment, entry points, data access, license, versioned artifact and clean reproduction |

The ten-output plan contains **1,430 fresh fitted trials**, estimated from six pilots at **11.78 single-worker hours**, with an engineering allowance of approximately **9–19 hours** and about **5.6 GB** of compressed arrays/models. It uses training-only standardized, linear-initialized quantile boosting on a linear Gaussian DGP. The design is deliberately disabled by default; no formal sweep was run for this audit. Its six pilots support readiness to execute and diagnose those settings, not the empirical conclusions of the unrun study.

## 6. Concrete issues found in this audit

### Manuscript and mathematical consistency

Some old author checks are already fixed: n>=2, positive scales, an atom-free assumption, the mean-index inequality, a search endpoint, final-subset quantile indexing and a quadratic sign. Repeating the old list as wholly open would understate progress.

Current remaining issues include calling a max scalarization “invertible,” half-open cells equated to a closed box, a critical GWC expression undefined at the calibration mean, capped ties/zero scales not fully reconciled with the original theorem, coverage-balance language and qualifications on complexity/test-input reuse. The newer derivation already gives cleaner answers to several of these. The detailed theory audit distinguishes source inconsistencies from missing fundamental results.

Related work still needs correction: Point CHR and CQHR should be described separately, and competing methods cannot be dismissed as necessarily nonrectangular. Sampson and Chan explicitly develop point and quantile hyperrectangles; Tumu et al. explicitly include hyperrectangular templates. A targeted primary-source check confirms those distinctions. This audit is not an exhaustive novelty search through all 2026 literature. [Sampson and Chan](https://onlinelibrary.wiley.com/doi/full/10.1002/sam.11710), [Tumu et al.](https://proceedings.mlr.press/v242/tumu24a/tumu24a.pdf).

### Evidence consistency and code

- A targeted pytest run found a **real stale-figure test failure**: all eight figures listed in `figure_style_audit.json` have some values that differ from current source CSVs, across 448 checked fields. This is an evidence integration failure, not evidence that the envelope algorithm failed.
- The Point CHR odd-half calibration-rank bug is fixed in source, with corrected Air and Crime results. Generic historical summary paths still contain older values. Corrected Air/Crime coverage is approximately 0.9030/0.9171, compared with 0.8970/0.8945 in the stale table.
- Fresh-sampling/tie/zero-boundary checks passed in this audit. The targeted reviewer tests had 67 passes and one stale-figure failure; all eight cqr10 runner tests passed. More details and command outputs are in the code audit.
- Some older Naive/TSCP-S/Point CHR functions can produce NaNs on degenerate constant-score inputs. The envelope has a documented whole-domain fallback. A consistent policy and targeted tests belong in a public implementation.
- Air's feature imputer is fitted before splitting, using full-cohort covariates; Crime's feature selection uses full-cohort missingness. This is not held-out label leakage. A symmetric transductive protocol can preserve exchangeability, but an ordinary inductive pipeline should fit preprocessing on training data and be evaluated accordingly.
- Some generator branches need careful description: the first gamma coordinate uses shape zero, while current Cauchy/Student-t branches do not apply the supplied scale in the same way as other branches. Check intended versus implemented distributions before writing general scale claims.

### Reproducibility packaging

The strongest parts are per-trial seeds, raw archives, hashes, paired fits, explicit sample counts, recomputation audits, and the cqr10 runner's version/corruption checks. The weakest part is the top-level handoff to a new reader: machine-specific paths, no unified pinned environment or top-level package/test configuration, no project license, no CI workflow, and multiple competing result generations.

The repository needs a small reproducible entry point, not a requirement that a reviewer understand 232,697 files. A release should provide code plus a smoke example, a selected-results rebuild path, full experiment configs, a versioned data/artifact archive, and a mapping from every paper table/figure to its source. Local dependency installations and TeX caches should not be part of the research release. Preserve the quarantine separately with its provenance; do not feed it back into current results.

## 7. What JMLR adds to the decision

JMLR's reviewer criteria emphasize a significant, technically correct advance, adequate evaluation, replication detail, proper comparison with prior work, and accessible explanation of strengths and limits. Those criteria fit the kind of method developed here; the current synchronization problems obstruct evaluating it reliably. [JMLR reviewer guide](https://www.jmlr.org/reviewer-guide.html).

Current author instructions require JMLR LaTeX/PDF and a cover letter addressing overlap, coauthor consent, conflicts, suggested action editors/reviewers and keywords. Papers above 50 pages require a length justification; appendices count. The instructions also specify a sub-5-MB upload, an abstract no longer than 200 words, five keywords and a short running title. Preprints are allowed. For a resubmission, confirm the editor's permission and decision-specific instructions. [JMLR author information](https://www.jmlr.org/author-info.html).

Observed local state: the main PDF is **64 pages / 1.29 MB**, the response **13 pages**, and the revision summary **two pages**. The current abstract is approximately **122 words** and five keywords are present. The envelope report is 20 pages; the outlier report is nine. The main manuscript still has placeholder editor/publication metadata, and the revision summary is not evidently a complete submission cover letter. Funding/competing-interest disclosure and author approval need a final check. The folder does not establish the live editorial permission/status.

Do not increase a 64-page submission by appending every new report. Rewrite around one contribution, move detailed diagnostics into well-indexed supplementary material as appropriate, remove repeated motivational/proof/experiment material, and justify whatever length remains. This is editorial judgment based on the observed source structure, not a fixed journal rule about the ideal length of this particular paper.

## 8. Recommended path to submission

| Priority | Work | Concrete completion criterion |
|---|---|---|
| 1 | Choose final scientific scope and names | A one-page contribution statement identifies the final algorithm, guarantee, comparator and claimed practical benefit |
| 2 | Freeze canonical theory and implementations | Every theorem assumption, tie/empty/infinite policy and comparator definition points to the exact implementation; remaining proof concerns resolved |
| 3 | Freeze canonical evidence | One manifest selects corrected fresh result cohorts, units and estimands; every table/figure is regenerated and the stale-figure checks pass |
| 4 | Answer the few remaining empirical questions | Finish stronger-model d=10 CQR if central; add focused real signed-CQR and/or independently held-out nonlinear/misspecified checks, or explicitly narrow claims |
| 5 | Explain gains and failures | A compact regime analysis separates scale adaptation, envelope tightening, signing, model error and outlier sensitivity; no unsupported dominance claims |
| 6 | Consolidate manuscript and response | Signed results no longer called future work; every reviewer item links to a final section/result; historical status notes replaced |
| 7 | Make the release reproducible | A clean environment runs the smoke example and rebuilds selected paper outputs; code and data access/version/license documented |
| 8 | Final publication pass | Numerical/text/figure agreement, proof/citation checks, document-length decision, PDF QA, cover/metadata and editorial instructions all reconciled |

For the next empirical stage, prefer **a small number of decisive experiments** over more slight variations of already covered Gaussian settings. The high-value questions are: does signing help with accurate quantile fits at higher output dimension; does it help on actual quantile-prediction data; and how does the benefit change with misspecified/nonlinear models and heavy/skewed tails? Existing saved trials can already answer much of the n/d/heterogeneity/alpha mechanism analysis without new fitting.

A theoretically valid robust variant would be interesting future work, but substituting median/MAD or clipping into a candidate-dependent standardization is not an automatic correctness-preserving fix. Likewise, conditional coverage, time-series validity, categorical/missing outputs and exhaustive full-LWC comparisons are separate research goals unless the final article explicitly claims them.

The most useful next joint deliverable is a **claim-to-evidence matrix**: each proposed claim, its exact theorem or experiment, its limitations, and its intended location in the paper. That will determine what truly still needs to be run or proved before the later writing and figure work.
