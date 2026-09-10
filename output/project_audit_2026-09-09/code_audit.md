# Implementation and reproducibility audit — September 9, 2026

This is a read-only assessment of the existing implementation, notebook interfaces, experiment runners and saved verification evidence. Small targeted tests were executed; large experiments, manuscript builders, result exports and archive-rewriting verification scripts were not rerun. No research code or results were changed.

## What is implemented

| Component | Existing capability | Main evidence |
|---|---|---|
| Region geometry | Axis-aligned rectangles, containment, intersection, coordinate lengths, volume, 2D drawing | `utility/rectangle.py:5` |
| Shared conformal primitives | Validated conformal rank, partition-based order statistic, explicit infinity when rank exceeds calibration size, deterministic small jitter for historical methods | `utility/conformal_utils.py:7`, `:16`, `:43` |
| Original TSCP family | Global worst-case bound (TSCP-GWC), inexpensive rectangular shortcut (TSCP_R), full local union (TSCP_LWC), ratio/scaled variants and passive search diagnostics | `utility/res_rescaled.py:237`, `:491`; method dispatch `utility/exps.py:100` |
| New signed surface envelope | Exact interval supremum, signed inverse link, reusable calibration object, backward/exhaustive/rank-localized searches, test-dependent lower domains, explicit empty sets, zero-scale and infinite-rank fallback | `utility/envelope.py:13`, `:23`, `:38`, `:90`, `:105` |
| Baselines | Naive in-calibration standardization; separately fitted standardization TSCP-S; supplied-population-statistic benchmark; Point CHR; unscaled maximum; Bonferroni; empirical copula; signed CQHR | `utility/data_splitting.py:7`, `:42`, `:84`, `:127`; `utility/unscaled.py:6`, `:39`; `utility/copula.py:6`; `utility/cqhr.py:28` |
| Synthetic data | Linear multitarget regressions with several noise laws, equicorrelation and AR(1) dependent Gaussian noise; extra contamination and partial heteroskedasticity generators in reviewer builder | `utility/data_generator.py:95`, `:158`; `envelope_method/experiments.py:52` |
| Fitting/evaluation | OLS point predictors, coordinate quantile boosting, random forests and real-data pipelines; signed/capped/shifted CQR; trial and coordinate summaries | `utility/exps.py:338`, `:412`, `:459`, `:900`, `:1227`, `:1737`, `:1819` |
| Auditable fresh simulation runner | Frozen scenario specifications, per-trial fresh draws/refitting, paired methods, NPZ observations/residuals/bounds, JSON metrics and hashes, resumable configuration directories | `envelope_method/experiments.py:44`, `:157` |
| Research report experiment suite | Dedicated 2,400-fit suite, source fingerprints, serialized observations/parameters, independently recomputed metrics and serial runtime study | `envelope_method/report_revision/README.md`, `run_metadata.json`, `verification.json` |
| Ten-output CQR runner | Fresh training/validation/calibration/test observations, 20 fitted quantile models per trial, validation diagnostics, persisted models, corruption detection, code/version/configuration checks and process-safe resume | `envelope_method/cqr10/runner.py:55`, `:185`, `:198`; `test_runner.py` |
| Publication engineering | Figure/style/provenance checks, coverage audits, LaTeX integration checks, real-coordinate/search diagnostics, PDF-render QA scripts | `reviewer_update/` |

The code now supports a substantially broader research program than the root README describes. The README still focuses on the original TSCP submission and lists only five dependencies; it does not orient a new reader to the envelope method, fresh protocol, report revision, CQR extension or verification commands (`README.md:15`, `:21`).

## Notebook inventory

There are nine root notebooks:

- `exps.ipynb`: original synthetic scenarios and local-method comparisons; 28 cells, 24 code cells.
- `real_exps.ipynb`: original real-data driver; 10 code cells, nine execution counts retained and two cells with saved outputs.
- `reviewer_abs_res_experiments.ipynb`: dependence, coordinate profiles, heterogeneity and alpha sweeps; 19 cells, 11 code cells.
- `reviewer_cqr_experiments.ipynb`: capped/shifted CQR studies; 26 cells, 13 code cells.
- Five smoke notebooks: general, coordinate lengths, CQHR, CQR and dependent noise; 4–5 code cells each.

All root synthetic/reviewer/smoke notebooks currently have cleared execution counts and outputs. This is consistent with the protocol migration and is not evidence of execution failure. No saved notebook error outputs were found. The saved notebook-validation report records 10 notebooks and 83 compiled code cells, including the separate disabled full-LWC notebook (`envelope_method/notebook_validation.json`). Compilation is weaker than executing a notebook in a clean Jupyter environment.

`envelope_method/full_lwc_manual.ipynb` has its run switch off by default. `envelope_method/cqr10/cqr10_experiments.ipynb` is also disabled by default. The ten-output formal grid has 22 settings and 1,430 planned trials; its six timing/quality pilots are separate from formal evidence. The delivery README explicitly says live Jupyter and nbformat schema validation were unavailable (`envelope_method/cqr10/README.md:124`).

## Checks executed now

Runtime: the existing Python 3.12 bundled executable, with repository root and `tmp/diagnostic_packages` on `PYTHONPATH`; `PYTHONDONTWRITEBYTECODE=1`; pytest cache provider disabled.

1. `reviewer_update/test_search_diagnostics.py`: **63 passed**. Instrumented/plain historical TSCP rectangles match a preserved reference across binary, backward, fallback, global and full-local modes.
2. `reviewer_update/test_figure_unification.py`: **4 passed, 1 failed**. Styles, geometry of figures and other checks pass; saved plotted values no longer match current source CSV values.
3. `envelope_method/cqr10/test_runner.py`: **8 passed**. This checks the grid, independent draws/seeds, saved model predictions, checkpoint corruption detection, changed-source rejection, rank/backward equivalence, no fitting dependence on calibration/test/validation labels, two-process execution and runtime accounting.
4. `python envelope_method/test_protocol.py`: **passed**. Both public synthetic runners redraw observations and refit per trial; 500 generated tied-score cases are audited with zero-scale cases skipped; all 900 exact-zero boundary cases pass across three searches.
5. Small degeneracy probes were executed, described below.

Total pytest outcome: **75 passed, 1 failed**, plus the passing standalone protocol script. This is a targeted audit, not a comprehensive certification of every method or saved result.

The following stronger historical evidence was inspected, not rerun: formula verification (150 augmentation and inverse cases, 1,200 scalar optimization comparisons, 400 search comparisons, 600 nonnegative-containment cases and 33,651 candidate checks); 9,200 rank/backward saved comparisons; report verification of 2,400 archive hashes and metrics, 60 rerun algorithm pairs, and 120 translation checks. These records supplement mathematical proofs and should not be described as proofs (`envelope_method/verification.json`, `rank_search_audit.json`, `report_revision/verification.json`).

## Concrete findings that affect publication readiness

### 1. Figure provenance is currently stale

The failing test is `reviewer_update/test_figure_unification.py:39`. All eight figures recorded in `multi_target_scaling_latex/figure_style_audit.json` have numerical differences from their current source CSVs: 448 mismatched coverage/volume/runtime/repetition fields in total. For example, the independent-Gaussian figure records empirical-copula coverage 0.714184375 at d=10, n_cal=30, while the current CSV has 0.716559375.

This is an evidence synchronization issue following fresh reruns, rather than merely cosmetic figure work. Freeze the intended result cohort, then regenerate tables, figures, quantitative prose and provenance records together; require the source-match check to pass before submitting.

### 2. A corrected baseline still has historical and corrected output branches

Point CHR now computes the second-half recalibration rank from that half's actual size (`utility/data_splitting.py:178`). The former rank was wrong for certain odd-size splits. The impact audit explicitly preserves historical tables and writes corrected sidecars (`envelope_method/check_chr_rank_fix.py:26`; `results/rf2_remaining/chr_rank_fix_verification.json`).

The saved audit reports air mean coverage changing from 0.8970 to 0.9030 and crime from 0.8945 to 0.9171, with substantial volume changes. Original even-size RF2 and independently saved reduced-RF2 bounds match the corrected implementation. The experiment audit confirms that `envelope_method/results/REAL_COMPARISON.md` and `real_comparison_audited.csv` already use the corrected values, while the generic `real_summary.csv` retains historical entries. Publication tables must consistently select the corrected branch; otherwise a rerun of current code can disagree with a displayed comparison.

### 3. Resume safety differs substantially between runners

The main fresh runner writes a configuration file and resumes any existing trial JSON whose `version` is 2; it does not verify the stored archive hash, compare the previous config/source/environment or include these in its resume gate (`envelope_method/experiments.py:157–176`). Hashes are written at save time and separate archive audits exist, but they are not enforced on ordinary resume.

The real diagnostic runner similarly accepts an existing prepared pickle or residual NPZ by existence alone (`reviewer_update/run_real_diagnostics.py:66–91`). Its README tells users to choose a new output directory after a protocol change.

The CQR10 runner already demonstrates the desired stronger pattern: compare configuration, implementation fingerprint and software versions; verify every saved artifact checksum; refuse mixed cohorts (`envelope_method/cqr10/runner.py:185–213`). Generalizing this pattern would make accidental stale-result mixing much harder.

### 4. Degenerate-score behavior needs an explicit shared policy

The envelope validates finite nonempty inputs and uses a documented whole-domain fallback when any calibration coordinate has zero scale (`utility/envelope.py:93–116`). In direct probes, Naive and TSCP-S return NaN bounds for all-zero scores or one constant coordinate; Point CHR returns NaNs for all-zero scores. The direct historical TSCP implementation raises floating-point warnings in those cases and does not always match the envelope's conservative extension.

Relevant divisions are `utility/data_splitting.py:28–38`, `:110–122`, `:172–187`. This does not disprove results under the positive-scale assumptions. It does mean a reusable public implementation needs documented input conditions or deliberate fallbacks, plus small edge tests for constant coordinates, all-zero capped residuals, tiny calibration sizes, infinities and empty regions. Prefer one canonical method wrapper so paper runners and public entry points share the same policy.

The envelope's tie/zero-boundary implementation is more deliberate: closed-cell comparisons, an exact-forward-score check at finite domain boundaries, and no jitter (`utility/envelope.py:121–124`, `:163–165`). Historical baselines use small deterministic jitter (`utility/conformal_utils.py:43`), which should remain disclosed when comparing near-zero numerical differences.

### 5. Real preprocessing currently uses the full cohort's covariates

Air's mean imputer is fitted on all feature rows before splitting (`utility/exps.py:1788–1792`), and Crime drops columns according to full-cohort missingness (`:1798`). Train/calibration/test splitting and model fitting happen later (`utility/exps.py:1840–1854`).

These transformations use covariates, not held-out outcome labels. A permutation-symmetric feature-only transformation can preserve held-out exchangeability in an appropriate transductive fixed-cohort setup, so this observation alone does **not** establish invalid coverage. It does differ from an ordinary inductive procedure fitted entirely on training observations, and evaluation covariates affect preprocessing. Use train-only pipeline preprocessing for the intended deployment protocol, run a paired sensitivity comparison if these datasets support a main claim, or precisely disclose the transductive preprocessing convention.

### 6. Synthetic law descriptions should match the actual generators

The Gamma generator uses `shape=target_index`, so the first output has zero Gamma noise; its `scale` parameter is not its standard deviation. Cauchy and Student-t branches currently ignore the supplied coordinate `scale` (`utility/data_generator.py:46–63`). These may be intentional legacy scenarios, but captions must not describe them as arbitrary `noise_list`-scaled errors. The Cauchy branch also has no finite population mean/variance, which matters when interpreting a simulated finite-sample “population oracle” standardization benchmark.

## Reproduction assets and remaining release work

Already useful:

- Stable SHA-256-derived seeds, fresh fitted trials and paired-method comparisons.
- Frozen scenario manifests; trial-level arrays and metrics; quarantine manifests for obsolete synthetic results.
- Four local ARFF datasets (`real_exps/data/`), cached UCI datasets, split indices and residual arrays. Five additional dataset loaders use UCI IDs 390, 320, 360, 211 and 242 (`utility/exps.py:1737`).
- A pinned real-diagnostics requirement file and a separate CQR10 requirement file.
- Local report reproduction instructions, artifact hashes, serial timing protocols and finite/infinite/empty distinctions.

Before public release:

1. Establish a small canonical release tree: maintained code, frozen configs, tested environment, scripts, readable results and manuscript source. Keep raw archives and obsolete quarantines in separately versioned data deposits with checksums and a manifest; do not push the entire working directory into Git.
2. Update the root README with the final method names, manuscript/report relationship, current experiment map, data sources, dependency installation, quick smoke command and exact figure/table rebuild path.
3. Provide a portable environment and supported Python version. Root has no package metadata, root requirements/lockfile, license, CI workflow, `.gitignore` or top-level test configuration. Existing requirements cover separate workflows; the root's five-package list omits plotting, notebook, QA and test requirements. The new report guide relies on machine-specific `E:\...` and user-cache paths.
4. Add a license and citation metadata, review dataset redistribution permissions and provide fetch/checksum instructions when redistribution is unsuitable. Preserve research archives rather than deleting them; exclude caches, runtime dependencies, generated bytecode and build debris from the public code package.
5. Define a clean-environment smoke/verification target and a figure-only rebuild target. `reviewer_update/build_experiment_update.py:1352` calls experiment “ensure” functions before plotting; a nominal builder can launch experiments if valid caches are missing. Separate simulation from rendering to make reproduction costs predictable.
6. Freeze one coherent commit/release plus matching data-manifest version. The current worktree has modifications to methods/notebooks/outputs and the new envelope tree/core module is untracked, so a fresh checkout does not yet represent this workspace's current research state.

The strongest conclusion is that substantial implementation and audit infrastructure already exists, including convincing small tests of the new protocol and envelope boundary handling. The immediate software bottleneck is turning several overlapping, differently versioned research workflows into one reproducible publication release, while resolving the concrete stale-figure/baseline and edge-policy issues above.
