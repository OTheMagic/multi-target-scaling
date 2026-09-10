# Envelope versus old shortcut: September 9 research revision

Numeric tables, trial archives, and generated audit JSON are stored under the repository's `data/envelope_method/report_revision/`, retaining the relative paths described below. LaTeX sources and generated table inputs remain here. `report_main.tex` compiles locally with this folder's `figures/`; the parent report uses `envelope_method/figures/report_revision/`.

The revised report is `../signed_envelope.tex` and `../signed_envelope.pdf`.
The portable reading/source copy is `../../output/pdf/envelope_shortcut_report/`.
The previous TeX and PDF are preserved as `original_signed_envelope.*` here.

The report contains 11 sections of interpretation and experiments, eight
appendix sections with the complete audited mathematical development, nine
vector figures, and two tables. It distinguishes uniform weak size containment
on identical nonnegative scores from empirical signed-versus-capped comparisons
and runtime differences. No uniform signed or runtime superiority is claimed.

## New experiments

- 2,400 independent fresh fitted trials; 800 training and 1,200 test observations per trial.
- Four noise laws: Gaussian, Gaussian equicorrelation 0.8, Laplace, and centered lognormal.
- Three features, two covariate-dependent scale groups, heterogeneous outcome scales.
- Primary grid: six outcomes, calibration size 30/80/200, target/base miscoverage 0.1; 120 trials per setting.
- Target sensitivity: Gaussian/Laplace, n=80, target miscoverage 0.05/0.2/0.3 in addition to the primary 0.1.
- Signed contraction studies: two Gaussian outcomes, n=80, base miscoverage 0.1 and 0.02, target 0.1.
- OLS centers with groupwise empirical training-error quantile offsets; all model parameters estimated from training only.
- All methods share data and predictions within each trial. Raw signed quantile scores are never fed into the old nonnegative-score formula.
- The old wrapper returns the whole domain at a zero calibration scale or an infinite conformal rank. The envelope has the same conservative extension. These cases lie outside the positive-scale containment theorem and are reported explicitly.

`trials.csv` stores outcome volumes, simultaneous coverage, all coordinate lengths
and coverages, negative-threshold frequency, empty rates and infinity rates.
`containment.csv` records all 4,800 same-score comparisons across 2,400 fits.
`summary.csv`, `table_n80.csv`, `table_volume.tex`, `evidence.json`, and
`signed_comparison.csv` support the reported numerical statements.

Ratios are averaged within paired trials, with finite positive denominators.
They are not ratios of pooled means. Finite-pair conditioning is explicitly
reported. Coverage summaries retain infinite-region trials. Uncertainty is
across independent trials, never across pooled test observations. The plot
intervals are pointwise Student-t Monte Carlo intervals, not simultaneous tests.

`data/config_XX/trial_YYY.npz` retains full observations, fitted parameters,
predictions, residuals, score thresholds, and empty flags. Its JSON sidecar
contains metrics, seed, design and SHA-256. `manifest.json` indexes every archive.
All 2,400 archive hashes were verified; training-data hashes are distinct.
All metrics were recomputed from the archives. Sixty predetermined trials had
the methods rerun, and 120 translation-invariance checks passed.

## Runtime and geometry

`runtime.csv` and `runtime_summary.csv` are separate serial benchmarks: n in
30/80/200/500/1000, d in 2/6/12, 15 freshly fitted arrays per setting. Each
method has one warmup and three timed calls in randomized order. Times include
all statistics, sorting, search, and the old wrapper's validation/fallback
checks, but exclude fitting, file I/O, plotting, and test evaluation. Dataset
medians are summarized with medians and interquartile bands. Signed-envelope
timing includes both test-width domains with one calibration object.

The two geometry datasets have separate deterministic seeds and fixed test
index 0. The report shows every accepted point of two 241x241 full-conformal
grids inside the envelope. These are geometric diagnostics, not Monte Carlo
coverage estimates or exact full-conformal areas. No full-cell LWC is launched.

## Reproduce in this workspace

From `E:\multi-target-scaling`, using the installed scientific packages:

```powershell
$env:PYTHONPATH='E:\multi-target-scaling\tmp\diagnostic_packages'
$env:PYTHONDONTWRITEBYTECODE='1'
$reportPython='C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $reportPython envelope_method/report_revision/experiment_suite.py all
& $reportPython envelope_method/report_revision/plot_report.py
& $reportPython envelope_method/report_revision/verify_report.py
& $reportPython envelope_method/report_revision/build_report.py
$env:TECTONIC_CACHE_DIR='E:\multi-target-scaling\tmp\tectonic-cache'
& tmp/tectonic-0.17.0/tectonic.exe -C --keep-logs envelope_method/signed_envelope.tex
```

`simulate` resumes only this suite's already completed trial checkpoints.
`benchmark` reruns timings; results vary with load. `geometry` regenerates the
two fixed geometry datasets. `all` invokes all three. No former simulation
pool, model, or quarantined result enters these experiments.

Python scientific dependencies: NumPy, pandas, SciPy, Matplotlib, threadpoolctl.
The executed versions of NumPy, pandas and Python are in `run_metadata.json`.
Original execution-source hashes and final reproduction-source hashes are
recorded separately; later runner edits added the signed runtime measurement
without changing the data generator or simulation calculations.

To render for visual review:

```powershell
& 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -r 115 -png envelope_method/signed_envelope.pdf tmp/pdfs/envelope_revision_qa/page
& $reportPython envelope_method/report_revision/qa_report.py
```

QA additionally uses pdfplumber and Pillow. The final PDF is visually inspected
page by page through contact sheets and enlarged chart/proof pages. Its TeX log
has no overfull boxes, underfull boxes, unresolved references or citations.

## Accuracy revisions

1. Connected the old global transformation explicitly to the exact GWC
   supremum on nonnegative scores before jitter.
2. Generalized the off-coordinate minimum scale to the projection of the mean
   into the clipped GWC domain. When GWC contains the mean it reduces to s.
3. Retained distinctions for signed inverse branches, below-domain emptiness,
   closed cells, ties, zero-scale fallback and infinite conformal rank.
4. Preserved the original finite-sample and search proofs; numerical checks
   supplement rather than replace these proofs.
5. Added original CQR and conformal-rank references and verified their source details.
6. Reported small gains, equality to numerical precision, signed counterexamples,
   infinite-region rates and slower runtime cases alongside favorable results.

The core implementations were inspected and tested, but not modified in this
report revision. Other ongoing repository changes are preserved.
