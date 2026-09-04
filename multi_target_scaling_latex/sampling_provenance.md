# Synthetic Sampling: Verification Needed

The author specifies independently generated training, calibration, and test
observations in each synthetic repetition. The manuscript now presents that
single intended design, with 7,200/800 training/test observations for the
absolute-residual comparisons and the stated smaller CQR/template sizes.

The visual standardization does not establish the sampling history of archived
results. No simulation was rerun and no numerical result was changed in this pass.

## Evidence Requiring Reconciliation

- `../exps.ipynb`, code cell 11 (zero-based), defines `run_synthetic_experiment`.
  Its `make_multitarget_regression(n_samples=8000, ...)` call precedes the
  calibration-size and repetition loops. The repetition loop calls
  `train_test_split(X, y, test_size=0.2, ...)`, giving 6,400/1,600 observations.
  Calibration observations are generated within the loop.
- The notebook's plotting cells read the CSVs in `../syn_exps`. Those same
  archived summaries supply the eight restyled figures below and the five
  distribution tables. These CSVs have no per-run sampling metadata, so the
  notebook alone does not prove that it was the exact generator used for every
  saved CSV. It does conflict with the currently specified sample sizes and
  redraw rule.
- The newer results in `../reviewer_update/data` carry `redraw_train_test=True`
  and explicit sample counts. `../reviewer_update/audit_coverage.py` checks those
  fields for its stated scope; it does not validate the archived CSVs.

## Affected Figures

| Current PDF | Archived summary family |
|---|---|
| `fig_body_abs_independent_gaussian.pdf` | `syn_exps/gaussian/*_gaussian.csv` |
| `fig_body_abs_homogeneous_gaussian.pdf` | `syn_exps/gaussian/*_unit_gaussian.csv` |
| `fig_app_oracle_approximation.pdf` | `syn_exps/laplace/*_laplace.csv`, d=2 |
| `fig_app_local_enclosure_2d.pdf` | Same family, d=2 |
| `fig_app_local_enclosure_10d.pdf` | Same family, d=10 |
| `fig_app_heavy_tail_n30.pdf` | `syn_exps/t/*_t.csv`, d=10, n=30 |
| `fig_app_heavy_tail_n500.pdf` | Same family, d=10, n=500 |
| `fig_app_dimension_scaling.pdf` | `syn_exps/laplace/*_laplace_30sample.csv` |

## Required Resolution

Either provide the generating code/run records supporting the stated design for
these CSVs, provide corrected summaries, or rerun the affected studies and
update both their figures and matching numerical tables. The existing CSVs and
original PDFs are preserved. A rerun must retain the compared methods and
settings, including the low-dimensional local-union comparison, and must not
silently combine results from different runs.

This is an author-facing provenance check, not material for the scientific
experiment narrative. It remains open pending clarification.
