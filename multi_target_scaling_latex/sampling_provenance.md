# Synthetic Sampling: Fresh-Only Rerun

All active synthetic runners now generate independent training, calibration,
and test observations and refit within each trial. The selectable reused-pool
branch has been removed from `utility/exps.py`, its callers, and notebook-local
implementations. Real-data resplitting is unchanged.

The new signed-envelope study has 225 completed synthetic configurations and
35,463 fresh fitted trials, plus 1,730 fitted toy trials. Frozen settings record
the actual sample counts (including 6,400/1,600, 7,200/800, and smaller CQR/pilot
designs). These sizes are not reused pools. See `../envelope_method/README.md`,
`RESULTS.md`, and the per-trial data/audits for the current study.

The September 9 follow-up audits baseline-specific tables as well as envelope
tables. It fills missing raw CQR Unscaled/Empirical Copula comparisons on the
same fresh per-trial fits and adds a baseline-only d=50 Laplace configuration.
One tiny smoke fit had undocumented training parameters; its three-trial fresh
replacement is explicitly labeled as a replication. See
`../envelope_method/FOLLOWUP_AUDIT.md` for paired signed-score gains and the full
method/scenario rerun audit. The signed full-LWC illustration remains deferred.

## Preserved Historical Evidence

- The former notebook generated training/test observations outside its trial
  loop. Its source and executed outputs are preserved in the review folder.
- The former synthetic CSVs lacked reliable per-trial sampling metadata or
  used that superseded design. They were moved, not deleted. Fresh tables have
  been exported under the original notebook paths with explicit protocol and
  repetition metadata.
- The newer results in `../reviewer_update/data` carry `redraw_train_test=True`
  and explicit sample counts. `../reviewer_update/audit_coverage.py` checks those
  fields for its stated scope; it does not validate the archived CSVs.

The single review folder is
`../quarantine/obsolete_synthetic_2026-09-08/`, with hash manifests and reasons.
It is not an input to the fresh runners.

## Historical Figures

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

## Manuscript Integration Boundary

The standalone method derivation and current figures are under
`../envelope_method/`. The historical manuscript PDFs, captions, and embedded
figures have not been rewritten as part of this standalone-method task. They
must not be presented as results from the new method or as the fresh rerun.
Updating the manuscript requires reconciling its stated sample counts and
captions with the explicit new configurations, rather than merely swapping PDFs.
Full-LWC computation is manual-only going forward, per the author's latest
instruction; `../envelope_method/full_lwc_manual.ipynb` is ready and disabled by
default. Results that completed before that instruction remain preserved.
