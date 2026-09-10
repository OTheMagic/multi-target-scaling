# Superseded Synthetic Results for Review

Nothing here has been deleted. Files retain their original project-relative
paths so the provenance can be inspected before deciding what to remove.

- `manifest.json`: 87,039 files, approximately 2.97 GB. Includes known or
  suspected reused-pool synthetic data, preliminary envelope runs using that
  setting, and aggregates/figures derived from them.
- `additional_manifest.json`: retired unfitted toy results and their former
  driver code, plus the pre-migration inventory.
- `notebook_outputs/`: former executed synthetic or mixed notebook outputs,
  archived before clearing the active notebooks.

Reasons distinguish a known reused-pool run from uncertain provenance or an
unfitted score-only toy. The latter is not asserted to have reused a dataset;
it is superseded because it did not train a model as now required.

Active synthetic runners always redraw train/calibration/test observations and
refit in each trial. Frozen configurations and current data are under
`../../envelope_method/`; those runners do not require this review folder.
Fresh exports also replace the old notebook CSV paths. Historical manuscript
PDFs and embedded figures are retained as draft documents, not current results.

## Comparator Rerun Audit (September 9)

The follow-up audit covers all 609 archived CSV files, independent method/scenario
requirements, and mixed aggregates. It found and filled missing raw CQR
Unscaled/Empirical Copula runs and a baseline-only d=50 Laplace configuration.
The undocumented tiny smoke fit was replaced by an explicitly specified fresh
replication. Existing completed small full-LWC outputs are retained; the signed
full-LWC illustration remains deferred, and no new full-LWC work was launched.
See `../../envelope_method/FOLLOWUP_AUDIT.md` and
`../../envelope_method/results/comparator_rerun_audit.csv` for exact coverage of
each method/scenario and the smoke-replication qualification. Nothing here was
deleted or used as input residuals to the new empirical results.
