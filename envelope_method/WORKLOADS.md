# Signed surface-envelope TSCP workloads

User request: derive the method from GWC with full proofs, justify the comparison
with the old shortcut, and rerun the old experiments with saved data.

## Standardized outlier controls (2026-09-09)

Formal report delivered as LaTeX and compiled PDF under
`output/pdf/outlier_sensitivity_report/`, with a portable ZIP alongside it.
The nine-page report has six tables and three vector figures. All table summaries
were checked against 2,400 trial-method records; the final PDF has no TeX layout
warnings or unresolved references, and all nine rendered pages were inspected.
The builder is `build_outlier_sensitivity_report.py`; no new fits were run for
report preparation.

The requested rf2 comparison is complete: 200 new retained-outlier fits with
training-only target standardization, paired with the 200 previously completed
removed-outlier standardized fits. Envelope mean full volume changes from
101,671.05 to 14,691.51 at 90.052%/90.068% coverage; its mean-volume ratio to
Point CHR changes from 6.347 to 1.477, without a ranking reversal. Saved data and
checks: `results/rf2_standardized_outlier_control/REPORT.md`.

Screened all eight existing real cohorts. Crime was the other candidate starting
with larger envelope mean volume. A fixed maximum-input-population deletion was
tested for 200 matched splits in each of four conditions, requiring 600 new fits
and 200 reused original fits. Standardized envelope/CHR mean-volume ratio changes
from 38.615 to 0.101 after deletion. This numerical reversal is driven by rare
extreme-volume calibration splits: 11/200 contribute 99.90% of retained envelope
volume. After deletion envelope is smaller in 100/200 splits; a descriptive
split-bootstrap interval for its mean-volume ratio includes one. The removed
community is an influential size-tail observation, not a verified recording error.

All new fits and reports are complete. The Crime reporting audit recomputes 1,600
method records and checks 184 exact test-only deletion controls. Historical
Point CHR bounds remain preserved; all current comparisons use the corrected
recalibration rank. Report, figure, CSV data, and archive checksums are under
`results/real_outlier_screen/`. No source data were changed and no full-LWC work
was run. No experiment process from this task remains running.

## Follow-up Audit (2026-09-09)

User asked for signed-score gains and proof that all archived comparators,
not just the envelope, have fresh-protocol reruns. Full LWC remains deferred.

- `signed_gain_analysis.py` separates signed versus capped, old shortcut,
  signed GWC, shifted shortcut, and CQHR comparisons.
- `audit_comparator_reruns.py` checks every independent method/scenario/trial
  requirement, including baseline-only rows omitted by the old shortcut table.
- Gaps found and filled: raw CQR Unscaled/Empirical_copula on 2,230 already-fresh
  paired fits, capped/raw comparators for 930 fitted toy trials, d=50 Laplace
  n_cal=30 (30 new fresh trials), and an explicit three-trial smoke replication.
- New capped-toy evaluation exposed inverse cancellation at an exact zero
  boundary. Forward self-score comparison now retains accepted boundary atoms;
  900 regression checks pass. Rechecking all 2,230 main CQR trials found no
  materially changed existing capped-envelope result.
- Consolidation/export complete: 225 configs / 35,463 primary trials; 104
  notebook-compatible table groups and 40 study PDFs. No absolute containment
  violations in 33,233 trials. Comparator audit: 3,401 covered entries and only
  one deferred signed full-LWC illustration, no affordable missing entries.
- All 3,160 comparator sidecars / 9,110 method records pass source/sidecar hash
  and independent metric checks. Formula verification, 63 search tests,
  9,200 rank cases, and disabled manual-notebook validation pass.
- `FOLLOWUP_AUDIT.md` contains the signed-versus-capped results, exceptions,
  repaired comparator list, and the undocumented-smoke-design qualification.
- Final complete archive audit passed all 35,463 archives / 225 configurations.
  All follow-up work is complete except the explicitly deferred full-LWC
  illustration. No experiment process remains running.

LATEST REQUIREMENT (2026-09-08): All synthetic empirical trials must generate
fresh train/calibration/test observations and refit. Remove the selectable
fixed-pool branch. Preserve real-data resplitting. Move known/suspected obsolete
data into one review folder, never delete it. Old workload counts below are
historical checkpoints, not completion claims for the fresh-only rerun.

LATEST RESOURCE LIMIT: User requested no further full-LWC work; provide a
manual Jupyter notebook instead. `full_lwc_manual.ipynb` is ready, off by default,
sequential, and checkpointed. The small full-LWC run had already completed when
the stop arrived. Do not start or extend any full-LWC runs.

Current checkpoint: fresh-only migration complete. Archive manifest contains
87,039 moved files (2.97 GB), plus notebook outputs. Six tied-zero witnesses
exposed a bug in mean_index_solver, now fixed and regression-tested. Witnesses
are in the review folder, not current experimental data. Eight real cohorts
are complete (200 splits each). Fresh fitted toys and all main sweeps complete.

## Workload 1: mathematical specification

Status: complete in `signed_envelope.tex`; 11-page PDF compiled and rendered.
Direct-order-statistic search proposition included. No unresolved references,
unresolved citations, overfull boxes, or underfull boxes in the PDF audit.
Audited augmentation, signed inversion, interval suprema,
coverage by pathwise oracle containment, boundary cases, and positive-score
dominance. Write a standalone TeX document. Do not assume the empirically
observed binary-search predicate without proof.

## Workload 2: implementation and verification

Status: implemented in `utility/envelope.py`. `verify.py` passed (see
`verification.json`): 1,200 optimization comparisons, 33,651 candidate checks,
400 search comparisons, 600 positive dominance checks. Tested analytical
suprema against independent optimization, signed oracle containment, nonnegative
dominance, ties, empty sets, infinite conformal quantiles, and search behavior.
`test_protocol.py` verifies fresh datasets and fit counts for both synthetic APIs
plus 500 atom-heavy cases. `audit_rank_search.py` passed all 9,200 saved cases.

## Workload 3: experiment inventory and reruns

Status: `settings.json` has 175 absolute configurations
(32,800 trials) and 22 CQR configurations (1,430 trials), all completed in
separate checkpointed processes. `notebook_settings.json` adds 26 configurations / 1,200
exploratory trials. Six cached real datasets complete (200 splits each,
alpha 0.1/0.3/0.5/0.7/0.9); air/crime complete at 0.1. Cover the original synthetic notebooks, reviewer
simulation studies, cached real-data cohorts, and the recent signed/CQHR toys.
Retain scenario sizes and repetitions, but always redraw/refit. Save trial-level
data, paired comparisons, seeds/configurations, summaries, and progress per
workload. Distinguish exact replay from new replications. Never overwrite old
experiment results or silently reduce repetitions.

## Workload 4: integration audit

Status: complete for the non-full-LWC scope. Final archive audit passed for all
223 configurations / 35,430 raw archives: checksums, shapes, distinct draws,
fit metadata, and one formula replay per configuration. Notebook validation
passed for 10 notebooks / 83 code cells; the manual runner was not called.
Study plot/report exports and PDF inspection complete. Full-LWC extensions
remain explicitly deferred for manual execution, as requested.

## Existing artifacts

- Old theory: `multi_target_scaling_latex/body.tex`, `supplementary.tex`.
- Old implementation: `utility/res_rescaled.py`.
- Recent prototype: `tmp/envelope_cqhr_toy.py` and other `tmp/envelope_*.py`.
- Bundled Python:
  `C:/Users/admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
- Preserve pre-existing untracked files under `tmp/`.

## Execution checkpoint (2026-09-08)

- Absolute, CQR, auxiliary, notebook pilots, fitted toys, rank audit, real replays,
  and archive migration finished. Primary/pilot total: 223 configs, 35,430 trials.
- Final archive audit passed; 39 study PDFs cover 34 source-study labels.
- No full-LWC process is running or planned.
- Python environment needs `PYTHONPATH=E:/multi-target-scaling/tmp/diagnostic_packages`.
- Compile with `TECTONIC_CACHE_DIR=E:/multi-target-scaling/tmp/tectonic-cache`
  and `tmp/tectonic-0.17.0/tectonic.exe --only-cached --keep-logs envelope_method/signed_envelope.tex`.
- Review folder: `quarantine/obsolete_synthetic_2026-09-08`. Do not delete it.
- Consolidation complete: zero absolute containment violations in 33,200 trials.
- Fresh notebook-compatible CSV exports complete (103 tables/groups).
- Formula tests, 63 search diagnostics, fresh-protocol tests, and 9,200 rank
  comparisons passed. A final notebook check corrected one **kwargs dictionary
  left over from the removed synthetic pool API.
- Signed-versus-capped paired summaries report finite-pair counts explicitly.
  A saved witness shows all-positive GWC bounds with a negative envelope bound
  in one coordinate; it is not a claim of total-volume shrinkage versus base.
- Main-manuscript historical figures/PDFs were not rewritten; the standalone
  method note and fresh results are the current outputs for this task.

## Ten-outcome extension checkpoint (2026-09-09)

Completed code, bounded timing/quality pilots and a manual notebook under
`cqr10/`. The 1,430-trial formal extension remains unrun by design after estimating
11.78 single-worker hours. Detailed workload status: `cqr10/WORKLOADS.md`.
No full-LWC computation was launched. Earlier workload records above describe
the previous three-outcome/absolute-score cohorts, not this new study.
