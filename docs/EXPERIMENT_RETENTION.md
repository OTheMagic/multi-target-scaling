# Experiment retention map

Date: 2026-09-09. Companion to `docs/PAPER_EXPERIMENT_PLAN.md`. **KEEP means retain the experiment and its numerical results.** Active array storage follows the [full/scores/compact contract](ARCHIVE_STORAGE_GUIDE.md) and [exact full-retention policy](ARCHIVE_RETENTION_POLICY.md). Main/appendix/exploratory are paper-selection labels, not judgments that other research data are disposable.

## Always preserve the provenance backbone

| Path | Retention role |
|---|---|
| `envelope_method/settings.json` | 197 primary unique configurations and their source mappings. |
| `envelope_method/notebook_settings.json` | 26 additional notebook configurations; different role and repetition counts. |
| `envelope_method/repair_settings.json` |Two omission-repair configurations/33 fresh trials. |
| `envelope_method/results/configurations.csv` |Human-readable map for all 225 current primary/pilot/repair configurations. |
| `envelope_method/results/absolute/` and `cqr/` |All 225 configurations and 35,463 trial results, JSON sidecars, seeds and metrics remain active. Preserve 1,319 full NPZs unchanged; the other existing trials retain all non-X/y arrays in scores NPZs. Their original full NPZ/JSON pairs are staged separately with hashes and a restore path. |
| `envelope_method/results/auxiliary/`, `cqr_baselines/`, `toy_baselines/` |Paired comparator records and links to the original fresh fit. Keeping only the main-method archives would lose comparator provenance. |
| `envelope_method/results/final_audit.json`, `retained_storage_audit.json`, `audit.json`, `comparator_rerun_audit.*`, `comparator_backfill_verification.json`, `fresh_exports.json`, `obsolete_table_classification.csv` |Historical full-archive audit, current tier-aware verification, completion, historical-to-fresh correspondence and explicit exceptions. The old full-archive audit is not a claim that every active NPZ still has X/y. |
| `docs/cleanup/retention_policy.json`, `docs/cleanup/archive_migration_2026-09-09/`, `deletable/archive_migration_2026-09-09/` |Full-retention selection, source/active hashes, transaction records, verification and staged originals. Keep records permanently; exact rollback and omitted raw-data checks require the staged originals or an external copy. |
| `envelope_method/results/*_trials.csv`, `*_summary.csv`, `paired_overview.csv`, `signed_gain_by_*.csv` |Derived analyses retained with source lineage. Some contain historical comparator rows; use the authoritative corrected view for the paper. |
| `syn_exps/`, `reviewer_exps/`, `reviewer_update/data/` |Current notebook-compatible fresh exports. Preserve complete families and coordinates, not only plotted rows. |
| `reviewer_update/real_diagnostics/cache/` |Six source dataset caches, metadata, exact row splits and fitted residual archives; reused by envelope real results. |
| `envelope_method/results/real/`, `extra_real/` |Six replay cohorts plus Air/Crime reconstructed fitted cohorts. Preserve metadata, row indices and archives. |
| `envelope_method/results/real_comparison_audited.csv`, `REAL_COMPARISON.md`, `real_table_audit.json`, `results/rf2_remaining/chr_rank_*` |Current corrected comparison and audit/correction records. `real_summary.csv` is not a substitute for these corrected tables. |
| `envelope_method/results/toys/`, `toy_baselines/`, fitted toy summaries |1,730 fresh fitted toy trials, supporting counterexamples, shrinkage and shape intuition. |
| `envelope_method/report_revision/` |2,400 additional fresh fitted trials, complete report provenance/runtime/geometry/verification; supports appendix computational refinement. |
| `envelope_method/results/rf2_diagnosis/`, `rf2_remove_one/`, `rf2_remaining/`, `rf2_standardized_outlier_control/`, `real_outlier_screen/` |All original, deletion, standardized, robust-shape, ordered, conditional and Crime controls. Preserve unfavorable findings as carefully as favorable ones. |
| `envelope_method/cqr10/` |Prepared runner/notebook/configs, model-quality and timing pilots, fitted pilot models, requirements, tests and cost estimates. Formal results are not yet present; keep the disabled state. |
| `utility/`, experiment runners, plotting/build/export/audit scripts and notebook sources |Required to regenerate results and figures. Storage readers, resume checks and audits now distinguish full, scores and compact evidence; scientific method definitions remain separately versioned. |

## Paper-role labels

**MAIN KEEP:** the M1–M7 selections and exact IDs in `PAPER_EXPERIMENT_PLAN.md`. The main table keeps all eight real datasets. Do not omit rf2 or Crime because they lose to Point CHR. Do not omit the homogeneous Gaussian control because Unscaled can be smaller.

**APPENDIX KEEP:** complete original noise grids, homogeneous t, higher-df t, target-alpha and dependence sweeps, contamination, all CQR base/shift settings, toy counterexamples, complete coordinate tables, shape-template benchmark, serial runtime/search evidence, controlled outlier details and envelope-versus-old refinement. All results and companion comparator records remain retained. Raw observations are active for selected full archives and otherwise available in staged originals while those originals remain intact.

**EXPLORATORY RETAIN:** ten-output timing/model-selection pilots; notebook smoke/probe studies; candidate robust-shape caps; adaptive-scale rf2 attempt; conditional/ordered rf2 diagnostics; Crime screening and deletion. Some of these provide essential limitations even though they are not confirmatory benchmark studies. Their exploratory status is not permission to hide a failure or erase a record.

**HISTORICAL RETAIN, DO NOT USE AS CURRENT RESULTS:** `quarantine/obsolete_synthetic_2026-09-08/`, its manifests and notebook-output archive; `reviewer_update/pre_*`; original standalone-report copies; legacy manuscript figures/tables/PDFs. Preserve chronology and hashes. The new paper should clearly point to current fresh results and freshly regenerated figures.

**MANUAL/DEFERRED RETAIN:** `envelope_method/full_lwc_manual.ipynb`, `results/full_lwc_scaling/`, existing completed full-LWC/2D union records and their settings. No further full-LWC job should start from ordinary reporting/cleanup. One signed full-LWC illustration remains explicitly deferred and is not necessary to establish the proposed main story.

## Retention rules

1. Never move/delete based solely on age, a duplicate-looking filename, or whether a row appears in the main paper. Multiple tables can be deliberate views of one cohort; similar-looking cohorts can have different feature/training sizes.
2. Preserve the ancestry of every selected claim: source identity/provenance → split/config/seed → retained fitted predictions or residuals → method bounds → trial metrics → aggregate/figure. State what the storage tier supports. A source hash records an original's identity; it does not replace omitted raw arrays or validate the current reduced NPZ. Preserve both source and active hashes.
3. A proposed move must distinguish authoritative data, derived exports, generated figures, historical snapshots and disposable execution caches. This document does not classify arbitrary `tmp/` contents as disposable; earlier tasks stored real research prototypes and outputs there.
4. Keep one explicit public manifest identifying the authoritative paper table/figure source. Preserve historical numbers as historical, especially pre-correction Point CHR Air/Crime records and obsolete synthetic pools.
5. Keep warnings that change interpretation: full versus residual volume, finite-pair conditioning, count of repeated fits versus repeated method evaluations, fresh-per-trial protocol, model-training quality, random versus ordered splits, and confirmed versus hypothetical data errors.
6. Do not load archived joblib/pickle models from untrusted external sources. Existing locally generated models are part of the saved provenance; preservation does not require executing them during cleanup.

No experiment, repetition, measurement or comparator row is removed by the authorized archive migration. Most absolute/CQR raw observation payloads move out of the active representation: original full NPZ/JSON pairs are staged, and scores NPZs preserve every other original array. The [archive cleanup report](ARCHIVE_CLEANUP_REPORT.md) records actual counts and verification; the [earlier cleanup report](CLEANUP_REPORT.md) covers caches, previews and verified duplicate files.

New `run_spec` trials default to scores for absolute and compact for CQR, with `--storage full` available for selected demonstrations and `--out` for separate runs. Existing richer tiers are not downgraded on resume. Staged originals remain restorable with `python docs/cleanup/migrate_archives.py restore` while the backups and transaction records exist. Removing them sacrifices that exact rollback and raw-data verification for trials outside the full-retention selection, even though all saved scores and numerical results remain.

