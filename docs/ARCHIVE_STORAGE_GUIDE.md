# Archive storage and reproduction

Updated September 10, 2026. Numerical files now live under the root `data/` tree; see [data layout](DATA_LAYOUT.md). The author has removed `deletable/`, so the historical restoration instructions below no longer have a local backup to restore. This is the current storage contract for the main `envelope_method/experiments.py` runner and the authorized absolute/CQR archive migration. The [storage measurements](STORAGE_DECISIONS.md) describe the original archives; the [retention policy](ARCHIVE_RETENTION_POLICY.md) records exact full-retention exceptions. The [archive cleanup report](ARCHIVE_CLEANUP_REPORT.md) records migration progress and verification.

## What remains available

All **225 configurations and 35,463 fitted trials** remain represented under `data/envelope_method/results/`. Every per-trial measurement, seed, configuration, summary and comparator result is preserved. Storage reduction changes the saved array payload, not the scientific experiment set or its repetition counts.

The policy preserves **1,319 original full NPZs, totaling 796,388,868 bytes**: 778 absolute and 541 CQR trials. These include nine complete small cohorts and trial 0 of every other configuration. The complete signed-witness cohort is preserved, rather than selecting only a favorable witness. The remaining 34,144 existing absolute/CQR trials retain scores archives, including all non-X/y arrays needed for unfinished score-based comparisons. See [exact IDs and rationale](ARCHIVE_RETENTION_POLICY.md).

All toys, real cohorts and controls, comparator sidecars, the separate report study, completed full-LWC outputs, historical quarantine, and the six ten-output CQR pilots remain untouched by this migration. A main/appendix/exploratory label does not authorize removing an experiment or an unfavorable result.

| Tier | Active per-trial files | What can be reproduced directly |
|---|---|---|
| `full` | JSON measurements and an NPZ containing all arrays produced by that runner | Saved-score comparisons plus raw-input checks supported by those arrays; the absolute runner also retains OLS parameters. This does not imply every runner saves its fitted model. |
| `scores` | JSON measurements and an NPZ retaining all arrays except `X_train`, `y_train`, `X_cal`, `y_cal`, `X_test`, `y_test` | Method calibration/evaluation from saved calibration/test scores and retained method inputs, including CQR base lengths; stored bounds and parameters remain where originally present. Raw draw and fit reconstruction require the full archive. |
| `compact` | JSON measurements/provenance and derived CSVs; intentionally no trial NPZ | Stored numerical results, aggregate summaries and figures built from those records. New score-based comparators and raw-input checks require a separately retained scores/full run. |

Existing CQR trials were selected for **scores** retention, even though new CQR trials default to **compact**. This preserves their residuals and base lengths for unfinished baselines. Existing richer checkpoints are never automatically downgraded by a later request for a lower tier.

## New experiments: choose storage and destination

Fresh generation and refitting are unchanged: each synthetic trial draws fresh training, calibration and test observations and fits anew; methods within that trial share those observations and predictions. Storage controls what is retained after computing the trial.

`run_spec(item, storage_mode=None, output_root=None)` defaults to `scores` for absolute trials and `compact` for CQR trials unless the item explicitly supplies a storage mode. The main CLI exposes `--storage {compact,scores,full}` and `--out RESULT_ROOT`; it appends the kind and configuration ID to the output root. Its standard grid comes from `settings.json`; notebook and omission-repair configurations have their own drivers.

Run these examples from the repository root with the project's scientific dependencies installed:

```powershell
# New primary absolute run: retain residuals for further method comparisons.
python envelope_method/experiments.py --kind absolute --storage scores --out data/envelope_method/results_new --workers 4

# New primary CQR run: retain measurements for reporting.
python envelope_method/experiments.py --kind cqr --storage compact --out data/envelope_method/results_new --workers 4

# Retain CQR scores when additional residual-based comparisons are planned.
python envelope_method/experiments.py --kind cqr --storage scores --out data/envelope_method/results_cqr_scores --workers 4

# Explicit full archive for the first configuration of a separate demonstration.
python envelope_method/experiments.py --kind absolute --storage full --out data/envelope_method/results_full_demo --limit 1 --workers 1
```

These are experiment commands, not required steps for reading or plotting saved results. `--limit 1` selects one configuration and preserves its configured trial count. For a chosen configuration rather than the first grid item, call `run_spec(selected_item, storage_mode="full", output_root=Path("data/envelope_method/results_full_demo"))` after selecting the exact settings item. Use a separate output root when a richer archive or a changed implementation needs a fresh fit; do not overwrite a completed study to obtain it.

The ten-output CQR runner also defaults to compact for new work and exposes storage choices. Its six existing pilots are retained as they were. Follow its own [protocol and saved-data guide](../envelope_method/cqr10/README.md); the formal sweep remains unrun. Full LWC remains manual-only and disabled by default in its notebook.

For a cluster, use the validated [CQR10 source bundle](../output/cqr10_cluster_source.zip) and [file manifest](../output/cqr10_cluster_source_manifest.json). The ZIP contains 25 source/documentation payload files, including `README_CLUSTER.md` with installation and separate `/scratch/...` output commands. It contains no results, models, pilots, local dependencies or historical archives. The source bundle is sufficient to prepare a new run; it does not contain the local empirical evidence.

## Resume, provenance and integrity

The main runner checks the saved configuration and score transformations before resuming. New checkpoints record Python and key package versions plus hashes of the model/generator/method source files. When recorded provenance differs from the current execution environment, resumption stops and directs the run to a separate output root. A per-configuration `.run_spec.lock` prevents simultaneous main-runner writes to the same configuration; only remove a stale lock after confirming its worker has stopped.

Legacy checkpoints without execution provenance remain explicitly **unknown** with respect to historical code/environment identity. The migration does not relabel them as produced by today's source code. Seeds and deterministic code are useful for a future regeneration, but do not prove exact equality after source or library changes. The original t/Cauchy scale and Gamma generator caveats also remain; storage migration does not correct those scientific designs.

Completed scores/full checkpoints must have their corresponding NPZ and pass integrity checks. Missing or corrupted arrays are errors; the runner does not silently redraw data beneath an existing result. Compact checkpoints intentionally have no NPZ. A richer existing tier can satisfy a lower requested tier unchanged, but a compact/scores checkpoint cannot silently satisfy a full request. New versioned records also carry a digest of the saved measurements.

For migrated archives, two hashes serve different purposes:

- `storage.source_archive_sha256` identifies the original full NPZ. Existing comparator/source links may legitimately retain this hash.
- `archive_sha256` and `storage.archive_sha256` identify the current active NPZ bytes. Use this hash to validate the file that is actually present.

The migration additionally records a digest of the retained NPY payloads and metadata for each removed array: shape, dtype, memory order, NPY byte count and SHA-256. Verification compares every retained payload directly with the staged original and checks unchanged measurement records. A removed-array hash is evidence of identity, not a way to reconstruct the array. A source-link match alone is not an active-file integrity check. Compact checkpoints have no active NPZ hash.

## Reporting and verification

Use saved per-trial JSON and CSV tables for summaries and figures. `summarize.py --summaries-only` refreshes the small summaries and score-representation tables from saved measurements; routine reporting does not need X/y observations and must not trigger fitting. Existing scores archives also support saved-score comparator and formula checks when the required inputs are retained. Raw observation/model reconstruction checks apply only where the required full arrays remain.

```powershell
python envelope_method/summarize.py --summaries-only
python envelope_method/final_audit.py
```

The current `final_audit.py` writes `data/envelope_method/results/retained_storage_audit.json`. It checks measurements and tier-specific evidence and reports unavailable raw-data checks explicitly. The historical `results/final_audit.json` records the earlier full-archive audit; keep it as historical evidence, not a claim that every active trial still contains X/y. Indexes show actual per-configuration full/scores/compact counts and active bytes. Regenerate them only after staging and verification finish:

```powershell
python docs/cleanup/build_indexes.py
```

For publication, preserve estimands and corrected source tables as well as storage integrity. In particular, `real_comparison_audited.csv` is the corrected eight-cohort view; older generic real summaries are historical. A valid checksum does not resolve generator semantics, method definitions, coverage qualifications or manuscript integration.

## Historical staged originals and restoration

Original NPZ/JSON pairs for migrated trials are staged under `deletable/archive_migration_2026-09-09/`, preserving their repository-relative paths. Their per-trial `.transaction.json` records are stored beside those staged pairs and are required for restoration. Full-retention exceptions stay active unchanged. The permanent plan and preparation/stage/verification reports live in `docs/cleanup/archive_migration_2026-09-09/`. Read [the archive cleanup report](ARCHIVE_CLEANUP_REPORT.md), [stage results](../data/docs/cleanup/archive_migration_2026-09-09/stage_result.json), and [verification](../data/docs/cleanup/archive_migration_2026-09-09/validation.json) for completed counts and exact bytes. The earlier [cleanup report](CLEANUP_REPORT.md) covers the separate cache/render/duplicate phase.

Moving originals within the same drive reduces the **active research folder** size but does not reclaim drive space. The originals occupy space until staged files are removed or transferred off the drive. While the originals and transaction records remain intact, restore the exact original active NPZ/JSON pairs with:

```powershell
python docs/cleanup/migrate_archives.py restore
```

Restoration validates paths and hashes before moving files. It preserves reduced pairs under `deletable/archive_migration_2026-09-09/replaced_scores/`, restores the byte-exact original pairs, and records the operation without deleting either representation. It supports resuming an interrupted restore and repeating a completed restore. A restored transaction cannot be staged again by the same reviewed migration.

Removing the staged originals removes this rollback path and the omitted observations for trials outside the full-retention selection. Their metrics and scores remain usable, but exact raw-data checks then need an external copy or a separately validated regeneration. Keep the migration manifest and historical hashes even after deciding how to store the originals long term.
