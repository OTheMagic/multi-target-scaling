# Archive cleanup and compact cluster output

> September 10 layout update: numerical files referenced here now live under `data/` with the same repository-relative suffix. See [data layout](DATA_LAYOUT.md). The author removed `deletable/` before this reorganization; historical rollback copies are no longer local.

The author authorized this second cleanup phase on September 9, 2026. It preserves every scientific experiment, repetition, metric and paired comparison while reducing saved raw observations. Staging, recovery and independent verification are complete. No experiment or repetition was removed.

## Completed storage reduction

| Active location | Before (GB) | After (GB) |
|---|---:|---:|
| `absolute/` | 40.529 | 4.171 |
| `cqr/` | 0.490 | 0.205 |
| Entire `envelope_method/results/` | 42.009 | 5.367 |

The replaced NPZs fell from **39.817 GB to 3.089 GB**, a **36.728 GB** active-NPZ reduction. New checkpoint metadata and retained full examples are included in the active folder totals above. `deletable/` currently contains approximately **40.613 GB** including the earlier cache cleanup and migration originals, old JSONs and transaction records. GB is decimal; exact bytes and GiB are in the [completion record](../data/docs/cleanup/archive_migration_2026-09-09/completion.json).

**The designated `deletable` contents are ready to remove under the approved retention policy.** No files were deleted automatically. Moving them on the same drive has not freed that drive space; removing the staged originals will. After removal, omitted X/y observations outside the selected full examples are no longer locally restorable, while the active scores, trial measurements and figures remain available.

## Retention

- Converted 34,144 full absolute/CQR NPZs to score archives, retaining every member except the six X/y observation arrays.
- Kept 1,319 full trials selected by the [retention policy](ARCHIVE_RETENTION_POLICY.md), including complete demonstration cohorts and trial 0 of every other configuration.
- Preserve all toys, real cohorts, outlier controls, CQR10 pilots, baseline sidecars, report-revision studies and historical archives.
- Keep all per-trial JSON/CSV results. No trial is discarded or rerun as part of this migration.

The original archives and checkpoint JSONs are moved intact to `deletable/archive_migration_2026-09-09/`, maintaining their original relative paths. Removing that folder after validation removes the ability to restore discarded observations from these local originals. Retained hashes and seeds identify observations but do not reconstruct them by themselves.

## Software

The primary runner and CQR10 runner support `compact`, `scores` and `full`. The primary runner defaults to scores for absolute experiments and compact for CQR; CQR10 defaults to compact. Compact output retains per-trial numerical measurements, seeds and provenance without NPZ/model payloads. Scores retains arrays required for conformal re-evaluation. Full preserves detailed raw evidence. Existing richer checkpoints remain valid under a leaner request; they are never silently downgraded by a resume operation.

Reporting can rebuild from trial JSON without raw arrays, including paired statistics and coordinate tables. New summaries include dispersion, medians, quantiles and nonfinite counts. Selected geometry plots still use the full toy archives that remain available.

New checkpoints validate integrity and reject source/environment drift on resume. The primary runner locks each configuration against overlapping writers; CQR10 locks individual trials. Migrated legacy checkpoints retain their original source hashes, while a separate active hash identifies the reduced NPZ. Updated consumers preserve baseline provenance and distinguish unavailable raw-data checks from checks that passed. Historical audits remain historical records.

See the [storage and cluster guide](ARCHIVE_STORAGE_GUIDE.md) and [CQR10 guide](../envelope_method/cqr10/README.md) for commands. No formal CQR jobs were launched by this cleanup.

## Migration evidence

- [Reviewed policy](cleanup/retention_policy.json)
- [Prepared scope](../data/docs/cleanup/archive_migration_2026-09-09/prepared.json)
- [Per-file plan](../data/docs/cleanup/archive_migration_2026-09-09/plan.json.gz)
- [Conversion-time source/environment snapshot](../data/docs/cleanup/archive_migration_2026-09-09/migration_environment.json)
- [Migration and restore implementation](cleanup/migrate_archives.py)

The migration verifies retained NPY payloads byte for byte, preserves unchanged records, records the removed arrays' shapes/dtypes/NPY hashes, checks original backups, and protects all 805 pre-existing result CSVs. The final independent pass compares active payloads directly with the staged originals.

## Completed verification

- **34,144 replacements:** original NPZ/JSON hashes, unchanged trial measurements, active archive hashes and exact retained payload equality against staged originals passed.
- **1,319 full exceptions and 805 result CSVs:** every protected hash is unchanged.
- **225 configurations / 35,463 trials:** the updated retention audit passed; formula replay covered trial 0 of every configuration. Raw-fit checks are explicitly restricted to retained full evidence.
- **3,160 comparator sidecars / 9,110 method records:** independent coverage and volume recomputation passed.
- **128 distinct pytest tests:** storage, migration/recovery/restore, CQR runner and search tests passed. Fresh-draw/refitting checks plus 500 tied-score and 900 exact-zero cases passed.
- **CQR delivery:** six preserved pilots and 120 fitted models verified; compact notebook plotting and clean extraction of the 86 KB source bundle passed. No formal study was launched.

The initial pass left three operations pending before transaction creation after a destination path check refused to proceed. Originals remained intact. Destination directories are now created serially before worker moves; the resumed pass and independent verification passed. The mover continues to reject paths outside the workspace and records failures immediately.

[Independent migration verification](../data/docs/cleanup/archive_migration_2026-09-09/validation.json) · [Retained-data audit](../data/envelope_method/results/retained_storage_audit.json) · [Comparator verification](../data/envelope_method/results/comparator_retained_verification.json) · [Cluster source ZIP](../output/cqr10_cluster_source.zip) · [Cluster manifest](../output/cqr10_cluster_source_manifest.json)
