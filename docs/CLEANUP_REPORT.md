# Cleanup and organization report

> September 10 layout update: numerical files referenced here now live under `data/` with the same repository-relative suffix. See [data layout](DATA_LAYOUT.md). The author removed `deletable/` before this reorganization; historical rollback copies are no longer local.

**Historical phase 1 report.** The author subsequently authorized selective raw-data retention and script changes. See the [phase 2 archive cleanup report](ARCHIVE_CLEANUP_REPORT.md) and [current storage guide](ARCHIVE_STORAGE_GUIDE.md). Statements below about preserving every raw archive, unchanged source hashes, and undecided migration options describe the earlier phase only.

Completed September 9, 2026 (Pacific time), using the author's working goal and conservative retention instructions. Start with [the proposed experiment plan](PAPER_EXPERIMENT_PLAN.md), [retention register](EXPERIMENT_RETENTION.md), and [storage decisions](STORAGE_DECISIONS.md).

## What changed

- Rewrote the root README as a current project entry point and added the [project map](PROJECT_MAP.md).
- Recorded the [working paper goal](WORKING_PAPER_GOAL.md), seven main evidence groups, appendix/exploratory preservation and prioritized missing comparisons.
- Added a [results map](../envelope_method/results/README.md) and readable configuration indexes for [179 absolute-score configurations](../envelope_method/results/absolute/INDEX.md) and [46 CQR configurations](../envelope_method/results/cqr/INDEX.md). The IDs and existing source/data paths stay stable.
- Added `.gitignore` rules for local caches, dependencies and `deletable`; no scientific result directories were newly ignored. This does not untrack already tracked caches.
- Moved the safe candidates below into [deletable](../deletable/README.md), preserving original relative paths and recording SHA-256 hashes. Nothing was deleted.
- Staged ten now-empty cache/render directory trees as well, clearing their old locations. Their [separate directory log](../data/docs/cleanup/empty_directory_moves.json) records the moves; they contain no files and add no logical data bytes.
- Preserved the earlier folder inventory losslessly in its existing gzip form and updated its audit validator to accept that form. Its redundant 49 MB uncompressed copy is staged.

## Staged files

**311 files / 116,730,584 bytes = 116.7 MB (111.3 MiB).**

| Category | Files | Bytes |
|---|---:|---:|
| Regenerable PDF raster previews/contact sheets | 253 | 44,751,602 |
| Python bytecode | 43 | 615,979 |
| pytest execution cache | 5 | 6,619 |
| Matplotlib font cache | 2 | 171,614 |
| TeX editor/file-recorder/SyncTeX tracking | 6 | 653,837 |
| Redundant compiler ZIP, matching retained executable | 1 | 21,060,223 |
| Uncompressed inventory, matching retained gzip data | 1 | 49,470,710 |

See the permanent [per-file plan and hashes](../data/docs/cleanup/move_plan.json), [actual move journal](../data/docs/cleanup/stage_journal.jsonl), and [operation result](../data/docs/cleanup/stage_result.json). Staging did not reduce drive usage; removing `deletable` later reclaims the stated space. The README itself and directory allocation add negligible overhead.

Only the designated QA raster locations were cleared. Scientific figures and final PDFs remain. Future visual QA follows the documented render-then-inspect order. Small JSON/text verification records remain, including the `report.json` read by the portable report packager.

## Protected data and source

All 225 current primary/pilot/repair configurations, 35,463 fitted trials, all fitted toys, the separate 2,400-trial report suite, eight real cohorts, comparator correction/backfill sidecars, completed full-LWC results and exploratory controls remain in place. No `.npz`, `.csv`, fitted model, raw dataset, scientific source, result checkpoint or manuscript figure was staged.

All reviewer reference snapshots, unique historical compile snapshots and the obsolete-data quarantine were retained. Despite their names, some are used by validators or preserve unique author revisions. `tmp/diagnostic_packages`, the extracted TeX compiler/cache and imported `tmp/envelope_*.py` modules are active dependencies and remain.

Small formula/search verification inputs are retained because current auditors consume them and they occupy about 6 MB. Generated visual-verification previews were staged. The distinction is based on dependency and scientific use, not merely whether a filename contains “test” or “verify.”

## Integrity checks

The mover resolved every absolute source/destination under `E:/multi-target-scaling`, rejected internal/reparse paths and overwrites, checked every source hash before moving, and checked every destination hash afterward. It moved files individually with native PowerShell `Move-Item -LiteralPath`.

The [post-cleanup validation](../data/docs/cleanup/validation.json) **passed**: all staged hashes, unchanged metadata for 218,544 retained research/source files, unchanged SHA-256 for 211 scientific source files, exact equivalence of the retained compressed index, and 306 local navigation links. Large retained arrays are checked for unchanged size/modification time in this pass; the cleanup did not claim to recompute every experiment metric.

The preserved checks passed after staging: **63 search-regression tests**, fresh draws/refitting in both synthetic runners, **500 generated tied-score cases** (with the test's documented degenerate exclusions), and **900 exact-zero boundary checks**. Their outcome is recorded in [post-cleanup checks](../data/docs/cleanup/post_cleanup_checks.json). The pre-existing numerical figure/source mismatch remains outside this cleanup; result/manuscript synchronization is still planned work.

## Restore

Before removing `deletable`, run from the project root:

```powershell
./docs/cleanup/stage_cleanup.ps1 -Mode restore
```

The restore operation verifies every staged hash and refuses to overwrite a recreated file. Inspect any collision before restoring; do not force an overwrite. The permanent manifest/journal remains outside `deletable`.

## Why the folder is still large

`envelope_method/results/absolute/` contains about **40.53 GB**, including **40.15 GB of compressed formal trial archives**. Original X/y observations occupy 92.29% of those archive bytes. The stored method-bound vectors occupy only about 25 MB. Deleting checks and previews cannot remove most of the 40 GB.

No whole-NPZ duplicates were found among the 37,201 archives inspected across absolute, CQR and toy directories. However, identical observation-array members account for **17.25 GB (16.06 GiB) of repeated compressed payload**. A shared-array format could retain all scientific information and recover much of this, after accounting for new index/reconstruction overhead. That requires a separate verified migration; original NPZs cannot simply be removed while keeping current checksums/readers/checkpoint behavior.

## Decisions left to the author

| Candidate | What it does | Size / possible saving | Current action |
|---|---|---|---|
| Lossless shared-array storage | Stores repeated X/y arrays once, while preserving every experiment/model/score/bound | About 17.25 GB repeated payload identified; final saving requires a format prototype | Recommended next storage option; no migration performed |
| External formal archive | Moves complete immutable trial files to another storage device with verified hashes/retrieval | About 40.53 GB for `absolute/` locally | Keep local until destination and access protocol are chosen |
| Superseded synthetic quarantine | Preserves old protocols, corrections and comparator history; consumed by historical audits | About 3.00 GB | Retained; retirement would change historical reproducibility |
| All 26 notebook/pilot cohorts | Exploratory quantile/dependence/smoke configurations, including some 200-trial studies | About 445.6 MB | Retained; small savings and useful exploratory evidence |
| Five old compile-verification snapshots | Preserve unique historical TeX plus complete copied build inputs | About 262.8 MB | Retained pending judgment on manuscript history |
| Remove original X/y while retaining scores/metrics | Keeps many existing figures and conformal re-evaluations, loses direct raw-data/fit checks | About 37.05 GB absolute observation payload | Not recommended as routine cleanup; needs explicit weaker archival contract |

The storage figures above are alternatives with overlap, not additive savings. [Storage decisions](STORAGE_DECISIONS.md) explains the exact scope and consequences. A smaller paper selection does not automatically authorize deleting the unselected research.
