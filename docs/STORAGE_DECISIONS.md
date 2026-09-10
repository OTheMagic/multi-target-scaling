# Storage measurements and the chosen retention contract

> September 10 layout update: numerical files referenced here now live under `data/` with the same repository-relative suffix. See [data layout](DATA_LAYOUT.md). The author removed `deletable/` before this reorganization; historical rollback copies are no longer local.

The numerical inventory below was measured on September 9, 2026 **before the authorized archive migration**. It is a historical baseline, not the current active folder size. That measurement step was read-only; the later migration stages original full NPZ/JSON pairs and installs reduced active archives. Sizes are logical file bytes, not filesystem allocation; GiB = bytes / 1,073,741,824. Moving files to another folder on the same drive does not reduce that drive's total usage.

## Current decision

**Retain every experiment and numerical result, with explicit storage tiers.** All 225 configurations and 35,463 fitted absolute/CQR trials remain active. The [retention policy](ARCHIVE_RETENTION_POLICY.md) preserves 1,319 full trials unchanged (796,388,868 NPZ bytes); the other 34,144 trials retain every non-X/y member in scores NPZs. Original full NPZ/JSON pairs are staged under `deletable/archive_migration_2026-09-09/`. Toys, real cohorts and controls, comparator sidecars, the report study, full-LWC outputs and existing ten-output CQR pilots remain unchanged.

The [archive cleanup report](ARCHIVE_CLEANUP_REPORT.md) records actual staging and verification; the generated [results indexes](../envelope_method/results/README.md) report active storage counts and bytes. Do not infer current savings by subtracting the historical payload totals below: selected full archives, NPZ container overhead and new checkpoint metadata affect the result. Staging reduces the active research footprint; total drive space is not reclaimed while staged originals remain on the same drive.

The main runner now validates completed scores/full archives rather than silently skipping missing data. New main-runner defaults are scores for absolute and compact for CQR, with explicit `--storage` and `--out`; a richer existing archive is not downgraded on resume. Current audits distinguish evidence supported by each tier. The [storage guide](ARCHIVE_STORAGE_GUIDE.md) explains source versus active hashes, new execution-provenance checks, legacy unknown provenance and restoration. These changes supersede the original audit's recommendation to keep every raw archive active.

## Pre-migration folder sizes

At measurement time, `envelope_method/results` contained **123,492 files and 42,009,302,113 bytes (39.124 GiB)**.

| Directory / files | Files | Bytes | GiB | What it contains |
|---|---:|---:|---:|---|
| `absolute/` | 67,182 | 40,528,714,676 | 37.745 | 33,233 fitted absolute-score trial NPZs; trial JSONs, configurations, tables |
| `cqr/` | 4,647 | 489,707,397 | 0.456 | 2,230 fitted CQR trials, signed/capped bounds, base widths and raw observations |
| `toys/` | 3,523 | 298,194,279 | 0.278 | 1,730 fitted toy trials plus eight saved boundary-test arrays |
| Top-level result files | 35 | 292,401,177 | 0.272 | Mostly combined/coordinate trial CSVs used by reporting scripts |
| `auxiliary/` | 35,248 | 127,747,982 | 0.119 | Additional baseline/oracle bounds paired to primary fits |
| `real/` | 2,419 | 70,804,991 | 0.066 | Six real cohorts, multi-target-alpha bounds/timing records |
| `rf2_remaining/` | 860 | 48,955,751 | 0.046 | Model/time/outlier diagnostics and residual/index arrays |
| `real_outlier_screen/` | 1,212 | 44,087,124 | 0.041 | Crime controls and outlier-screen evidence |
| `extra_real/` | 810 | 36,759,728 | 0.034 | Additional real datasets, split indices and residuals |
| `rf2_remove_one/` | 407 | 26,927,506 | 0.025 | Removed-observation reruns |
| `rf2_standardized_outlier_control/` | 405 | 26,887,444 | 0.025 | Standardized retained-observation controls |
| `cqr_baselines/` | 4,552 | 9,241,112 | 0.009 | 2,230 extra comparator sidecars; paired to existing CQR fits |
| `toy_baselines/` | 1,878 | 7,795,371 | 0.007 | 930 extra comparator sidecars; paired to existing toy fits |
| `rf2_diagnosis/` | 10 | 676,141 | 0.001 | Compact diagnostic tables/report |
| `full_lwc_scaling/` | 304 | 401,434 | <0.001 | 140 small, computationally expensive full-LWC result sidecars |

An empty `legacy_auxiliary/` directory consumes no logical file bytes in this inventory.

Within the original `absolute/`, NPZs occupied **40,146,515,296 bytes**, JSONs **269,023,032 bytes**, and CSVs **113,176,348 bytes**. The NPZs were already ZIP-compressed NumPy archives. A complete ZIP-directory scan established that **92.29% of original absolute NPZ bytes were X/y observations**, while all six methods' bound vectors occupied only 0.062%.

| Content in all 33,233 absolute NPZs | Compressed payload bytes | GiB | Share of NPZ size |
|---|---:|---:|---:|
| Training X and y | 30,083,202,701 | 28.017 | 74.934% |
| Calibration X and y | 888,134,213 | 0.827 | 2.212% |
| Test X and y | 6,079,901,469 | 5.662 | 15.144% |
| Calibration and test residual scores | 2,948,844,380 | 2.746 | 7.345% |
| Fitted model coefficients/intercepts | 28,491,146 | 0.027 | 0.071% |
| DGP coefficients | 23,929,232 | 0.022 | 0.060% |
| Six methods' bound vectors | 24,887,515 | 0.023 | 0.062% |
| ZIP headers and directory overhead | 69,124,640 | 0.064 | 0.172% |

The six X/y members total **37,051,238,383 bytes (34.507 GiB)**. These are the space-consuming saved observations underlying the results; they are not figure files, numerical optimizers' caches, or repeated method-bound vectors.

## Original formal-experiment and diagnostic inventory

The three settings maps contain **225 distinct configuration IDs**, with no repeated IDs across the maps:

| Settings map | Configurations | Fresh fitted trials | Complete primary-folder bytes |
|---|---:|---:|---:|
| `envelope_method/settings.json` | 197 | 34,230 | 40,441,594,827 |
| `envelope_method/notebook_settings.json` | 26 | 1,200 | 445,564,531 |
| `envelope_method/repair_settings.json` | 2 | 33 | 131,262,715 |

The map labels do not make their results expendable. The historical full-archive audit and comparator-completeness summaries included these 225 configurations, all of which remain in the retained experiment set. The 26 notebook configurations include 20 ten-trial exploratory CQR probes, but also six 100/200-trial notebook cohorts. Their sources and roles are recorded in each `config.json`. The sizes in this section are pre-migration measurements.

- The 20 ten-trial CQR probes together occupy **27,490,672 bytes**. They are candidates for a deliberate narrower *paper selection*, with their archived results retained separately; they are not the source of the storage problem.
- Two smoke-labeled absolute cohorts, `ba873459937c95e9` and `686fa458729fc0ea`, occupy **136,141,985** and **214,632,937 bytes** respectively. Each actually has 200 fresh fitted trials. Deleting them would alter current notebook/comparator coverage.
- `toys/boundary/` is a **6,228,329-byte** numerical search diagnostic (9,200 score cases / 18,700 coordinate sequences), distinct from the 1,730 fitted empirical trials. Keep it as compact regression evidence; do not count its cases as fresh fitted empirical repetitions.
- `auxiliary/`, `cqr_baselines/`, `toy_baselines/`, and `full_lwc_scaling/` mostly save *additional method outputs on existing trial fits*, not a second copy of the raw fitted datasets. They preserve paired comparisons. Deleting them sacrifices comparator evidence for little space.

## Retention decisions and remaining storage choices

| Decision | Grounded size information | Effect / status |
|---|---:|---|
| **ADOPTED: scores retention for most existing absolute/CQR trials, selected full archives** | 1,319 full NPZs remain unchanged at 796,388,868 bytes; actual reduced bytes are in the archive cleanup report | Every original non-X/y member and all metrics survive. Original full pairs are staged and restorable while retained. No configuration or repetition is removed. |
| **KEEP: all JSON measurements, configurations, seeds and summary/figure sources** | All 35,463 absolute/CQR trial records remain | The saved experiment set and numerical results remain available independently of storage tier. |
| **KEEP: current baseline sidecars, real controls, boundary evidence and completed full-LWC results** | 0 | Small; these contain scientific/comparator results that should not be confused with caches. |
| **KEEP: all toys and the six existing CQR10 pilots unchanged** | Toy family originally 298,194,279 bytes; separate pilots are outside the measured formal absolute/CQR set | Preserves demonstrations, verification arrays and pilot fitted models without a new migration. |
| **LONG-TERM CHOICE: keep staged originals locally or in a verified external archive** | See actual staged bytes in the archive cleanup report | Keeping an intact local/external copy preserves omitted X/y and exact original hashes. Same-drive staging saves no total drive space. Removing the only original copy ends raw-data verification and byte-exact rollback for those trials. |
| **FUTURE OPTION: deduplicate original arrays into shared immutable objects** | Historical repeated compressed payload: 16.063 GiB across absolute/CQR/toys; 15.843 GiB within absolute | A possible lossless external/archive format, not the migration chosen here. Requires new readers, metadata and numerical/byte-round-trip checks. This original-array duplication is not still present in every active scores archive. No shared-array deduplication was performed. |
| **KEEP: derived CSVs at their current expected paths** | Up to 280,284,423 bytes for the two largest combined tables alone | `simulation_trials.csv` and `absolute_coordinate_trials.csv` are regenerable (`summarize.py:82`–`:139`), but reporting and comparison audits read them directly. Small relative to raw data; do not break the current workflow for this saving. |

The historical alternative of removing every absolute NPZ and keeping only JSON/CSVs would have removed 37.389 GiB of archives, including residuals needed for unfinished method comparisons. That alternative was not selected. Existing absolute/CQR scores and all non-X/y arrays are retained; future compact CQR runs intentionally make a narrower retention promise.

## Why seed-only regeneration is a different promise

The runner deterministically generates and refits each trial. That makes future regeneration possible in principle, but seeds alone do not prove regenerated arrays, fitted parameters, timing values or compressed-file hashes equal the originals under a changed code/library environment. The historical full-archive audit checked saved X/y shapes, trial draws, model predictions against residuals and original SHA-256 values. The current tier-aware audit performs raw-data checks only where full arrays remain and reports them unavailable elsewhere. New checkpoints carry source/environment provenance and reject recorded drift on resume; absent provenance on legacy records stays unknown.

Identical raw arrays across some configurations can be legitimate common-random-number pairing: the primary runner's data seeds depend on d, calibration size and trial, while configuration-specific noise/model/target choices may differ. This does not mean a model is reused across trials, and does not make the distinct result configurations interchangeable.

## Verified duplication in the original archives

The pre-migration ZIP-directory scan covered **37,201 NPZ files**: all 33,233 absolute, 2,230 CQR, and 1,738 toy archives. It completed without archive-reading errors. Candidate whole-file duplicates were screened by ordered ZIP member metadata and checked by complete-file SHA-256 where needed. **There were no byte-identical whole NPZ duplicates in this scope.** The authorized reduction therefore changes storage representation; it is not deletion of redundant whole trial files.

However, the six original-data array members do contain substantial repeated content across configurations. Candidate groups with equal member name, uncompressed size and ZIP CRC were decompressed and SHA-256 hashed to verify equality. This found **25,719 groups and 97,530 redundant array-member copies**. These are identical saved arrays, not necessarily identical models, residuals, bounds or scientific configurations.

| Identical member type | Redundant copies | Repeated compressed payload bytes |
|---|---:|---:|
| `X_train` | 27,320 | 13,344,629,809 |
| `X_cal` | 29,690 | 432,474,726 |
| `X_test` | 27,320 | 2,800,040,462 |
| `y_train` | 2,110 | 476,891,363 |
| `y_cal` | 8,979 | 105,244,453 |
| `y_test` | 2,111 | 88,661,990 |
| **Total** | **97,530** | **17,247,942,803 (16.063 GiB)** |

The total is the sum of existing compressed member sizes beyond one smallest stored representative per identical-array group. It measures removable *payload duplication in a redesigned shared-array archive*, before adding the new index, reconstruction metadata and container overhead. It is not a promised filesystem saving from a deletion command. The measured within-absolute component is **17,011,573,338 bytes (15.843 GiB)**. No array deduplication was performed.

A possible future project is a lossless shared-array representation of the staged/external originals with round-trip checks. It would preserve observations beyond the chosen active scores contract. If original NPZ checksums are to remain valid, verify reconstruction of original file bytes as well as numerical array equality. Keep the original archive until any such separate migration passes.

Companion evidence:

- `output/project_audit_2026-09-09/storage_npz_measurements.json`: compact category totals, member totals, counts, verification scope and an identical-array example.
- `output/project_audit_2026-09-09/storage_identical_arrays.json.gz`: compressed mapping of all verified identical-member groups to original archive paths, member names and compressed sizes (494,487 bytes).

Only the six named X/y members were checked for identical array content; additional redundancy in scores, coefficients or other study directories was not claimed. Whole-file duplicate checks covered the 37,201 original archives above, not the entire repository. These historical measurement artifacts are preserved; the subsequent [archive cleanup report](ARCHIVE_CLEANUP_REPORT.md) is authoritative for the implemented migration and its verification.
