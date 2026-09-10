# Current results map

> Numerical files are centralized under root `data/`. This source-side index is included on GitHub; local data is excluded. The earlier staged originals were removed by the author on September 10.

Start with the [paper experiment plan](../../docs/PAPER_EXPERIMENT_PLAN.md), [retention register](../../docs/EXPERIMENT_RETENTION.md), and [archive storage guide](../../docs/ARCHIVE_STORAGE_GUIDE.md). Configurations, trial IDs, result rows and reporting paths remain stable. Most existing absolute/CQR trials now use scores archives; selected full archives and the separate toy/real/pilot studies are retained.

## Formal synthetic cohorts

| Location | Evidence | Entry point |
|---|---|---|
| `absolute/` | 179 configurations / 33,233 fitted trials: noise laws, scales, dependence, heteroskedasticity, tails, calibration/dimension/alpha stress | [Readable configuration index](absolute/INDEX.md) |
| `cqr/` | 46 configurations / 2,230 fitted trials: quantile scores, bases, shifts and calibration sizes | [Readable configuration index](cqr/INDEX.md) |
| `auxiliary/` | Additional ordinary baseline/oracle/2D-union outputs on the same fits | Match configuration IDs to the absolute index |
| `cqr_baselines/`, `toy_baselines/` | Later paired comparator corrections/backfills; not new independent fits | [Comparator follow-up](../FOLLOWUP_AUDIT.md) |
| `toys/` | 1,730 fitted empirical trials plus separate small boundary-verification arrays | [Toy summary](../../data/envelope_method/results/toy_summary.csv), [signed toy comparisons](../../data/envelope_method/results/signed_gain_by_toy.csv) |
| `../report_revision/` | Separate 2,400-trial controlled report suite, fixed geometry and serial timing | [Report reproduction](../report_revision/README.md) |

Use `configurations.csv` to map old notebook/table names to fresh configuration IDs. `absolute_paired_summary.csv`, signed-gain tables and `paired_overview.csv` are comparisons with specific estimands. Full outcome volumes and positive-residual box volumes differ; do not combine old and new units without conversion.

## Real-data results and explanation

| Location | Role |
|---|---|
| [REAL_COMPARISON.md](REAL_COMPARISON.md), [real_comparison_audited.csv](../../data/envelope_method/results/real_comparison_audited.csv) | Canonical eight-cohort comparison with corrected Point CHR ranks |
| `real/` | Six cached real cohorts and alpha evaluations; shares fits with reviewer diagnostics |
| `extra_real/` | Air and Crime fitted cohorts |
| [rf2_standardized_outlier_control/REPORT.md](rf2_standardized_outlier_control/REPORT.md) | Main-paper candidate: matched retained/deleted outlier with standardized model training |
| [rf2_remaining/REPORT.md](rf2_remaining/REPORT.md) | Tail/shape/model/temporal/subgroup investigations and Point CHR correction |
| [real_outlier_screen/REPORT.md](real_outlier_screen/REPORT.md) | Eight-cohort screen and qualified Crime deletion result |
| `rf2_diagnosis/`, `rf2_remove_one/` | Supporting earlier stages; retain complete provenance |

**Historical branch warning:** generic `real_summary.csv` still contains older Air/Crime Point CHR values. It is retained for history and compatibility, but is not the authoritative final-paper table. Original random-split real results do not establish forecasting validity.

## Prepared or deferred work

- [Ten-output CQR](../cqr10/README.md): six pilots completed and unchanged; formal sweep unrun. Its new trials default to compact storage. A focused base-alpha 0.1 selection is proposed in the paper plan.
- `full_lwc_scaling/`: small completed expensive-comparator outputs, retained. Further full LWC is manual-only and disabled by default in the notebook.
- Existing formula/search/checksum audits remain. The historical `final_audit.json` describes the original full archives; the updated `final_audit.py` writes `retained_storage_audit.json` and reports which raw-data checks the active storage can support.

## Active storage inventory

These counts and logical bytes are read from active configuration directories when the indexes are generated. They exclude staged originals and other result families.

| Location | Configurations | Trials | Full | Scores | Compact | Active directory bytes |
|---|---:|---:|---:|---:|---:|---:|
| `absolute/` | 179 | 33,233 | 778 | 32,455 | 0 | 4,170,655,243 |
| `cqr/` | 46 | 2,230 | 541 | 1,689 | 0 | 205,303,089 |

The [retention policy](../../docs/ARCHIVE_RETENTION_POLICY.md) selects full cohorts and representative trial IDs without selecting on outcomes. Existing scores archives retain all original non-X/y members and all measurement rows so unfinished method comparisons can continue. New `run_spec` trials default to scores for absolute and compact for CQR; `--storage` and `--out` make other choices explicit. Resuming a richer archive does not downgrade it.

The [archive cleanup report](../../docs/ARCHIVE_CLEANUP_REPORT.md) and [migration verification](../../data/docs/cleanup/archive_migration_2026-09-09/validation.json) record the actual staged originals and checks. The [earlier cleanup report](../../docs/CLEANUP_REPORT.md) covers caches, renders and duplicate copies. Staging originals under `deletable/` reduces the active research footprint but does not reclaim drive space until those staged files are removed. Their original hashes remain provenance links; the active NPZ has its own integrity hash. Restore is available while the staged originals and transaction records remain intact. No configuration, trial measurement, comparator row or scientific figure was removed by the storage migration.
