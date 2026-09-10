"""Index retained experiments without changing settings, checkpoints, or CSVs.

Run only after archive staging/verification completes. Importing does not write.
Storage counts come from active checkpoints, not an assumed migration state.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import re

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / "envelope_method"
DATA_METHOD = ROOT / "data/envelope_method"
MODES = ("full", "scores", "compact")


def load_items():
    items = []
    for name, role in [("settings.json", "primary"), ("notebook_settings.json", "notebook/exploratory"), ("repair_settings.json", "omission repair")]:
        for original in json.loads((METHOD / name).read_text(encoding="utf-8")):
            items.append(dict(original, source_map=name, role=role))
    if len(items) != 225 or len({item["id"] for item in items}) != 225:
        raise ValueError("Expected the reviewed 225 distinct configurations.")
    return items


def active_storage(item):
    """Read active metadata and file sizes; this is not a hash audit."""
    folder = DATA_METHOD / "results" / item["config"]["kind"] / item["id"]
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if status["completed"] != item["trials"]:
        raise ValueError(f"Incomplete configuration: {folder}")
    if (folder / ".run_spec.lock").exists() or any(folder.glob("*.scores.pending.npz")):
        raise ValueError(f"A run or migration may be active: {folder}")
    counts = Counter({mode: 0 for mode in MODES})
    archive_bytes = 0
    for trial in range(item["trials"]):
        checkpoint_path = folder / f"trial_{trial:03d}.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        storage = checkpoint.get("storage")
        if storage is not None and storage.get("schema_version") != 1:
            raise ValueError(f"Unsupported storage schema: {checkpoint_path}")
        mode = (storage or {}).get("mode", "full")
        if mode not in MODES:
            raise ValueError(f"Unknown storage mode {mode!r}: {checkpoint_path}")
        if not checkpoint.get("records"):
            raise ValueError(f"Missing trial measurements: {checkpoint_path}")
        archive = checkpoint_path.with_suffix(".npz")
        if mode == "compact":
            if archive.exists():
                raise ValueError(f"Unexpected NPZ for compact checkpoint: {archive}")
        else:
            if not archive.is_file():
                raise FileNotFoundError(f"Missing {mode} archive: {archive}")
            archive_bytes += archive.stat().st_size
        counts[mode] += 1
    return dict(counts=counts, archive_bytes=archive_bytes,
                active_bytes=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()))


def totals(selected, storage):
    return dict(
        configurations=len(selected), trials=sum(item["trials"] for item in selected),
        counts={mode: sum(storage[item["id"]]["counts"][mode] for item in selected) for mode in MODES},
        archive_bytes=sum(storage[item["id"]]["archive_bytes"] for item in selected),
        active_bytes=sum(storage[item["id"]]["active_bytes"] for item in selected))


def render_index(kind, selected, storage):
    summary = totals(selected, storage)
    counts = summary["counts"]
    lines = [
        f"# {kind.upper()} experiment index", "",
        "Every listed configuration, trial and saved result remains active under root `data/envelope_method/results/`. Full archives retain observations; scores archives retain every original non-X/y member; compact trials retain measurements in JSON/CSV. Storage counts below are read from the active checkpoints.", "",
        "[Paper selection](../../../docs/PAPER_EXPERIMENT_PLAN.md) | [Retention register](../../../docs/EXPERIMENT_RETENTION.md) | [Storage guide](../../../docs/ARCHIVE_STORAGE_GUIDE.md) | [Results map](../README.md)", "",
        f"**{summary['configurations']} configurations / {summary['trials']:,} fitted trials.** Active storage: **{counts['full']:,} full / {counts['scores']:,} scores / {counts['compact']:,} compact**. Trial NPZs: **{summary['archive_bytes']:,} bytes**; complete configuration directories: **{summary['active_bytes']:,} bytes**. Staged originals under `deletable/` are excluded.", "",
        "Each trial has fresh observations and refitting. Shared study views and comparator sidecars are not extra fits. Columns show actual settings and original source views. Student-t/Cauchy original generators use unit-scale noise despite their stored scale vectors; Gamma has coordinate-index shapes including zero. Read the paper plan before interpreting law labels.", "",
    ]
    groups = defaultdict(list)
    for item in selected:
        groups[str(item["config"].get("noise_type", "custom"))].append(item)
    for noise, group in sorted(groups.items()):
        lines += [f"## {noise}", "",
            "| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|"]
        for item in sorted(group, key=lambda i: (i["config"]["d"], i["config"]["n_cal"], i["id"])):
            config = item["config"]
            observed = storage[item["id"]]
            mode_counts = observed["counts"]
            source = "; ".join(item.get("sources", [])).replace("\\", "/").replace("|", " / ")
            lines.append(f"| [`{item['id']}`](../../../data/envelope_method/results/{kind}/{item['id']}/config.json) | {item['role']} | {config['d']} | {config['n_cal']} | {config['alpha']} | {config.get('base_alpha', '—')} | {config['n_train']:,} / {config['n_test']:,} | {item['trials']} | {mode_counts['full']} | {mode_counts['scores']} | {mode_counts['compact']} | {observed['active_bytes'] / 1024**2:.2f} | {source} |")
        lines += [""]
    lines += ["## Reading one configuration", "",
        "`config.json` fixes the design. `trial_*.json` records measurements, seeds, hashes and the declared storage tier. Legacy checkpoints without storage metadata mean full storage. A full `trial_*.npz` includes training/calibration/test observations; a scores NPZ omits only the six X/y arrays. Compact checkpoints intentionally have no NPZ. `trials.csv`/`summary.csv` are reporting exports, and `status.json` records completion. Comparator sidecars may be in adjacent `auxiliary/` or `cqr_baselines/` directories.", "",
        "Active MiB counts all files in each configuration directory, including JSON/CSV, and uses bytes / 1,048,576. These counts are an index, not a substitute for `final_audit.py` or migration hash verification. Missing scores/full NPZs are errors and are not automatically redrawn. Preserve original source hashes for old comparator links and use the active hash to check the current NPZ bytes.", "",
        "The author removed the earlier `deletable/` originals. Exact original observations are locally available only for retained full trials; scores archives continue to support their documented comparisons. See the [storage guide](../../../docs/ARCHIVE_STORAGE_GUIDE.md).", ""]
    return "\n".join(lines)


def render_overview(items, storage):
    summaries = {kind: totals([item for item in items if item["config"]["kind"] == kind], storage) for kind in ("absolute", "cqr")}
    storage_rows = []
    for kind, summary in summaries.items():
        counts = summary["counts"]
        storage_rows.append(f"| `{kind}/` | {summary['configurations']} | {summary['trials']:,} | {counts['full']:,} | {counts['scores']:,} | {counts['compact']:,} | {summary['active_bytes']:,} |")
    return """# Current results map

Start with the [paper experiment plan](../../docs/PAPER_EXPERIMENT_PLAN.md), [retention register](../../docs/EXPERIMENT_RETENTION.md), and [archive storage guide](../../docs/ARCHIVE_STORAGE_GUIDE.md). Configurations, trial IDs, result rows and reporting paths remain stable. Most existing absolute/CQR trials now use scores archives; selected full archives and the separate toy/real/pilot studies are retained.

## Formal synthetic cohorts

| Location | Evidence | Entry point |
|---|---|---|
| `absolute/` | 179 configurations / 33,233 fitted trials: noise laws, scales, dependence, heteroskedasticity, tails, calibration/dimension/alpha stress | [Readable configuration index](absolute/INDEX.md) |
| `cqr/` | 46 configurations / 2,230 fitted trials: quantile scores, bases, shifts and calibration sizes | [Readable configuration index](cqr/INDEX.md) |
| `auxiliary/` | Additional ordinary baseline/oracle/2D-union outputs on the same fits | Match configuration IDs to the absolute index |
| `cqr_baselines/`, `toy_baselines/` | Later paired comparator corrections/backfills; not new independent fits | [Comparator follow-up](../FOLLOWUP_AUDIT.md) |
| `toys/` | 1,730 fitted empirical trials plus separate small boundary-verification arrays | [Toy summary](toy_summary.csv), [signed toy comparisons](signed_gain_by_toy.csv) |
| `../report_revision/` | Separate 2,400-trial controlled report suite, fixed geometry and serial timing | [Report reproduction](../report_revision/README.md) |

Use `configurations.csv` to map old notebook/table names to fresh configuration IDs. `absolute_paired_summary.csv`, signed-gain tables and `paired_overview.csv` are comparisons with specific estimands. Full outcome volumes and positive-residual box volumes differ; do not combine old and new units without conversion.

## Real-data results and explanation

| Location | Role |
|---|---|
| [REAL_COMPARISON.md](REAL_COMPARISON.md), [real_comparison_audited.csv](real_comparison_audited.csv) | Canonical eight-cohort comparison with corrected Point CHR ranks |
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
""" + "\n".join(storage_rows) + """

The [retention policy](../../docs/ARCHIVE_RETENTION_POLICY.md) selects full cohorts and representative trial IDs without selecting on outcomes. Existing scores archives retain all original non-X/y members and all measurement rows so unfinished method comparisons can continue. New `run_spec` trials default to scores for absolute and compact for CQR; `--storage` and `--out` make other choices explicit. Resuming a richer archive does not downgrade it.

The [archive cleanup report](../../docs/ARCHIVE_CLEANUP_REPORT.md) and [migration verification](../../docs/cleanup/archive_migration_2026-09-09/validation.json) record the actual staged originals and checks. The [earlier cleanup report](../../docs/CLEANUP_REPORT.md) covers caches, renders and duplicate copies. Staging originals under `deletable/` reduces the active research footprint but does not reclaim drive space until those staged files are removed. Their original hashes remain provenance links; the active NPZ has its own integrity hash. Restore is available while the staged originals and transaction records remain intact. No configuration, trial measurement, comparator row or scientific figure was removed by the storage migration.
"""


def main():
    items = load_items()
    storage = {item["id"]: active_storage(item) for item in items}
    # Validate and render the whole inventory before writing any index.
    outputs = {METHOD / "results" / "README.md": render_overview(items, storage)}
    for kind in ("absolute", "cqr"):
        selected = [item for item in items if item["config"]["kind"] == kind]
        outputs[METHOD / "results" / kind / "INDEX.md"] = render_index(kind, selected, storage)
    for path, contents in outputs.items():
        def relocated_link(match):
            target = match.group(1)
            if '://' in target or target.startswith('#'):
                return match.group(0)
            try:
                relative = (path.parent / target).resolve().relative_to(ROOT)
            except ValueError:
                return match.group(0)
            moved = ROOT / 'data' / relative
            if not moved.is_file():
                return match.group(0)
            return '](' + os.path.relpath(moved, path.parent).replace('\\', '/') + ')'
        contents = re.sub(r'\]\(([^)\n]+)\)', relocated_link, contents)
        heading, rest = contents.split('\n', 1)
        contents = heading + '\n\n> Numerical files are centralized under root `data/`. This source-side index is included on GitHub; local data is excluded. The earlier staged originals were removed by the author on September 10.\n' + rest
        path.write_text(contents, encoding="utf-8")
    print(json.dumps(totals(items, storage), indent=2))
    print("Wrote results map and indexes; all completion, storage-mode and NPZ-presence counts checked. No result CSVs changed.")


if __name__ == "__main__":
    main()
