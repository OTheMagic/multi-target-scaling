# Signed surface-envelope TSCP

**Current meeting report:** [meeting_report/report.pdf](meeting_report/report.pdf)
is the 10-page revision with plain paper typography, the original link and its
signed extension, GWC/full LWC review, detailed envelope construction, a separate
uniform-improvement proof, and focused absolute/signed comparisons. Its
[single-folder reproduction guide](meeting_report/README.md) links the TeX,
figures, local data archives, code and audits, including the new strong shifted
old-method comparisons. The earlier 20-page report remains at
`signed_envelope.pdf`, with its historical study at `report_revision/`.

The mathematical specification is [signed_envelope.tex](signed_envelope.tex).
It derives signed GWC, the surface-envelope refinement, finite-sample coverage,
positive-score containment in the old shortcut, a certified range for binary
search, and a direct order-statistic localization rule. The compiled
[reading copy](signed_envelope.pdf) is included. Start with [RESULTS.md](RESULTS.md)
for the experimental findings and qualifications.
The follow-up [signed-gain and comparator audit](FOLLOWUP_AUDIT.md) separates
the effect of signing scores from the envelope improvement and records the
baseline omissions filled on September 9.

The proof of uniform improvement concerns the **same nonnegative residuals**,
positive calibration scales, and compatible partitions and tie conventions.
It means a weakly smaller region with coverage still at least the target. It
does not mean higher coverage, strict improvement on every sample, dominance
over CQHR, or dominance when comparing signed scores with capped scores.

## Implementation

`utility/envelope.py` provides `EnvelopeCalibration`, `envelope_prediction`,
`interval_sup`, `score_at`, and `link`. Examples from the repository root:

```python
from utility.envelope import EnvelopeCalibration, envelope_prediction

# Absolute residuals, or any nonnegative score with attainable lower bound 0.
region = envelope_prediction(calibration_scores, alpha=0.1)
upper = region.upper

# Raw CQR residuals. Each test covariate has its own attainable lower bound.
calibrator = EnvelopeCalibration(raw_cqr_calibration_scores, alpha=0.1)
region = calibrator.predict(lower=-base_interval_lengths_at_x / 2)
if region.empty:
    prediction = None
else:
    lower_outcome = fitted_lower_quantiles_at_x - region.upper
    upper_outcome = fitted_upper_quantiles_at_x + region.upper
```

No random jitter is used. Closed cells retain boundary ties. A below-domain
threshold yields an empty set, not a clamped singleton. The conformal rank is
`ceil((n+1)*(1-alpha))`, with an infinite threshold when the rank exceeds `n`.
The conservative zero-calibration-scale fallback returns the whole domain.

The default implementation uses exact backward search. The TeX note proves
why the rightmost surviving cell maximizes the endpoint, even without a prefix
predicate. The optional binary-search result has an explicitly stated domain;
the implementation does not assume an unproved global prefix property.
`search="rank"` additionally uses the proved transformed-order-statistic rule
to discard a known-failing suffix, then runs the same exact backward scan.
It agrees with backward search in the 9,200 saved formula cases. It is not
necessarily faster when backward search already checks one cell.

The old shortcut's mean-cell indexing has been corrected for tied zeros.
Its former `argmax` could select `[0,0]` instead of a mean-containing cell.
Six failing capped-CQR comparisons motivated this fix; all six pass afterward.
The dominance theorem concerns the correctly defined old shortcut, not that bug.
The envelope also checks the forward self-score at finite cell boundaries.
This retains an exactly accepted zero when inverse-link cancellation would
round it slightly negative. It does not clamp genuinely negative signed bounds
to zero. Nine hundred exact-zero regression cases cover this edge condition.

## Separate Workloads

[WORKLOADS.md](WORKLOADS.md) records progress and resumption information.

| Workload | Runner | Main saved output |
| --- | --- | --- |
| Formula and containment checks | `verify.py` | `verification.json` |
| Original and reviewer simulations | `experiments.py` | `results/absolute/`, `results/cqr/` |
| Cached real-data cohorts and alpha sweeps | `run_real.py` | `results/real/` |
| Uncached air and crime cohorts | `run_extra_real.py` | `results/extra_real/` |
| Auxiliary original baselines and 2D local unions | `run_auxiliary.py` | `results/auxiliary/` |
| Exploratory notebook configurations | `run_notebook_pilots.py` | `notebook_settings.json`, trial directories |
| Recent signed/CQHR/positive toys | `run_toys.py` | `results/toys/` |
| Consolidation and pairing checks | `summarize.py` | `results/*_summary.csv`, `results/audit.json` |
| Document rendering audit | `qa_document.py` | `qa/report.json` |
| Tier-aware archive integrity and available fresh-data checks | `final_audit.py` | `results/retained_storage_audit.json`; old `final_audit.json` remains historical |
| Notebook compilation and safe manual handoff | `validate_notebooks.py` | `notebook_validation.json` |
| Missing CQR/toy comparators on fresh paired fits | `run_cqr_baselines.py` | `results/cqr_baselines/`, `results/toy_baselines/` |
| Baseline-only omitted configurations | `run_comparator_repairs.py` | `repair_settings.json`, new trial directories |
| All archived comparator requirements | `audit_comparator_reruns.py` | `results/comparator_rerun_audit.csv` |

Full LWC is now **manual-only**, as requested. Use `full_lwc_manual.ipynb`;
its run switch is off by default and it uses one worker. No full-LWC work is
started by the normal auxiliary runner. Results completed before the pause
request are preserved separately.

`settings.json` and `notebook_settings.json` contain the frozen simulation
configuration, its original source tables, its repetitions, and its CQR score
transformations. `results/configurations.csv` gives readable labels for the
configuration directory IDs. Common configurations are computed once and mapped
to every corresponding study. Original repetition counts are preserved.

Each simulation directory contains its configuration, per-trial JSON
measurements, trial/summary CSVs and completion status. Array retention is
explicit: **full** NPZs save all trial arrays; **scores** NPZs omit only the six
training/calibration/test X/y arrays; **compact** checkpoints intentionally have
no NPZ. New `run_spec` trials default to scores for absolute and compact for CQR.
Use `--storage` to choose a tier and `--out` for a separate result root.

All 225 existing absolute/CQR configurations and 35,463 trial results remain.
The [retention policy](../docs/ARCHIVE_RETENTION_POLICY.md) keeps 1,319 full NPZs
unchanged (796,388,868 bytes); the other existing trials retain scores and every
original non-X/y member for unfinished method comparisons. Their original full
NPZ/JSON pairs are staged under `deletable/` with a restore path. Toys, real
cohorts/controls, and existing ten-output pilots are unchanged. See the
[storage guide](../docs/ARCHIVE_STORAGE_GUIDE.md) and
[archive cleanup report](../docs/ARCHIVE_CLEANUP_REPORT.md).

Resumption validates completed records and scores/full NPZ integrity before
skipping a trial; missing data are not automatically redrawn. Richer existing
tiers remain unchanged when a lower tier is requested. Raw-data/refit checks
need full arrays, whereas reporting uses saved JSON/CSV measurements. Cached
real-data replays still store source paths and SHA-256 hashes instead of
duplicating the existing residual archives. Original obsolete data are preserved
in the review folder below; fresh replacements are explicitly labeled.

Volumes in the new trial tables are **outcome-space volumes**. For absolute
residuals this is `2**d` times the positive residual-box volume used by many old
tables. Signed CQR volumes include the original base lengths. The 2D local-union
records use union volume and joint coverage; their coordinate lengths describe
the enclosing box, as marked in each record. Timing under concurrent workloads
should not be interpreted as a controlled hardware benchmark.

## Provenance

**Every synthetic empirical trial now generates fresh training, calibration,
and test data and refits the model. There is no selectable reused-pool mode.**
The original scenario sample counts are retained in the frozen configurations;
they do not denote a shared pool. Only fixed DGP coefficients are retained across
trials. Comparisons within a trial use identical data and fitted predictions.
Real-data resplitting is unchanged.

New main-runner checkpoints record Python/package versions and selected source
hashes. Resumption rejects a mismatch with recorded execution provenance and
uses a per-configuration lock to prevent concurrent writes. Historical
checkpoints without that provenance remain unknown; migration does not claim
they were generated by today's code. For migrated archives, the original
`storage.source_archive_sha256` preserves comparator lineage, while
`archive_sha256` checks the active reduced NPZ. Source identity alone does not
verify current file bytes. See the storage guide for limitations and restoration.

`../quarantine/obsolete_synthetic_2026-09-08/` contains known or suspected
obsolete synthetic results, their derived aggregates, and archived notebook
outputs. `manifest.json` records paths, reasons, sizes, and SHA-256 hashes.
Nothing in that folder is required to execute the fresh experiment runners.
`additional_manifest.json` covers the retired score-only toy results and code;
`notebook_outputs/` preserves their former notebook execution displays.

All empirical toys now train an OLS center on 1,000 fresh observations per
trial. CQR toys additionally estimate coordinate width multipliers from the
training-residual quantiles. The supplied width covariates retain the intended
homogeneous, heterogeneous, or misspecified shape; both methods receive the
same fitted intervals. Gaussian toys retain their specified signed Gaussian
dependence. These are new fitted replications, not exact score-only replays.
The conservative signed
toy has base marginal miscoverage 0.02, matching its purpose as a shrinkage
example. The Gaussian CQHR toys use base miscoverage 0.1 for both methods.
The larger gradient-boosted CQR studies retain their original base-alpha sweeps and additionally run
the requested base miscoverage 0.1.

## Reporting and reproduction

In this workspace, use the bundled Python executable and the pre-existing
scientific dependency directory:

```powershell
$env:PYTHONPATH='E:\multi-target-scaling\tmp\diagnostic_packages'
$python='C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $python envelope_method/verify.py
& $python envelope_method/summarize.py --summaries-only
& $python envelope_method/final_audit.py
& $python envelope_method/plot_toys.py
& $python envelope_method/report_results.py
& $python envelope_method/validate_notebooks.py
```

These commands use saved results or mathematical checks. New experiments are
separate operations. For a new primary grid, select storage and output explicitly:

```powershell
& $python envelope_method/experiments.py --kind absolute --storage scores --out envelope_method/results_new --workers 4
& $python envelope_method/experiments.py --kind cqr --storage compact --out envelope_method/results_new --workers 4
# Choose scores for future CQR comparisons or full for raw-data demonstrations.
& $python envelope_method/experiments.py --kind cqr --storage scores --out envelope_method/results_cqr_scores --workers 4
& $python envelope_method/experiments.py --kind absolute --storage full --out envelope_method/results_full_demo --limit 1 --workers 1
```

`--limit 1` selects the first configuration, with its configured repetitions.
For a particular configuration, select its settings item and call
`run_spec(item, storage_mode="full", output_root=Path("..."))`. A completed
compact/scores trial cannot silently be upgraded to full; use a separate output
root and a new fit. The standard CLI uses `settings.json`; notebook and repair
studies retain their separate drivers. The following are individual workload
commands, including fitting and backfills, rather than prerequisites for viewing
the completed evidence:

```powershell
& $python envelope_method/run_real.py
& $python envelope_method/run_extra_real.py prepare
& $python envelope_method/run_extra_real.py run --workers 4
& $python envelope_method/run_auxiliary.py
& $python envelope_method/run_notebook_pilots.py
& $python envelope_method/run_toys.py cqr
& $python envelope_method/run_toys.py positive
& $python envelope_method/run_toys.py boundary
& $python envelope_method/run_comparator_repairs.py
& $python envelope_method/run_cqr_baselines.py
& $python envelope_method/verify_comparator_backfills.py
& $python envelope_method/audit_rank_search.py
& $python envelope_method/test_protocol.py
& $python envelope_method/summarize.py
& $python envelope_method/export_fresh_tables.py
& $python envelope_method/audit_comparator_reruns.py
& $python envelope_method/signed_gain_analysis.py
```

The historical full-archive audit passed for 225 configurations / 35,463 trials,
including the 33 added trials in two configurations. Its evidence is recorded
in `results/final_audit.json`. The current `final_audit.py` writes
`results/retained_storage_audit.json`, verifies each active storage tier and
reports raw-data checks unavailable when X/y was not retained. The comparator
audit covers 3,402 archived
method/scenario entries: 3,401 complete and one signed full-LWC illustration
deferred. These include duplicate historical views, not 3,402 independent studies.
The revised standalone report compiles to 20 pages without unresolved references or
overfull boxes. Use `summarize.py --summaries-only` to refresh the small summary
and score-representation tables from saved measurements. No normal
reporting command launches full LWC.

## Ten-outcome extension (2026-09-09)

The new [local notebook](cqr10/cqr10_experiments.ipynb) extends the primary CQR
grids to ten outcomes with stronger, quality-checked quantile models. The full
study is not started automatically: 1,430 fresh fitted trials are estimated at
11.8 single-worker hours. See the [protocol and saved-data guide](cqr10/README.md)
and [per-experiment time estimates](cqr10/RUNTIME_ESTIMATES.md). Previous results
remain unchanged; six timing/quality pilots are saved separately.

For cluster execution, use the validated
[source bundle](../output/cqr10_cluster_source.zip) and
[manifest](../output/cqr10_cluster_source_manifest.json). Its included
`README_CLUSTER.md` covers installation, compact storage and an explicit scratch
output directory. The bundle contains source and documentation only; existing
results, pilots and fitted models remain in this research workspace.
