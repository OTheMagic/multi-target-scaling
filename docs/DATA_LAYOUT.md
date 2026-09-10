# Data layout and GitHub sharing

Updated September 10, 2026. All experiment tables, trial checkpoints, raw datasets,
model/residual caches and generated numerical audit records live under the root
`data/` folder. `.gitignore` excludes that folder. Source code, experiment settings,
Markdown documentation, LaTeX inputs and figure assets remain outside it.

Data keeps its original repository-relative path below `data/`. For example:

```text
multi-target-scaling/
  data/                              # local; excluded from Git
    envelope_method/
      results/                       # current main experiments and controls
      report_revision/               # separate 2,400-trial report study
      meeting_report/                # meeting report's saved evidence
      cqr10/                         # pilots, estimates and future cluster results
    real_exps/                       # original summaries + raw ARFF datasets
    syn_exps/                        # current synthetic summary exports
    reviewer_exps/                   # reviewer notebook summary/trial exports
    reviewer_update/                # cached real fits and figure-source tables
    multi_target_scaling_latex/      # evidence associated with manuscript figures
    quarantine/                     # explicitly historical experiment evidence
    output/                         # reading-package evidence and audit records
    docs/cleanup/                   # earlier cleanup manifests and checksums
    _organization/2026-09-10/        # relocation manifest and verification
  utility/                          # shared implementation and path helper
  envelope_method/                   # code, settings, reports and figures
  multi_target_scaling_latex/        # manuscript, bibliography and figures
  reviewer_update/                   # builders and historical source snapshots
  output/pdf/                       # compiled reports and their LaTeX/figures
  docs/                             # project map, scientific plan and data guide
  tmp/                              # local runtimes/build scratch; excluded
```

## Which data to use

| Data location, relative to `data/` | Meaning |
|---|---|
| `envelope_method/results/absolute/`, `cqr/`, `toys/` | Main retained synthetic trials; existing full/scores/compact storage contracts still apply |
| `envelope_method/results/auxiliary/`, `full_lwc_scaling/`, `cqr_baselines/`, `toy_baselines/` | Comparators on the associated fitted trials, not additional independent replications |
| `envelope_method/results/real/`, `extra_real/` | Current saved real-data evaluations |
| `envelope_method/results/real_comparison_audited.csv` | Corrected eight-dataset real comparison for the paper |
| `envelope_method/results/rf2_*/`, `real_outlier_screen/` | Outlier and model/data sensitivity investigations |
| `real_exps/data/` | Four raw ARFF datasets required by supported real-data loaders |
| `real_exps/*.csv` | Original historical real summaries; not substitutes for the audited current comparison |
| `syn_exps/`, `reviewer_exps/` | Derived notebook-compatible exports; retain their documented volume conventions |
| `reviewer_update/real_diagnostics/cache/` | Cached data, exact splits and residuals for six reviewer real cohorts |
| `quarantine/obsolete_synthetic_2026-09-08/` | Superseded experiments retained for historical scrutiny; not the current paper evidence |

The [results provenance guide](RESULTS_PROVENANCE.md) explains which tables are
derived views and which are genuinely different reruns. The [paper experiment
plan](PAPER_EXPERIMENT_PLAN.md) explains their scientific roles and missing work.

## Data associated with LaTeX figures

The numerical evidence stays under `data/`; compilation uses the saved PDF figures
and generated `.tex` tables. A reader can compile the retained documents without
copying the experiment data into the LaTeX folder.

| LaTeX document folder | Retained figure assets | Associated data, relative to `data/` |
|---|---|---|
| `multi_target_scaling_latex/` | `figures/`; generated table inputs also retained | `multi_target_scaling_latex/experiment_data/` is the manuscript-associated snapshot; builders read `reviewer_update/data/`, `reviewer_update/real_diagnostics/data/`, `syn_exps/`, `reviewer_exps/` and original real summary exports |
| `envelope_method/` | `figures/report_revision/` for `signed_envelope.tex`, plus other study figures | `envelope_method/report_revision/` for this report; `envelope_method/results/` for broader studies |
| `envelope_method/report_revision/` | `figures/` and local generated `.tex` tables | `envelope_method/report_revision/` |
| `envelope_method/meeting_report/` | `figures/` and `tex/` | `envelope_method/meeting_report/data/` and associated manifests |
| `output/pdf/envelope_shortcut_report/` | `figures/` and generated table inputs | `output/pdf/envelope_shortcut_report/evidence/`; full study in `envelope_method/report_revision/` |
| `output/pdf/outlier_sensitivity_report/` | `figures/` | `output/pdf/outlier_sensitivity_report/data/`; full controls in `envelope_method/results/` |
| `reviewer_update/pre_*/` | Historical source/figure snapshots stay in place | Matching `reviewer_update/pre_*/experiment_data/` preserves associated numerical snapshots |

Current experimental exports and some saved manuscript figures were already out
of sync before this reorganization. Moving files does not refresh the figures or
endorse their old numerical labels; that remains part of the planned figure work.

## Running scripts and sharing the repository

Current runners and figure builders now use the centralized data locations.
`utility.project_paths.data_path(...)` defines new numerical destinations;
`resolve_artifact(...)` finds historical source references at their relocated
paths. Saved metadata and checksums are preserved unchanged, including original
path strings. The resolver recognizes the recorded original Windows checkout
prefix so these references also work after copying the project elsewhere.

For new experiments, put explicit `--out` destinations under `data/`, or use a
separate cluster scratch directory. Storage mode remains an independent choice:
compact for routine CQR results, scores where further residual comparisons are
needed, and full for selected demonstrations. Run a changed implementation in a
fresh output location when the runner's provenance checks require it.

Commit source, settings, documentation and figure assets normally. `data/` and
`tmp/` are ignored. A source-only clone can compile the reports and start new
experiments after installing dependencies; reproducing existing numerical plots
requires obtaining the separate `data/` tree and placing it at the project root.
No data hosting service, Git commit or remote push is performed by this cleanup.

The original `deletable/` folder was removed by the author before this task. Its
old rollback copies are no longer locally available. The retained full/scores
archives were moved unchanged; this reorganization removes no scientific data.
