# Interpretable Multivariate Conformal Prediction

Research by Yunjie Fan and Matteo Sesia on simple coordinate standardization for simultaneous, interpretable rectangular prediction regions, without an additional calibration split for residual shape estimation.

[Original preprint](https://arxiv.org/abs/2512.15383). The working tree also contains newer signed-envelope development and fresh experiments that are not yet fully integrated into the main JMLR draft.

## Read the project

- [Current envelope/shortcut meeting report and complete report folder](envelope_method/meeting_report/README.md)
- [Project map](docs/PROJECT_MAP.md)
- [Data layout, figure sources and GitHub sharing](docs/DATA_LAYOUT.md)
- [Author's working paper goal](docs/WORKING_PAPER_GOAL.md)
- [Proposed paper experiments and missing evidence](docs/PAPER_EXPERIMENT_PLAN.md)
- [Experiment retention register](docs/EXPERIMENT_RETENTION.md)
- [Archive storage and reproduction](docs/ARCHIVE_STORAGE_GUIDE.md)
- [Storage measurements and decisions](docs/STORAGE_DECISIONS.md), [archive cleanup](docs/ARCHIVE_CLEANUP_REPORT.md), and [earlier cache/duplicate cleanup](docs/CLEANUP_REPORT.md)
- [Earlier full folder and publication audit](output/project_audit_2026-09-09/PROJECT_REVIEW.md)

## Main folders

| Folder | Contents |
|---|---|
| `utility/` | Envelope, original TSCP variants, Point CHR, CQHR and other baselines; data/model/evaluation helpers |
| `multi_target_scaling_latex/` | Main JMLR manuscript, mathematical supplement, figures and reviewer response |
| `envelope_method/` | Current derivation, experiment drivers, settings, result documentation, reports and figures |
| `data/` | All numerical experiment outputs, raw datasets, fitted residual caches and generated audit records; excluded from Git |
| `syn_exps/`, `reviewer_exps/`, `real_exps/` | Navigation/figure interfaces; their numerical files are under the matching `data/` paths |
| `reviewer_update/` | Reviewer study code, publication builders, figures and reference source snapshots |
| `output/` | Portable reports and the dated project audit |
| `docs/` | Current navigation, paper plan, retention decisions and cleanup records |
| `quarantine/` | Historical source/figure context; numerical evidence is under `data/quarantine/` |
| `tmp/` | Local dependencies/compiler and scratch outputs; contains active runtime inputs |

## Current evidence

The main fresh synthetic corpus has 225 configurations and 35,463 fitted trials, plus 1,730 fitted toys and a separate 2,400-trial report study. Eight real cohorts have 200 random splits each. Repeated views and comparator sidecars reuse the same fits and are not independent additional experiments. The prepared ten-output CQR study has six completed pilots; its formal sweep remains unrun.

Start at the [results map](envelope_method/results/README.md), [method results](envelope_method/RESULTS.md), [signed/comparator follow-up](envelope_method/FOLLOWUP_AUDIT.md), and [corrected real comparison](envelope_method/results/REAL_COMPARISON.md). For current real-data numbers use `data/envelope_method/results/real_comparison_audited.csv`; generic historical summaries may contain older Point CHR values.

Current source/CSV updates have not automatically rebuilt the manuscript's figures and quantitative prose. See [sampling provenance](multi_target_scaling_latex/sampling_provenance.md).

## Run and reproduce

Shared scientific dependencies include NumPy, SciPy, pandas, scikit-learn and ucimlrepo. Plotting, notebooks and verification add dependencies described by each workflow; the project does not yet have one unified portable environment.

- [Fresh envelope experiments and commands](envelope_method/README.md)
- [Controlled report suite](envelope_method/report_revision/README.md)
- [Real-coordinate and search diagnostics](reviewer_update/real_diagnostics/README.md)
- [Ten-output CQR notebook and requirements](envelope_method/cqr10/README.md)
- [Manuscript integration/build instructions](multi_target_scaling_latex/README_INTEGRATION.md)

Every formal synthetic trial generates fresh observations and refits; methods within a trial share data and predictions. Full-cell LWC remains manual-only in `envelope_method/full_lwc_manual.ipynb`, disabled by default. No experiment run is triggered by reading these documents or by the cleanup.

New main-runner trials default to **scores** storage for absolute residuals and **compact** storage for CQR. Choose `--storage full` explicitly when raw observations are needed, and use `--out` for a separate run destination. Existing richer archives are preserved on resume; missing scores/full data are reported rather than silently redrawn. See the [storage guide](docs/ARCHIVE_STORAGE_GUIDE.md) for commands, provenance checks and tier-specific reproduction.

## Cleanup policy

All configurations, trial measurements, summaries, comparator results, source code, figures and useful exploratory controls are retained. The absolute/CQR storage migration keeps **1,319 full trial archives (796,388,868 bytes)** and retains all non-X/y members in scores archives for the other existing trials. The author removed the staged `deletable/` originals before the September 10 data reorganization; those rollback copies are no longer locally available.

The [archive cleanup report](docs/ARCHIVE_CLEANUP_REPORT.md) records the earlier storage reduction. All retained numerical files now live under `data/`, with the same experiment identities and file contents. Code, settings, LaTeX inputs and figures remain outside that ignored folder. See [data layout](docs/DATA_LAYOUT.md) for locations and [reorganization verification](docs/DATA_REORGANIZATION.md) for checks.
