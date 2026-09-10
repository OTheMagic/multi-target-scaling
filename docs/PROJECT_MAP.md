# Project map

Numerical results and raw inputs are now centralized under `data/`; source, documentation and figures remain at their existing source locations. See [data layout and GitHub sharing](DATA_LAYOUT.md).

## Start here

1. [Working paper goal](WORKING_PAPER_GOAL.md): the author's current scientific direction.
2. [Paper experiment plan](PAPER_EXPERIMENT_PLAN.md): proposed story, main/appendix experiments and missing comparisons.
3. [Experiment retention register](EXPERIMENT_RETENTION.md): all preserved studies and their paper roles.
4. [Archive storage guide](ARCHIVE_STORAGE_GUIDE.md): full/scores/compact tiers, new-run options, resume checks and restoration.
5. [Archive retention policy](ARCHIVE_RETENTION_POLICY.md): exact full-retention exceptions and rationale.
6. [Archive cleanup report](ARCHIVE_CLEANUP_REPORT.md): staging, verification and actual bytes; [storage decisions](STORAGE_DECISIONS.md) preserves the pre-migration measurements.
7. [Earlier cleanup report](CLEANUP_REPORT.md): the separate cache/render/duplicate phase.
8. [Results provenance](RESULTS_PROVENANCE.md): current synthetic exports, historical real summaries, raw inputs and their relationship to richer trial evidence.

The earlier [full project audit](../output/project_audit_2026-09-09/PROJECT_REVIEW.md) remains a dated diagnostic snapshot. Its proposed narrative is superseded by the working goal above where they differ.

## Research and writing

| Location | Purpose | Best entry point |
|---|---|---|
| `multi_target_scaling_latex/` | Main JMLR draft, proofs and reviewer response | [Main source](../multi_target_scaling_latex/main.tex), [sampling boundary](../multi_target_scaling_latex/sampling_provenance.md) |
| `envelope_method/` | Current standardization/envelope studies and derivation | [Method guide](../envelope_method/README.md), [derivation](../envelope_method/signed_envelope.tex) |
| `utility/` | Shared methods, regions, generators and fitting/evaluation | [Envelope](../utility/envelope.py), [experiment helpers](../utility/exps.py) |
| `envelope_method/results/` | Current trial evidence, comparisons and exploratory controls | [Results map](../envelope_method/results/README.md) |
| `envelope_method/report_revision/` | Controlled six-output report study and serial timing | [Reproduction guide](../envelope_method/report_revision/README.md) |
| `envelope_method/cqr10/` | Prepared ten-output study; formal sweep unrun | [Study guide](../envelope_method/cqr10/README.md) |
| `output/pdf/` | Portable envelope and outlier reading/source packages | [Envelope report](../output/pdf/envelope_shortcut_report/README.md), [outlier report](../output/pdf/outlier_sensitivity_report/README.md) |

## Interfaces, history and infrastructure

| Location | Purpose | Cleanup treatment |
|---|---|---|
| Root notebooks | Synthetic/real/reviewer/smoke interfaces | Retain; outputs are not the canonical experiment record |
| `syn_exps/`, `reviewer_exps/` | Notebook-compatible fresh synthetic exports | Retain; repeated views are not extra independent trials |
| `real_exps/` | Original real tables and ARFF inputs | Retain; use corrected current comparisons for publication |
| `reviewer_update/` | Studies, builders, cached real fits and reference snapshots | Retain sources/caches/snapshots; stage disposable QA renders |
| `quarantine/` | Superseded experiment archives and provenance | Retain pending author judgment; not current experiment input |
| `tmp/` | Active dependencies/compiler plus QA/compile scratch | Mixed contents: do not delete this folder wholesale |
| `docs/cleanup/` | Retention policy, cleanup plans, hash manifests, transaction logs and maintenance scripts | Retain for auditability and restoration |
| `data/` | All numerical results, raw data, caches and audit records | Excluded from Git; contains the retained evidence, not disposable files |

Existing programmatic paths stay stable. Human-readable indexes explain hash-named trial folders and show actual full/scores/compact counts and active bytes; renaming those folders would break settings, manifests and resume logic. All 225 absolute/CQR configurations and 35,463 trial results are retained. The active representation keeps 1,319 full trials unchanged and scores for the others, and the author has removed the separately staged original pairs. New main-runner defaults are scores for absolute and compact for CQR, with explicit `--storage` and `--out` options. Toys, real cohorts and the existing ten-output pilots retain their saved representations.
