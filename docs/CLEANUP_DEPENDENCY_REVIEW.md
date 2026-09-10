# Cleanup dependency review

Reviewed September 9, 2026. This document records a read-only dependency review. No files were moved or removed by the reviewer. The root cleanup process should produce an exact per-file move manifest, preserve relative paths and hashes, and exclude any candidates changed after this review.

## Decision

Move reproducible render/cache duplicates into the cleanup staging directory. Keep source, mathematical and empirical verification records, complete historical source snapshots, active runtime dependencies, and every formal experiment archive/configuration/result in place. A directory named `tmp`, `pre_*`, `qa` or `quarantine` is not automatically disposable.

## Safe candidates for reversible staging

These candidates do not change scientific observations, fitted models, calibration scores, method bounds, result tables, manuscript source or retained JSON/text audit records.

| Existing path/filter | Files and bytes at inspection | Reason and regeneration |
|---|---:|---|
| `tmp/pdfs/**/*.png`, `tmp/pdfs/**/*.jpg`, `tmp/pdfs/**/*.jpeg` | 209 files; 38,770,438 bytes | Raster PDF review pages/contact sheets. Retain their source PDFs, TeX, JSON and extracted text. Re-render before running QA procedures that read PNG pages. |
| `reviewer_update/qa_renders/*.png` | 32 files; 2,889,050 bytes | `reviewer_update/render_figure_qa.py:11` writes these from the retained vector figures; `main()` creates the directory and regenerates pages/contact sheets. |
| `envelope_method/qa/*.png` | 12 files; 3,092,114 bytes | Old report page previews/contact sheet. `envelope_method/qa_document.py:16` regenerates previews; retain `qa/report.json` and `qa/extracted.txt`. The original report PDF remains in `report_revision/`. |
| `tmp/tectonic-0.17.0.zip` | 1 file; 21,060,223 bytes | Redundant download. ZIP entry `tectonic.exe` was read and its SHA-256 matched the kept `tmp/tectonic-0.17.0/tectonic.exe` exactly. Retain the extracted compiler and cache. |
| `tmp/mpl/` | 1 file; 85,807 bytes | Generated Matplotlib font cache; plotting recreates it. |
| `tmp/mpl-diagnostics/` | 1 file; 85,807 bytes | Generated Matplotlib font cache; diagnostics plotting recreates it. |
| `__pycache__/` | 9 files; 64,088 bytes | Interpreter cache. Exclude maintained `.py` source files. |
| `utility/__pycache__/` | 19 files; 279,988 bytes | Interpreter cache; utility sources retained. |
| `reviewer_update/__pycache__/` | 4 files; 115,921 bytes | Interpreter cache; reviewer scripts retained. |
| `reviewer_update/pre_real_diagnostics/__pycache__/` | 1 file; 26,173 bytes | Cache only. The adjacent `res_rescaled.py` is a required regression fixture and must stay. |
| `envelope_method/__pycache__/` | 6 files; 93,583 bytes | Interpreter cache; experiment scripts retained. |
| `tmp/__pycache__/` | 4 files; 36,226 bytes | Cache only. All four adjacent `envelope_*.py` modules remain active source dependencies. |
| `.pytest_cache/` | 5 files; 6,619 bytes | Generated test discovery/last-failure cache; tests do not rely on its retained state. |
| `multi_target_scaling_latex/main.fdb_latexmk` | 37,259 bytes | Generated editor/build tracking file. |
| `multi_target_scaling_latex/main.fls` | 48,683 bytes | Generated TeX file-recorder output. |
| `multi_target_scaling_latex/main.synctex.gz` | 533,383 bytes | Generated editor synchronization data. |
| `multi_target_scaling_latex/revision_cover.fdb_latexmk` | 8,917 bytes | Generated editor/build tracking file. |
| `multi_target_scaling_latex/revision_cover.fls` | 9,255 bytes | Generated TeX file-recorder output. |
| `multi_target_scaling_latex/revision_cover.synctex.gz` | 16,340 bytes | Generated editor synchronization data. |

The raster filter applies only to the three named render locations. Do not generalize it to all PNGs/PDFs in the repository: images can be manuscript figures or retained research evidence elsewhere. Do not generalize the cache filter into installed packages, compiler caches, results or archived cohorts.

Rasters are output caches, so their removal does not require editing the scientific code. The expected workflow remains “render, then inspect.” In particular, `envelope_method/report_revision/qa_report.py:19` expects `tmp/pdfs/envelope_revision_qa/page-XX.png`. The report README documents the preceding `pdftoppm` rendering command. The JSON audit and extracted text should remain in place. `package_report.py:16` directly reads `tmp/pdfs/envelope_revision_qa/report.json`; removing that JSON would break packaging.

## Keep active dependencies in place

### Runtime and prototype source under `tmp`

- **Keep `tmp/diagnostic_packages/`** (13,420 files, 390,223,845 bytes). Existing reproduction instructions and current test execution use it on `PYTHONPATH`; it is an installed scientific environment, not an experiment-output cache. Replacing it with a portable environment is separate work.
- **Keep `tmp/tectonic-0.17.0/tectonic.exe`** (51,538,432 bytes) and **`tmp/tectonic-cache/`** (49,683,276 bytes). Documented builds use this executable and cached packages; `--only-cached` builds require the cache. The ZIP is redundant, but the extracted binary and cached TeX assets are active dependencies (`envelope_method/WORKLOADS.md:143`, `report_revision/README.md:71`).
- **Keep all four `tmp/envelope_*.py` files.** `envelope_method/run_toys.py:8–15` adds `tmp` to the import path and imports `envelope_cqhr_toy`, `envelope_cqhr_misspecified_width`, `envelope_boundary_probe` and `envelope_boundary_stress`. These are live generator/helper modules despite their location. The boundary probe also imports the toy module. They are small and should eventually move into a maintained source package through an explicit import migration, not a cleanup move.
- Keep the two `tmp/envelope_boundary_*_summary.json` files for the retained boundary/prototype audit record. They are only about 20 KB combined.

### Reviewer snapshots and validators

Keep all `reviewer_update/pre_*` directories and the small `pre_integration_response.tex` historical source in this cleanup. These snapshots preserve the author's revisions and, in several cases, are executable validator inputs:

| Snapshot/input | Current consumers |
|---|---|
| `pre_real_diagnostics/res_rescaled.py` | Imported directly by `test_search_diagnostics.py:12` and `validate_real_diagnostics.py:16`; deleting/moving it breaks regression checks. |
| `pre_integration_latex/` | `check_latex_integration.py:14` expands its `main.tex`, compares body/supplement/source hashes and checks its figure assets. |
| `pre_figure_unification/` | `check_publication_experiments.py:10` compares experiment TeX, body, supplementary source and all old figure/data assets; `check_latex_integration.py:52` also reads its author body snapshot. |
| `pre_schematic_labels/figures/` | `check_latex_integration.py:102` compares preserved schematic PDFs/TeX. |
| `pre_final_editorial/` | `check_final_editorial.py:12` expands the snapshot document and compares labels, tables, bibliography/style and every figure/data asset. |
| `pre_publication_experiments/` | Small historical editorial snapshot referenced by retained integration notes; no need to discard it to save meaningful space. |

Do not move duplicate-looking reviewer figure/data folders: validators deliberately compare both retained copies. Source tests and verification scripts are tiny and should remain executable. Some historical validators already flag known stale publication data; cleanup must not introduce additional missing-input failures.

### Manuscript and report build inputs

- Keep manuscript `.tex`, `.bib`, `.sty`, all figure/vector assets, experiment data and final PDFs.
- Keep current `.aux`, `.bbl`, `.blg`, `.log` and `.out` files in this conservative pass. The response imports manuscript labels via `main.aux` (`multi_target_scaling_latex/response_to_reviewers.tex:9`); the canonical build regenerates them, but deleting them changes standalone response compilation until the paper is rebuilt.
- Keep `envelope_method/signed_envelope.log`: both `qa_document.py:33` and `report_revision/qa_report.py:29` read it for warnings/reference checks.
- **Keep `envelope_method/report_revision/original_signed_envelope.tex`.** Despite “original” in its name, `build_report.py:7` reads it as a construction input for the revised report. Keep its PDF alongside it as historical evidence.
- Keep all small report JSON/CSV verification/QA records and their source scripts. `report_revision/package_report.py` reads `run_metadata.json`, `final_qa` inputs, report sources and the current portable report package.

## Old compile-check trees are not wholly duplicate

No active Python/TeX/PowerShell/Markdown reference was found to these directory names in maintained source trees, but source-content comparison found unique historical revisions. They should **remain intact** until the user chooses whether to archive or discard the historical builds.

Both exact-byte hashes and UTF-8/BOM/newline-normalized text hashes were compared against current manuscript and retained reviewer snapshot TeX. The unmatched files below are not merely CRLF differences.

| Tree | Files/bytes | TeX without an identical retained counterpart |
|---|---:|---|
| `tmp/pdfs/final_editorial_clean_20260904/` | 154 / 52,480,475 | `body.tex`, `experiments_appendix.tex`, `experiments_body.tex`, `response_to_reviewers.tex`, `revision_cover.tex`, `supplementary.tex` |
| `tmp/pdfs/final_editorial_clean_verified/` | 154 / 52,480,429 | `body.tex`, `experiments_body.tex`, `revision_cover.tex`, `supplementary.tex` |
| `tmp/pdfs/publication_clean_20260904_031804/` | 146 / 52,512,195 | `body.tex`, `experiments_appendix.tex`, `experiments_body.tex` |
| `tmp/pdfs/style_clean_20260904_043539/` | 154 / 52,602,049 | `body.tex`, `experiments_appendix.tex`, `experiments_body.tex` |
| `tmp/real-diagnostics-compile-check/` | 147 / 52,739,008 | `experiments_body.tex` |

Their experiment-data copies match retained data/snapshot assets, but removing those subtrees would make these historical build snapshots incomplete. Preserving coherent history is preferable to reclaiming roughly 250 MB through piecemeal removals. `tmp/pdfs/final_editorial_sources/` contains four extracted-text files; preserve them as compact historical evidence. Empty output directories are harmless and need no special action.

## Quarantine and formal data are not cleanup candidates

Keep `quarantine/obsolete_synthetic_2026-09-08/` in place: 87,058 files and 3,004,557,669 bytes at inspection. It preserves old cohorts, corrections, source outputs and manifests, not active formal evidence to be mixed into current analysis.

It is also a live historical-audit dependency. `envelope_method/audit_comparator_reruns.py:10` fixes its root, reads its `manifest.json` at line 90, resolves archived file paths and reads archived `simulation_trials.csv` and `configurations.csv` at lines 150–152. Maintenance/migration scripts and boundary-correction preservation also write relative to that root. Moving it without changing all consumers and retaining relative structure would break audit reconstruction.

No formal NPZ/JSON per-trial archives, fitted-model files, settings, status/checkpoint records, result summaries, pilots, raw datasets, split caches, manuscript experiment data or correction sidecars were approved for staging in this review.

## Possible later organization, not needed for safe staging

The safe candidates above need no scientific path changes. Leave historical snapshot and quarantine paths unchanged now. If relocation is later desired, introduce one explicit archive-root resolver, preserve internal relative paths and manifests, update every validator and README, and rerun read-only path/hash checks before moving anything. The prototype generators should migrate into a small maintained module with updated imports and an import smoke check. Those are ordinary code changes with a reviewable diff, rather than justification for moving active files blindly.

The next organizational step can instead be documentation: a canonical entry-point README, a human-readable experiment/configuration index with absolute local paths, and a staging manifest showing exactly what is safe to delete and how to restore it. That improves navigation without changing the hashed experiment trees.
