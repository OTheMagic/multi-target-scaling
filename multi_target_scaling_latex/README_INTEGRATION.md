# Integrated Manuscript

## Main Files

- `main.tex`: normal manuscript entry point; its content is unchanged.
- `body.tex`: the numerical section inputs `experiments_body.tex`; the author's subsequent non-experimental edits are preserved.
- `supplementary.tex`: algorithms and proofs are unchanged; the numerical section inputs `experiments_appendix.tex`.
- `experiments_body.tex`: self-contained main experiments organized around coordinate adaptation, dependence, partial heteroskedasticity, CQR, real-data performance, and computation; includes the two-point energy illustration.
- `experiments_appendix.tex`: one unified Appendix C containing all distribution benchmarks, sensitivity and stress studies, oracle/approximation comparisons, computational scaling, real-data coverage, and Monte Carlo uncertainty. Earlier experiments are integrated by topic, not isolated in a legacy appendix.
- `figures`: all original experimental assets plus 19 new PDF figures (15 earlier figures and four follow-up real-data diagnostics). The two original methodological schematics and their TikZ sources now include the requested R2.m1/R2.m3 labels.
- `experiment_data`: supporting CSV records for the new experiments; `real_diagnostics` contains the follow-up trial data, summaries, generated tables, dataset metadata, manifest, and validation report.
- `response_to_reviewers.tex`: point-by-point response with actual manuscript references and explicit completion statuses.
- `unresolved_reviewer_concerns.md`: author action queue, including non-experimental issues not changed here.
- `revision_cover.tex`: section-by-section change summary relative to arXiv v1, with author-check labels.
- `integration_audit.json`: machine-checked preservation and dependency inventory.
- `publication_edit_audit.json`: preservation of experimental figure inclusions, table inputs, numerical tables, labels, and 114 figure/data assets during the editorial reorganization.

## Build

With TeX Live or MiKTeX and PowerShell:

```powershell
.\build.ps1 -Engine pdflatex
```

Or run `pdflatex main`, `bibtex main`, and `pdflatex main` twice, followed by two LaTeX passes each for `response_to_reviewers.tex` and `revision_cover.tex`.

The response imports labels from `main.aux` and `supplementary.aux`, so compile the manuscript first. The response and cover are standalone documents, not pages inserted into the manuscript.

With a portable Tectonic executable:

```powershell
.\build.ps1 -Engine tectonic -TectonicPath 'C:\path\to\tectonic.exe'
```

`compile_main.tex` supplies engine-compatibility punctuation mappings and unique appendix hyperlink destinations without editing the manuscript sections. The script copies its compiled outputs to the conventional `main.*` names before building the response and cover. Tectonic may need network access to fetch standard TeX packages on its first run; it does not upload the manuscript. A compiler binary is not bundled in this manuscript folder.

The entry point uses the supplied `jmlr2e.sty`, standard LaTeX packages, and `biblio.bib`. The only bibliography addition is the reference needed for the new shape-template experiment. No compile-time figure or source input points outside this folder.

## Interpretation and Preservation

No experiment was deleted. Both real-data table presentations are retained: the compact dataset-level table is in the unified appendix and the expanded table is in the main text. The original energy example remains in the body. Unused original figure assets and commented-out table source remain available. The contents of `experiments_legacy.tex` have been incorporated into the two experiment files; the pre-edit file is preserved in `../reviewer_update/pre_publication_experiments` and is no longer a compilation dependency. Revision-history explanations belong to the cover and response, not to the scientific narrative.

The fixed-pool synthetic benchmarks retain their data and resampling protocol; the independent-observation studies use fresh training/calibration/test draws. These protocols are explicitly distinguished in Appendix C.1 without describing the history of the revision. The uncertainty analysis covers its stated 58 displayed settings, not every experimental result. Heavy-tail and capped-score results are not claimed to extend the standing assumptions.

The follow-up real-data cohort uses 200 reruns on each of the six fixed datasets, with the original 75%/5%/20% split proportions, hash seeds, and statistical model settings. It adds 40,800 split--method--coordinate records and a 6,000-configuration computational stress sweep. It is a new rerun, not a reconstruction of the original fitted models or timing environment. Its uncertainty statements concern random splits conditional on a fixed dataset. Reproduction instructions are in `../reviewer_update/real_diagnostics/README.md`; the cached data and residuals remain outside this compilation folder.

The response now records 13 addressed, six partially addressed, and one open reviewer point. The follow-up experiments resolve the missing backward-search, fallback-frequency, and aggregate real-data coordinate evidence. The Figure 1/Figure 2 annotations resolve R2.m1 and R2.m3. The latest author edits also correct the local conditional-coverage inference, remove the unqualified shift-invariance wording, and clarify that the intended tightness claim concerns coverage balance. The response and audit acknowledge these changes while retaining the outstanding balance, tie-handling, and piecewise-proof questions.

Figure 1 now labels the right-panel marginal coverages as 92% for Y1 and 93% for Y2, with an explicit note that all coverage percentages are illustrative. Figure 2 labels the solid enclosure's residual thresholds, mathcal L_1 and mathcal L_2, at the existing axis endpoints. New annotations are blue. The original drawing geometry and all surrounding manuscript prose are retained; pre-edit PDFs and sources are preserved in `../reviewer_update/pre_schematic_labels`.

The initial source snapshot is outside this folder at `../reviewer_update/pre_integration_latex`. The publication-edit snapshot, including the latest author body source, is at `../reviewer_update/pre_publication_experiments`. The validators are `../reviewer_update/check_latex_integration.py` and `../reviewer_update/check_publication_experiments.py`; none of these external files is needed to compile the paper.

This folder is buildable, but the scientific revision is not submission-ready until the open/partial reviewer concerns and author-check labels are resolved. The provided response explicitly distinguishes these states.

## Verification on September 4, 2026

- Tectonic 0.17.0 successfully built the manuscript (70 pages), response (12 pages), and cover (one page). A separate source-only build also passed without old auxiliary files. The pdfLaTeX path is provided but was not tested because pdfLaTeX is not installed on this machine.
- All 136 manuscript labels and all citations resolve, as do the manuscript references in all 22 response blocks (two editor requests and 20 reviewer points). The missing Figure 1 label was restored with the author's permission; no other body-source edits were made during this pass. Both preservation audits passed.
- All document pages were rendered for layout review. The reorganized experiment sections, distribution tables, appendix transitions, updated response blocks, and one-page cover were checked visually. The two revised schematic PDFs and their placement in the manuscript were also checked at full-page scale. No experiment-section overflow warnings remain.
- All 63 search-instrumentation tests passed. Full numerical replay verified the 1,200 real-data split partitions and exact TSCP rectangle equality against the pre-instrumentation implementation, as well as coordinate metrics and branch counts. The six new TSCP joint-coverage means all have 0.9 within one across-split standard deviation; this diagnostic is not a confidence interval for the mean.
- The manuscript retains two overfull proof displays in the protected supplement (lines 349 and 410). They remain legible. Existing class/package and underfull-box warnings are non-blocking. Tectonic also attempts an unnecessary separate BibTeX pass on `supplementary.aux`; its missing-bibliography warnings do not affect the successful main bibliography build.
- The local compiler used for verification is `E:\multi-target-scaling\tmp\tectonic-0.17.0\tectonic.exe`. It is outside the manuscript folder and is not required when using an existing TeX installation.
