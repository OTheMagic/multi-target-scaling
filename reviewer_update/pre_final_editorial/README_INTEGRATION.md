# Integrated Manuscript

## Main Files

- `main.tex`: normal manuscript entry point; its content is unchanged.
- `body.tex`: the numerical section inputs `experiments_body.tex`; the author's subsequent non-experimental edits are preserved.
- `supplementary.tex`: algorithms and proofs are unchanged; the numerical section inputs `experiments_appendix.tex`.
- `experiments_body.tex`: self-contained main experiments organized around coordinate adaptation, dependence, partial heteroskedasticity, CQR, real-data performance, and computation; includes the two-point energy illustration.
- `experiments_appendix.tex`: one unified Appendix C containing all distribution benchmarks, sensitivity and stress studies, oracle/approximation comparisons, computational scaling, real-data coverage, and Monte Carlo uncertainty. Earlier experiments are integrated by topic, not isolated in a legacy appendix.
- `figures`: 27 active experimental PDFs with consistent `fig_body_*` and `fig_app_*` names. Eight archived plots are redrawn in the newer style, and the four real-data diagnostic plots now share its style helpers. Original PDFs remain available in their subdirectories but are not referenced by the experiment sections. The two methodological schematics retain the R2.m1/R2.m3 annotations.
- `experiment_data`: supporting CSV records for the new experiments; `real_diagnostics` contains the follow-up trial data, summaries, generated tables, dataset metadata, manifest, and validation report.
- `response_to_reviewers.tex`: point-by-point response with actual manuscript references and explicit completion statuses.
- `unresolved_reviewer_concerns.md`: author action queue, including non-experimental issues not changed here.
- `revision_cover.tex`: section-by-section change summary relative to arXiv v1, with author-check labels.
- `integration_audit.json`: machine-checked preservation and dependency inventory.
- `publication_edit_audit.json`: preservation of experimental content during the latest author's ordering and figure-style pass, allowing the documented figure renaming and restyling.
- `figure_style_audit.json`: source CSV hashes, plotted values, method styling, and old-to-new figure filename mapping.
- `sampling_provenance.md`: unresolved reconciliation of archived CSV generation with the author's stated sampling design. This is author-facing, not part of the paper.

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

The manuscript presents one synthetic sampling design: independent generation in each repetition, with sample sizes specified by experiment. The earlier two-protocol presentation is removed. The archived CSV provenance has not yet been reconciled with that declaration; the current notebook places its training/test generator outside the repetition loop. See `sampling_provenance.md` before submission. The uncertainty analysis retains its explicit scope of 58 settings and does not silently expand to all archived results. Heavy-tail and capped-score results are not claimed to extend the standing assumptions.

The follow-up real-data cohort uses 200 reruns on each of the six fixed datasets, with the original 75%/5%/20% split proportions, hash seeds, and statistical model settings. It adds 40,800 split--method--coordinate records and a 6,000-configuration computational stress sweep. It is a new rerun, not a reconstruction of the original fitted models or timing environment. Its uncertainty statements concern random splits conditional on a fixed dataset. Reproduction instructions are in `../reviewer_update/real_diagnostics/README.md`; the cached data and residuals remain outside this compilation folder.

The response now records 13 addressed, six partially addressed, and one open reviewer point. The follow-up experiments resolve the missing backward-search, fallback-frequency, and aggregate real-data coordinate evidence. The Figure 1/Figure 2 annotations resolve R2.m1 and R2.m3. The latest author edits also correct the local conditional-coverage inference, remove the unqualified shift-invariance wording, and clarify that the intended tightness claim concerns coverage balance. The response and audit acknowledge these changes while retaining the outstanding balance, tie-handling, and piecewise-proof questions.

Figure 1 now labels the right-panel marginal coverages as 92% for Y1 and 93% for Y2, with an explicit note that all coverage percentages are illustrative. Figure 2 labels the solid enclosure's residual thresholds, mathcal L_1 and mathcal L_2, at the existing axis endpoints. New annotations are blue. The original drawing geometry and all surrounding manuscript prose are retained; pre-edit PDFs and sources are preserved in `../reviewer_update/pre_schematic_labels`.

The initial source snapshot is outside this folder at `../reviewer_update/pre_integration_latex`; subsequent snapshots are `pre_publication_experiments` and `pre_figure_unification`. The latest snapshot preserves the author's partially reordered experiment sections. The validators are `../reviewer_update/check_latex_integration.py`, `../reviewer_update/check_publication_experiments.py`, and `../reviewer_update/test_figure_unification.py`; none is needed to compile the paper.

`../reviewer_update/unify_experiment_figures.py` reproduces the eight renamed plots and four real-data diagnostic figures from saved summaries without running simulations. It shares the palette, marker mapping, font settings, and panel formatting in `build_experiment_update.py`. All joint-coverage plots use the range [0.6, 1.0]; values below the display range are explicitly marked. Error bars are omitted. Volumes keep their existing residual-space/outcome-space definitions, and construction times use milliseconds. No numerical source CSV or table was modified.

This folder is buildable, but the scientific revision is not submission-ready until the open/partial reviewer concerns and author-check labels are resolved. The provided response explicitly distinguishes these states.

## Verification on September 4, 2026

- Normal and fresh source-only Tectonic 0.17.0 builds both pass, producing the 70-page manuscript, 12-page response, and one-page cover with no unresolved references or citations. The pdfLaTeX path is provided but was not tested because pdfLaTeX is not installed on this machine.
- Both preservation audits and all five figure-style tests pass. The source audit resolves 134 manuscript labels and all citations, including the references in all 22 response blocks. The missing Figure 1 label was restored under the author's prior approval; no other body-source edits were made during this pass.
- All document pages were rendered for layout review. The reorganized experiment sections, distribution tables, appendix transitions, updated response blocks, and one-page cover were checked visually. The two revised schematic PDFs and their placement in the manuscript were also checked at full-page scale. No experiment-section overflow warnings remain.
- All 63 search-instrumentation tests passed. Full numerical replay verified the 1,200 real-data split partitions and exact TSCP rectangle equality against the pre-instrumentation implementation, as well as coordinate metrics and branch counts. The six new TSCP joint-coverage means all have 0.9 within one across-split standard deviation; this diagnostic is not a confidence interval for the mean.
- The manuscript retains two overfull proof displays in the protected supplement (lines 349 and 410). They remain legible. Existing class/package and underfull-box warnings are non-blocking. Tectonic also attempts an unnecessary separate BibTeX pass on `supplementary.aux`; its missing-bibliography warnings do not affect the successful main bibliography build.
- The local compiler used for verification is `E:\multi-target-scaling\tmp\tectonic-0.17.0\tectonic.exe`. It is outside the manuscript folder and is not required when using an existing TeX installation.
