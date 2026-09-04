# Integrated Manuscript

## Main Files

- `main.tex`: normal manuscript entry point; its content is unchanged.
- `body.tex`: the original experiment section is replaced by an input of `experiments_body.tex`; all other sections are unchanged.
- `supplementary.tex`: algorithms and proofs are unchanged; the numerical section inputs `experiments_appendix.tex`.
- `experiments_body.tex`: reviewer-focused main experiments and the original two-point energy illustration.
- `experiments_appendix.tex`: new sensitivity studies, followed by `experiments_legacy.tex`.
- `experiments_legacy.tex`: all retained older experiments, with historical-protocol and coverage-interpretation clarifications.
- `figures`: all original assets plus the 15 new PDF figures.
- `experiment_data`: supporting CSV records for the new experiments.
- `response_to_reviewers.tex`: point-by-point response with actual manuscript references and explicit completion statuses.
- `unresolved_reviewer_concerns.md`: author action queue, including non-experimental issues not changed here.
- `revision_cover.tex`: section-by-section change summary relative to arXiv v1, with author-check labels.
- `integration_audit.json`: machine-checked preservation and dependency inventory.

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

No experiment was deleted. The old real-data table is retained in the historical appendix even though those measurements are also presented in the new main table. The original energy example remains in the body. Unused original figure assets and commented-out table source remain available.

Historical synthetic graphs retain their original data and resampling protocol; the new studies use independent training/calibration/test draws. The new uncertainty audit covers its stated 58 displayed points, not every historical result. Heavy-tail and capped-score results are not claimed to extend the standing assumptions.

The preserved source snapshot is outside this folder at `../reviewer_update/pre_integration_latex`. The integration validator is `../reviewer_update/check_latex_integration.py`; neither is needed to compile the paper.

This folder is buildable, but the scientific revision is not submission-ready until the open/partial reviewer concerns and author-check labels are resolved. The provided response explicitly distinguishes these states.

## Verification on September 3, 2026

- Tectonic 0.17.0 successfully built the manuscript (64 pages), response (10 pages), and cover (one page). A separate source-only copy also built successfully without any old auxiliary files. The pdfLaTeX path is provided but was not tested because pdfLaTeX is not installed on this machine.
- The manuscript's 126 labels and all citations resolve. All 22 response blocks (two editor requests and 20 reviewer points) have valid manuscript references. The preservation checks passed; see `integration_audit.json`.
- All document pages were rendered for layout review. The two coordinate-wise bar plots, retained tables, appendix transitions, response headings, and cover page were checked visually.
- The manuscript retains two overfull proof displays in the protected supplement (lines 349 and 410). They remain legible. Existing class/package and underfull-box warnings are non-blocking. Tectonic also attempts an unnecessary separate BibTeX pass on `supplementary.aux`; its missing-bibliography warnings do not affect the successful main bibliography build.
- The local compiler used for verification is `E:\multi-target-scaling\tmp\tectonic-0.17.0\tectonic.exe`. It is outside the manuscript folder and is not required when using an existing TeX installation.
