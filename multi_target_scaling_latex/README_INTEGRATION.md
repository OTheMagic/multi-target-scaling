# Integrated Manuscript

## Current Deliverables

- `main.tex` and `main.pdf`: manuscript, including the full supplement.
- `body.tex` and `supplementary.tex`: theory and algorithm sources, now copyedited for language and vector/scalar notation.
- `experiments_body.tex` and `experiments_appendix.tex`: the author's final experiment selection and order, copyedited without changing numerical results.
- `response_to_reviewers.tex` and `.pdf`: 22 response blocks, including two editor requests, with current manuscript references and explicit completion statuses.
- `revision_cover.tex` and `.pdf`: section-by-section comparison with arXiv:2512.15383v1. It first states the total of 12 added studies, then explains each study's purpose and finding.
- `unresolved_reviewer_concerns.md`: the author decision queue, organized under 16 stable `AUTHOR-CHECK` tags.
- `PUBLICATION_READINESS.md`: editorial assessment and remaining submission decisions.
- `experiment_inventory.json`: the counting rules, data sources, and figure assignments for the added studies.
- `final_editorial_audit.json`: current preservation, reference, asset, and document checks.

The `figures` folder supplies all 25 active experimental PDFs and two methodological schematics. The experimental assets comprise 17 added PDFs and eight restyled original PDFs. The original assets and unused saved results remain available; compilation does not require them to be removed. `experiment_data` supplies the saved summaries and diagnostic records. All compilation inputs are local to this manuscript folder.

## Build

With TeX Live or MiKTeX and PowerShell:

```powershell
.\build.ps1 -Engine pdflatex
```

With a portable Tectonic executable:

```powershell
.\build.ps1 -Engine tectonic -TectonicPath 'C:\path\to\tectonic.exe'
```

The script builds the manuscript before the response and cover because both import manuscript labels. The response and cover are standalone documents, not pages inserted into `main.pdf`. The manuscript uses `jmlr2e.sty`, standard LaTeX packages, and `biblio.bib`; a TeX compiler is not bundled.

For Tectonic, `compile_main.tex` supplies compatibility mappings and unique appendix hyperlink destinations. The build script copies its outputs to `main.*` before building the letters. A first Tectonic run may need to download standard TeX packages; it does not upload the manuscript.

## Final Copyedit and Preservation

The full author-edited source folder was preserved before this pass at `../reviewer_update/pre_final_editorial`. This is the baseline for the current audit. The copyedit covers the abstract, introduction, setup, methodology, discussion, algorithms, proofs, both experiment files, and active table captions. Numerical tables, CSVs, and all figure PDFs are unchanged. The only table-source wording change is capitalization of `Emp. Copula` in the energy illustration.

The author's current omissions and appendix placement are preserved. In particular, the earlier standalone small-calibration stress study, additional heavy-tail stress study, uncertainty section, compact duplicate real-data table, and diagnostic joint-coverage table were not restored. Their saved assets were not deleted, and the response and cover no longer claim that these items appear in the current draft. All 127 existing author labels are retained; two section labels were added for precise references.

The manuscript presents one synthetic sampling design. The provenance of the archived Gaussian/heavy-tail/oracle CSVs still needs reconciliation with that declaration: the saved notebook generates its training/test pool outside the repetition loop. This is an open author decision, not a newly endorsed second protocol. See `sampling_provenance.md` and `AUTHOR-CHECK DATA`.

The real-data diagnostic cohort uses 200 reruns on each of six fixed datasets and preserves the original 75%/5%/20% split proportions. These are new split/model reruns, not a reconstruction of the older fitted models or runtime environment. Its uncertainty descriptions concern random splits conditional on each fixed dataset. Reproduction details are in `../reviewer_update/real_diagnostics/README.md`; cached datasets and residuals are not compilation dependencies.

The 12-study count consists of ten simulation study questions and two real-data diagnostic study questions. It is not a count of independent datasets or independently generated cohorts: the dependence overview and dependence sweep share an experiment family, and the two real-data diagnostics share their rerun cohort. Coordinate panels are not counted separately. The runtime plot of existing real-data records is an additional visualization, not an added experiment.

## Reviewer and Author Decisions

The response records **14 addressed, five partially addressed, and one open reviewer point**. These statuses describe the requested evidence, not blanket validation of the theory. The editor's completeness and revision-color requests remain conditional on the author decisions.

Stable `% AUTHOR-CHECK TAG:` comments identify substantive questions in the LaTeX sources without printing them in the scientific narrative. The response and cover display the relevant author notes. `unresolved_reviewer_concerns.md` explains each issue and possible resolution; the audit lists its current source line. Scientific concerns were not silently treated as grammar corrections.

The language and presentation have been polished, but the folder should not be treated as unconditionally submission-ready while those checks remain. In particular, verify the scalarization terminology, piecewise coverage argument, search endpoint, zero-residual argument, related-work scope, and historical sampling provenance. The existing blue/brown revision colors and journal template metadata also need an author decision.

## Verification

Run the current validator from the project root:

```powershell
python reviewer_update/check_final_editorial.py
```

It checks the final copyedit against the author's latest snapshot, validates reference and citation targets, compares all figure/data assets and numerical tables, reconciles the study inventory and reviewer statuses, and checks the three compiled PDFs for unresolved references. It writes `final_editorial_audit.json`, including current page counts and PDF hashes.

`reviewer_update/render_integration_qa.py` renders every page for visual inspection. Rendered pages and contact sheets are intermediates under `../tmp/pdfs/integration_qa`. The final pass also reflows the two previously overflowing proof displays. Remaining underfull-box and existing package warnings are nonblocking. The pdfLaTeX build path is supplied but was not tested because pdfLaTeX is not installed on this machine.

On September 4, 2026, normal and fresh source-only Tectonic 0.17.0 builds passed. The current outputs are a 64-page manuscript, a 13-page response, and a two-page cover. The source audit resolves 129 labels with no missing reference or citation targets, and all document pages were rendered and checked for layout. The compiler is at `E:\multi-target-scaling\tmp\tectonic-0.17.0\tectonic.exe`, outside the manuscript folder.

The earlier `integration_audit.json`, `publication_edit_audit.json`, and their validators document previous stages with different preservation boundaries and experiment selections. They are historical records, not the current audit. Figure restyling and numerical diagnostic tests from earlier stages remain relevant to the unchanged assets; no numerical experiments were rerun in this final copyedit.
