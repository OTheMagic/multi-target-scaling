# Reviewer Audit and Author Actions

This audit compares the revised LaTeX sources, the integrated experiments, the original arXiv v1 PDF, and the two reviewer reports. It is not a certification of every proof. The author's latest non-experimental edits have been preserved and the affected audit entries updated on September 4, 2026. No non-experimental prose was rewritten during the experiment editing. At the author's request, Figures 1 and 2 have been annotated to address R2.m1 and R2.m3; their previous PDFs and drawing sources are preserved in `../reviewer_update/pre_schematic_labels`.

The earlier publication-style editorial pass consolidated every experimental study into `experiments_body.tex` and `experiments_appendix.tex`, organized by scientific topic, without changing experimental data, settings, or figure/table assets. Revision history is confined to the cover, response, and project documentation. The latest preservation checks are recorded in `publication_edit_audit.json`.

The subsequent figure-standardization pass preserves the author's ordering through Section 4.2, removes the two-protocol presentation, and redraws eight archived figures plus four real-data diagnostic figures in the common style. Numerical summaries and tables are unchanged. The sampling provenance of the archived summaries is **not yet verified against the author's stated independent-redraw design**; see item 0 below and `sampling_provenance.md`. This new source-verification item is separate from the twenty reviewer points.

The follow-up diagnostic study has now completed 200 new splits of each of the six real datasets. It closes the three previously missing empirical items below; it does not resolve the separate theoretical and editorial issues.

## Status Overview

| Point | Status | Evidence or remaining action |
|---|---|---|
| R1.C1: methodological intuition/enclosure | Partial | Revised Sections 2--3, local-union comparison, and corrected local pointwise implication; finish the common-event justification for the final piecewise proof. |
| R1.C2: assumptions and negative residuals | Partial | Explicit heavy-tail caveats and CQR studies; reconcile ties, zero scales, and implementation with Assumption 1. |
| R1.C3: main-text runtime evidence | Addressed | New synthetic runtime panel and six-dataset wall-clock figure. |
| R1.C4: contamination/outliers | Addressed | Exchangeable contamination stress test; efficiency inflation is disclosed. |
| R1.Q1: infinite variance versus validity theorem | Addressed | Both new and retained heavy-tail descriptions explicitly limit the theoretical claim. |
| R1.Q2: backward-search frequency/cost | Addressed | Actual branch counts, candidate counts, and serial timings on 1,200 real-data splits; additional targets exercise the backward branch. |
| R1.Q3: quantile-residual gains | Addressed | Main CQR comparison, coordinate bars, and sensitivity studies. |
| R2.M1: Point CHR versus CQHR in related work | Open | The overgeneralization remains in Section 1.2. |
| R2.M2: coordinate comparisons | Addressed | Existing synthetic bars plus all 51 real-data coordinates, paired length ratios, raw lengths, and marginal coverage over 200 splits per dataset. |
| R2.M3: regularity-condition failure frequency | Addressed | Actual fallback indicators: 0/200 per dataset at the main target, with finite-sample qualifications and a target-level stress sweep. |
| R2.M4: CQHR comparison | Addressed | Native CQHR using common fitted quantile models and disclosed width-ratio choices. |
| R2.M5: Tumu baseline and related work | Partial | Low-dimensional comparison and bibliography entry added; Section 1.2 still needs discussion. |
| R2.M6: partial heteroskedasticity | Addressed | Main-text experiment with five of ten affected coordinates. |
| R2.M7: CQR transformation and constant shift | Partial | Capped and shifted studies and a sufficient shift bound are explained; theoretical/implementation qualifications remain. |
| R2.m1: marginal labels in Figure 1 | Addressed | Right panel now shows 92% for Y1 and 93% for Y2, alongside 90% joint coverage; percentages are explicitly illustrative. |
| R2.m2: meaning of uniformly tight | Partial | Replaced by balanced coordinatewise coverage in Sections 2.4 and 2.5; equal moments alone still do not justify coverage balance. |
| R2.m3: boundary labels in Figure 2 | Addressed | Blue axis ticks and labels identify the solid enclosure's residual thresholds, mathcal L_1 and mathcal L_2. |
| R2.m4: main-text runtime plots | Addressed | Main synthetic and real-data runtime plots; old runtime plots also retained. |
| R2.m5: infinite Point CHR volume | Addressed | Allocation explanation and finite-sample quantile caveat now accompany the comparison. |
| R2.m6: vector/scalar typography | Partial | Many vectors are now bold; multi-indices and some scalar coordinates remain inconsistent. |

Total: **13 addressed, 6 partially addressed, 1 open**. The editor's overall completion and manuscript-wide color-marking requests remain partial until the open items are resolved and the final submission is audited.

## Highest-Priority Scientific Checks

### 0. Archived sampling provenance: open

- The author specifies independently generated training/calibration/test observations in every synthetic repetition, with 7,200/800 training/test observations for absolute residuals.
- The saved `exps.ipynb` generator in code cell 11 instead generates its 8,000-observation training/test data outside the repetition loop and splits it 80%/20% inside that loop. Calibration is redrawn. The archived CSVs lack run-level sampling metadata, so their exact generation history needs reconciliation rather than assumption.
- The restyled figures preserve the archived numerical values. Restyling does not validate the sampling design or convert those summaries into a new experiment.
- Required: generating-code/run provenance supporting the stated settings, corrected summaries, or authorization to rerun the affected studies and update matching tables. Details and all affected figure families are listed in `sampling_provenance.md`.
- The paper now uses the single stated design rather than describing two alternatives. The discrepancy is recorded here and on the cover/response, not embedded as revision commentary in the experiment narrative. Do not treat the draft as submission-ready until it is resolved.

### 1. Local conditional-coverage inference: corrected by the author

- Location: Section 3.4.1, immediately after `eq:lwc-local-containment`.
- The latest author edit replaces the conditional `1-alpha` claim with a pointwise implication when oracle acceptance and membership in the working region hold simultaneously. The previous objection to this sentence is therefore closed.
- Related response: R1.C1 remains partial because the final piecewise proof still needs a shared-event justification; see item 3. No cell-conditional or covariate-conditional guarantee is inferred.

### 2. Contradictory endpoint condition in the search-reduction lemma

- Location: Section 3.4.4, Lemma `lem:reduction-search`, sentence immediately after the maximization display.
- The latest author edit removes the explicit `> 0` after the maximizing value. The subsequent sentence still says `B_j` is zero at every index greater than **or equal to** the maximizer, including the maximizer itself. This would force the maximum to be zero, so the endpoint relation still needs verification for nonzero enclosures.
- Suggested direction: verify whether the intended relation is strictly greater, and check ties, the choice of maximizer, and the matching proof before changing it. This appears to be inherited from the old draft, not introduced by the experiment integration.

### 3. Piecewise coverage proof needs a shared event argument

- Location: proof of Theorem `thm:TSCP-coverage`.
- Selecting between two data-dependent procedures does not inherit their marginal coverage merely because each procedure separately has a lower bound. The intended construction may admit a common oracle-acceptance/containment argument, but that argument should be explicit for both branches.
- Suggested direction: verify and state the common pointwise event inclusion. This is a proof-exposition issue to check, not a claim here that the implemented procedure is invalid.

## Empirical Work Now Completed

### 4. Backward-search and fallback audit: addressed

- Related points: R1.Q2 and R2.M3; methodology at `body.tex:652` and `body.tex:658`.
- The new study uses the original hash-seeded splits, 75%/5%/20% allocation, and statistical model settings. It is labeled a new rerun, not a recovery of the original fitted models or historical timing environment.
- At miscoverage 0.1, both backward search and fallback occurred in **0/200 splits on every dataset**. All 10,200 coordinate searches used the binary branch. These are measured zeros, not missing values inferred from returned rectangles.
- The new body table `tab:real-search-diagnostics` reports split and coordinate denominators, with conditional runtimes. `fig:body-real-search-diagnostics` reports candidate counts and construction time. Means range from 0.466 to 7.239 ms under the new serial timing protocol; unobserved conditional runtimes are marked unavailable, not zero.
- For each dataset and either split event, the pointwise one-sided 95% zero-count upper bound is about 1.49%. It does not treat coordinates as independent or give simultaneous guarantees across datasets.
- The supplementary stress sweep (`app:real-search-stress`) observes both branches at other targets. For energy at miscoverage 0.5: 3 backward, 140 fallback, and 57 binary-only splits; conditional whole-call means are 0.295, 0.121, and 0.268 ms, respectively. All observed backward coordinates visited only one candidate, so these are not worst-case scans.
- The main-target result supports a qualified empirical statement that the condition held in the tested runs. It is not a theorem that the event is impossible, and does not repair the proof issues above.

### 5. Real-data coordinate comparisons: addressed

- Related point: R2.M2. Main-text `tab:toy` is the original pair of energy examples.
- The new body figure `fig:body-real-coordinate-bars` covers all six datasets and all 51 target coordinates using 200 paired reruns per dataset. It reports mean per-split ratios of full interval lengths to Unscaled Max; raw full lengths and their standard deviations are also saved.
- The appendix adds all coordinate-wise marginal coverages (`fig:app-real-marginal-coverage`) and matching joint coverages (`tab:real-diagnostic-coverage`). The supporting data contain 40,800 split--method--coordinate records, target names, standard deviations, and pointwise Monte Carlo summaries.
- The comparisons retain unfavorable results: TSCP is wider than Unscaled Max in all three student coordinates, and some energy/stock coordinates are wider despite volume gains. Infinite Point CHR stock intervals are explicitly labeled and retained in the summaries.
- All six new TSCP joint-coverage means have 0.9 within one across-split standard deviation. This is not a confidence interval for a mean or a proof of validity; uncertainty from repeated splits is conditional on the fixed dataset.

## Non-Experimental Edits for the Author

### 6. Correct and expand Section 1.2

- Related points: R2.M1 and R2.M5; `body.tex:44` and `body.tex:46`.
- Restrict the additional-split criticism of Sampson and Chan to Point CHR. Explain that native CQHR has the ordinary training/calibration split but no auxiliary residual-scale split.
- Discuss Tumu et al. and rectangular shape templates. The experiment and its new bibliography entry do not constitute a revision of the literature review.
- Check the attribution of a demonstrated comparison to Baheri and Amiri Shahbazi against the actual evaluated methods. Qualify broad claims that all alternative learned geometries must be nonrectangular or have the same data/computation requirements.

### 7. Reconcile capped CQR, ties, and fixed shifts

- Related points: R1.C2 and R2.M7; Section 2.3 and Assumption `eq:assumption-scores`.
- The supplied revision recognizes ties and suggests coordinate jitter. The saved TSCP code adds a reproducible perturbation to scalar upper-bound scores, which is not the same operation and does not remove coordinate zeros or handle all zero empirical scales.
- The integrated experiments explicitly say that the capped study does not meet pairwise distinctness verbatim. Either justify the exact implemented convention, modify it with a corresponding analysis and rerun if needed, or retain the empirical qualification.
- The main score is `max(S_j,0)`, not `abs(S_j)`. Its region expands the fitted interval and cannot shrink an already conservative interval.
- The appendix uses fixed shifts 100, 300, and 1000, with inversion by subtracting the shift. Its near-invariance is an observed property of this design, not a general theorem for the local approximation.
- A uniform fitted-width bound `w_j(x) <= M` gives the sufficient shift `C >= M/2`. An observed calibration minimum alone does not guarantee nonnegativity for future inputs. The latest author edit removes the unqualified shift-invariance assertion from Section 2.3; that wording objection is closed. The appendix retains the fixed-shift study with explicit qualifications.

### 8. Define or weaken the tightness/balance claim

- Related point: R2.m2; Sections 2.4 and 2.5.
- The latest author edit replaces "uniformly tight" with "balanced coordinatewise coverage", clarifying the intended quantity. Equal first and second moments, however, do not imply equal standardized tail shapes or equal marginal coverage. The substantive justification remains incomplete.
- Suggested wording: coordinate-adaptive intervals that reduce inefficiency due to heterogeneous residual location/scale. Stronger balance or asymptotic claims require their own conditions and proof.

### 9. Schematic figure labels: addressed

- R2.m1: `figures/joint_prediction.pdf` and its TikZ source now show distinct right-panel marginal coverages: 92% for Y1 and 93% for Y2, while retaining 90% joint coverage. A note explicitly identifies the percentages as illustrative, not experimental estimates.
- R2.m3: `figures/partition.pdf` and its TikZ source now label the solid rectangle's horizontal and vertical residual thresholds with `mathcal L_1` and `mathcal L_2`. The ticks are tied directly to the existing rectangle endpoints; the E1/E2 axes and partition geometry are unchanged in the drawing source.
- New annotations are blue. These two updates were explicitly authorized after the initial experiment-only integration; no other non-experimental manuscript content was changed.

### 10. Finish the notation pass

- Related point: R2.m6.
- `body.tex:264`: the coordinate quantities for population mean and standard deviation are bold even though they are scalars.
- Section 3.4.4 and Algorithm 2: the vector multi-index `h*` and some vector mean/scale symbols remain unbolded. Check the appendix algorithms and proofs as well.
- Preserve the useful distinction between outcome interval endpoints `L_j,U_j` and residual thresholds `mathcal L_j`.

### 11. Final editorial consistency

- Section 5 is unchanged. Consider cross-referencing the new contamination and CQR studies while distinguishing a fully signed method from the present capped/fixed-shift experiments.
- Check that all changed non-experimental passages are colored for the editor. A full source-level diff against v1 was not possible because v1 was provided as a PDF.
- Hardware provenance of historical real-data timings is not reconstructed by the current machine specification; add verified run information if available.
- Two pre-existing overfull displayed equations remain in `supplementary.tex:349` and `supplementary.tex:410`. They compile and remain legible, but should be rewrapped by the author if desired. No mathematical content was changed to suppress these warnings.

## Verified Preservation and Integration

- `main.tex`, `abstract.tex`, and `jmlr2e.sty` are unchanged from the supplied folder.
- The author's latest edits to Sections 2--3 are preserved in `../reviewer_update/pre_figure_unification/body.tex`. Discussion/Software/Acknowledgments and the algorithm/proof portions of the supplement remain unchanged from the supplied source. The validator distinguishes author edits from this editorial pass rather than reverting them to the initial snapshot.
- The missing `fig:joint-prediction` label was restored with the author's permission. This is the only body-source change made during the publication edit; the current text is otherwise identical to the latest author snapshot.
- All 95 original active labels, 10 original graphic inclusions, and seven original table labels are retained. Of the 26 original figure-directory files, 22 are unchanged; the two schematic PDFs and their two drawing sources are intentionally updated for R2.m1/R2.m3. Their pre-edit copies are preserved, and the source audit checks that only the annotation blocks were added.
- Of the 19 previously added experimental PDFs, 15 are unchanged and four real-data diagnostic figures have been restyled. Eight archived experimental plots are redrawn under unified filenames, giving 27 active experimental PDFs; their original PDFs remain available at the old paths. Supporting CSV files are unchanged in `experiment_data`, with the follow-up records in `experiment_data/real_diagnostics`.
- Passive branch instrumentation passed 63 regression tests. A full replay verified all 1,200 split-index partitions, coordinate metrics, branch counts, and exact equality with the pre-instrumentation TSCP rectangles. No zero empirical-scale coordinate or fitting warning occurred in these reruns; this does not resolve general tie/degeneracy assumptions.
- `integration_audit.json` records structural checks; `publication_edit_audit.json` now compares against the author's latest files in `../reviewer_update/pre_figure_unification`, checks all 59 pre-edit experimental labels, 27 graphic inclusions with the authorized filename mapping, nine table inputs, the inline numerical table, and preservation of 110 existing assets. Four real-data PDFs are restyled and eight renamed PDFs are added. `figure_style_audit.json` records plotted values and source hashes. `reviewer_update/pre_integration_latex` preserves the initial supplied folder for comparison.
- A separate compilation wrapper handles Unicode punctuation and appendix hyperlink destinations under Tectonic; it changes neither the manuscript's scientific text nor printed numbering.
- The figure-style unit checks verify exact correspondence with source summaries, runtime conversion to milliseconds, canonical method styling, the coverage display range, retained local-union dimensions, vector PDF fonts, and legend labels. These checks do not certify archived sampling provenance.
- Both preservation audits and all five figure-style tests pass. Normal and fresh source-only Tectonic builds produce the 70-page manuscript, 12-page response, and one-page cover with resolved references and citations. All 27 experimental figures and the compiled document pages were rendered for visual review; no experiment-section overflow warnings remain.
