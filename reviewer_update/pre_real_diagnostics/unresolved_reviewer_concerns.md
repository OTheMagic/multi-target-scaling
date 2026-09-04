# Reviewer Audit and Author Actions

This audit compares the supplied revised LaTeX sources, the integrated experiments, the original arXiv v1 PDF, and the two reviewer reports. It is not a certification of every proof. No non-experimental manuscript section or schematic figure was edited during integration.

## Status Overview

| Point | Status | Evidence or remaining action |
|---|---|---|
| R1.C1: methodological intuition/enclosure | Partial | Revised Sections 2--3 and retained local-union comparison; correct the conditional-coverage inference below. |
| R1.C2: assumptions and negative residuals | Partial | Explicit heavy-tail caveats and CQR studies; reconcile ties, zero scales, and implementation with Assumption 1. |
| R1.C3: main-text runtime evidence | Addressed | New synthetic runtime panel and six-dataset wall-clock figure. |
| R1.C4: contamination/outliers | Addressed | Exchangeable contamination stress test; efficiency inflation is disclosed. |
| R1.Q1: infinite variance versus validity theorem | Addressed | Both new and retained heavy-tail descriptions explicitly limit the theoretical claim. |
| R1.Q2: backward-search frequency/cost | Open | No saved branch indicators or conditional runtimes. |
| R1.Q3: quantile-residual gains | Addressed | Main CQR comparison, coordinate bars, and sensitivity studies. |
| R2.M1: Point CHR versus CQHR in related work | Open | The overgeneralization remains in Section 1.2. |
| R2.M2: coordinate comparisons | Partial | Two new synthetic bar plots; the real-data illustration is anecdotal, not an aggregate analysis. |
| R2.M3: regularity-condition failure frequency | Open | No saved fallback indicators. |
| R2.M4: CQHR comparison | Addressed | Native CQHR using common fitted quantile models and disclosed width-ratio choices. |
| R2.M5: Tumu baseline and related work | Partial | Low-dimensional comparison and bibliography entry added; Section 1.2 still needs discussion. |
| R2.M6: partial heteroskedasticity | Addressed | Main-text experiment with five of ten affected coordinates. |
| R2.M7: CQR transformation and constant shift | Partial | Capped and shifted studies and a sufficient shift bound are explained; theoretical/implementation qualifications remain. |
| R2.m1: marginal labels in Figure 1 | Open | Original schematic still shows only joint coverage in its right panel. |
| R2.m2: meaning of uniformly tight | Open | Ambiguous claim remains in Sections 2.4 and 2.5. |
| R2.m3: boundary labels in Figure 2 | Open | Original partition schematic lacks the final residual-boundary labels. |
| R2.m4: main-text runtime plots | Addressed | Main synthetic and real-data runtime plots; old runtime plots also retained. |
| R2.m5: infinite Point CHR volume | Addressed | Allocation explanation and finite-sample quantile caveat now accompany the comparison. |
| R2.m6: vector/scalar typography | Partial | Many vectors are now bold; multi-indices and some scalar coordinates remain inconsistent. |

Total: **8 addressed, 6 partially addressed, 6 open**. The editor's overall completion and manuscript-wide color-marking requests remain partial until the open items are resolved and the final submission is audited.

## Highest-Priority Scientific Checks

### 1. Conditional coverage inferred from a marginal guarantee

- Location: `body.tex:470`, Section 3.4.1, immediately after the local-containment display.
- The revised paragraph claims coverage at least `1-alpha` conditional on the working hypothesis that the outcome belongs to a region. Marginal oracle coverage and containment on that event do not, by themselves, imply that conditional guarantee.
- Suggested direction: express the argument as a pointwise implication of the oracle acceptance event on the cell containing the test outcome, then aggregate over the partition to obtain the intended marginal result. Do not infer cell-conditional or covariate-conditional coverage without additional assumptions.
- Related response: R1.C1. This statement was not present in this form in the original v1 paragraph and needs review before the revision is submitted.

### 2. Contradictory endpoint condition in the search-reduction lemma

- Location: `body.tex:645`, Section 3.4.4, Lemma `lem:reduction-search`.
- The statement first says that the maximizing value `B_j` is positive, then says it is zero for every index greater than **or equal to** the maximizer, including that same maximizer.
- Suggested direction: verify whether the intended relation is strictly greater, and check ties, the choice of maximizer, and the matching proof before changing it. This appears to be inherited from the old draft, not introduced by the experiment integration.

### 3. Piecewise coverage proof needs a shared event argument

- Location: `body.tex:669`, proof of Theorem `thm:TSCP-coverage`.
- Selecting between two data-dependent procedures does not inherit their marginal coverage merely because each procedure separately has a lower bound. The intended construction may admit a common oracle-acceptance/containment argument, but that argument should be explicit for both branches.
- Suggested direction: verify and state the common pointwise event inclusion. This is a proof-exposition issue to check, not a claim here that the implemented procedure is invalid.

## Empirical Work Still Needed

### 4. Backward-search and fallback audit

- Related points: R1.Q2 and R2.M3; methodology at `body.tex:652` and `body.tex:658`.
- Current result files have aggregate runtime, coverage, and volume but no branch indicators. Equality between TSCP and GWC output does not identify the executed branch.
- Replay the six datasets' original 200 splits when possible, or identify the results as a new diagnostic study. For each dataset report fallback count/rate, count/rate of runs with any backward search, the fraction of coordinate searches using that branch among non-fallback runs, and conditional runtimes.
- Define denominators explicitly. Count a fallback once per calibration split, not once per test observation. A branch observed zero times has an unavailable conditional runtime, not a runtime of zero.
- Replace the qualitative assertion that the condition typically holds with evidence or a qualification. Do not insert zeros into unmeasured cells.

### 5. Real-data coordinate comparisons

- Related point: R2.M2. Main-text `tab:toy` is the original pair of energy examples.
- New absolute and CQR coordinate bars answer the synthetic component. Two example test points do not answer the request for real-data coordinate summaries.
- Add mean full lengths or marginal coverages over the same real-data splits, with units and uncertainty; alternatively state why this extension is omitted. The existing aggregate tables cannot reconstruct these quantities.

## Non-Experimental Edits for the Author

### 6. Correct and expand Section 1.2

- Related points: R2.M1 and R2.M5; `body.tex:44` and `body.tex:46`.
- Restrict the additional-split criticism of Sampson and Chan to Point CHR. Explain that native CQHR has the ordinary training/calibration split but no auxiliary residual-scale split.
- Discuss Tumu et al. and rectangular shape templates. The experiment and its new bibliography entry do not constitute a revision of the literature review.
- Check the attribution of a demonstrated comparison to Baheri and Amiri Shahbazi against the actual evaluated methods. Qualify broad claims that all alternative learned geometries must be nonrectangular or have the same data/computation requirements.

### 7. Reconcile capped CQR, ties, and fixed shifts

- Related points: R1.C2 and R2.M7; `body.tex:149` through the standing assumption at `body.tex:164`.
- The supplied revision recognizes ties and suggests coordinate jitter. The saved TSCP code adds a reproducible perturbation to scalar upper-bound scores, which is not the same operation and does not remove coordinate zeros or handle all zero empirical scales.
- The integrated experiments explicitly say that the capped study does not meet pairwise distinctness verbatim. Either justify the exact implemented convention, modify it with a corresponding analysis and rerun if needed, or retain the empirical qualification.
- The main score is `max(S_j,0)`, not `abs(S_j)`. Its region expands the fitted interval and cannot shrink an already conservative interval.
- The appendix uses fixed shifts 100, 300, and 1000, with inversion by subtracting the shift. Its near-invariance is an observed property of this design, not a general theorem for the local approximation.
- A uniform fitted-width bound `w_j(x) <= M` gives the sufficient shift `C >= M/2`. An observed calibration minimum alone does not guarantee nonnegativity for future inputs. Clarify the sign/magnitude of the shift and avoid an unqualified invariance assertion in Section 2.3.

### 8. Define or weaken the tightness/balance claim

- Related point: R2.m2; `body.tex:227` and `body.tex:271`.
- Equal first and second moments do not imply equal standardized tail shapes or equal marginal coverage. Define whether tightness means width, volume, oracle approximation, or coverage.
- Suggested wording: coordinate-adaptive intervals that reduce inefficiency due to heterogeneous residual location/scale. Stronger balance or asymptotic claims require their own conditions and proof.

### 9. Update the schematic figures

- Related point R2.m1: `figures/joint_prediction.pdf` and `figures/joint_prediction_draw.tex`; the right panel lacks both marginal coverage labels. Use measured values or clearly marked schematic percentages, not invented empirical numbers.
- Related point R2.m3: `figures/partition.pdf` and `figures/partition_draw.tex`; add the final residual bounds using the current notation `mathcal L_1, mathcal L_2`, and identify residual-space axes.
- These are methodological figures and were left byte-for-byte unchanged as requested.

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
- Sections 1--3, Discussion/Software/Acknowledgments, and the algorithm/proof portions of the supplement are unchanged as source text.
- All 95 original active labels, 10 original graphic inclusions, and seven original table labels are retained. All 26 original figure-directory files are unchanged, including unused assets.
- All 15 new PDFs are copied without alteration. New supporting CSV files are in `experiment_data`.
- `integration_audit.json` records the structural checks. `reviewer_update/pre_integration_latex` outside this folder preserves the complete supplied folder for comparison.
- A separate compilation wrapper handles Unicode punctuation and appendix hyperlink destinations under Tectonic; it changes neither the manuscript's scientific text nor printed numbering.
