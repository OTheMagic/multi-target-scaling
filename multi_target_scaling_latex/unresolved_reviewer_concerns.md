# Final Reviewer Audit and Author Decisions

Reviewed against the author's final LaTeX sources, arXiv:2512.15383v1, both reviewer reports, the editor's decision, and the saved experiment summaries. The latest pre-copyedit manuscript is preserved in `../reviewer_update/pre_final_editorial`.

The language and notation pass is complete, but this is **not a certification of scientific submission readiness**. The items below are author decisions. They do not all require new experiments: several can be resolved by precise definitions, a short proof argument, or narrower claims.

Stable tags appear as nonprinting `% AUTHOR-CHECK TAG:` comments in the manuscript sources and as visible author notes in the response and cover. They deliberately do not appear in the scientific narrative. `final_editorial_audit.json` records their exact current file and line locations.

## Reviewer Status

| Point | Status | Current evidence or remaining decision |
|---|---|---|
| R1.C1: methodological intuition and enclosure | Partial | Revised oracle/link/local narrative and retained enclosure comparison; EVENT, ENDPOINT, BOUNDARY, and related proof checks remain. |
| R1.C2: assumptions and negative residuals | Partial | Nonnegativity, moments, distinctness, heavy-tail limitations, and capped CQR are explicit; TIES and ZERO remain. |
| R1.C3: main-text wall-clock evidence | Addressed | Dependent-Gaussian runtime panel and six-dataset runtime figure. |
| R1.C4: contamination/outliers | Addressed | Exchangeable contamination sweep, with severe volume inflation retained. |
| R1.Q1: infinite variance versus theorem scope | Addressed | Original heavy-tail studies explicitly lie outside the finite-variance assumption for degrees of freedom 1.5 and 2. |
| R1.Q2: backward-search frequency/cost | Addressed | Actual split/coordinate counts and serial construction timings in the appendix; larger miscoverage levels exercise the exceptional branches. |
| R1.Q3: quantile-residual gains | Addressed | Native CQHR comparison, coordinate lengths, and base-interval/shift sensitivities. |
| R2.M1: Point CHR versus CQHR in related work | Open | LIT: Section 1.2 still generalizes the extra-split requirement to the entire Sampson and Chan reference. |
| R2.M2: coordinate comparisons | Addressed | Synthetic length bars and all 51 real-data coordinates, paired length ratios, and marginal coverage. |
| R2.M3: frequency of regularity-condition failure | Addressed | Actual fallback counts, including 0/200 per dataset at miscoverage 0.1, not inferred from equality of returned rectangles. |
| R2.M4: CQHR comparison | Addressed | Native CQHR, common fitted quantile models, explicit reference coordinate and width floor. |
| R2.M5: Tumu comparison and related work | Partial | The two-dimensional shape-template experiment is included; LIT remains for Section 1.2. |
| R2.M6: partial heteroskedasticity | Addressed | Five of ten coordinates depend on the input, with marginal second moments preserved. |
| R2.M7: CQR transformation and constant shift | Partial | Capped and shifted studies are included; TIES and SHIFT remain. |
| R2.m1: marginal labels in Figure 1 | Addressed | Distinct illustrative 92%/93% marginal labels and 90% joint coverage. |
| R2.m2: meaning of uniformly tight | Partial | The wording now identifies coverage balance, but BALANCE remains substantively unsupported by equal moments alone. |
| R2.m3: boundary labels in Figure 2 | Addressed | Residual boundaries are labeled without changing the drawing geometry. |
| R2.m4: runtime plots in main text | Addressed | Both synthetic and real-data wall-clock comparisons are in the body. |
| R2.m5: infinite Point CHR volume | Addressed | The final calibration allocation, rather than an intrinsic method failure, is explained. |
| R2.m6: vector/scalar typography | Addressed | The final pass standardizes multi-indices, scalar population coordinates, and algorithmic mean/scale vectors throughout the main text and supplement. |

**Total: 14 addressed, 5 partially addressed, 1 open.** These statuses describe the requested response evidence, not blanket validation of the method. The editor's completeness and color-marking requests remain conditional on author decisions.

## Scientific Checks

### AUTHOR-CHECK LIT: related-work attribution and scope

Location: Section 1.2; reviewer points R2.M1 and R2.M5.

The extra residual-scale split belongs to Point CHR, not native CQHR. Restrict that criticism accordingly; CQHR still uses ordinary model-training and calibration separation. Discuss the rectangular shape-template family of Tumu et al., which is now compared experimentally, and qualify the claim that competing learned constructions are necessarily nonrectangular. Also check the sentence asserting an empirical comparison with Baheri and Amiri Shahbazi: the displayed primary comparisons are Unscaled Max, Point CHR, Emp. Copula, and CQHR, not a separately identified implementation of that method.

### AUTHOR-CHECK DATA: synthetic sampling provenance

Location: Section 4.2; details in `sampling_provenance.md`.

The manuscript specifies fresh training/calibration/test observations per synthetic repetition, with 7,200/800 training/test observations for the absolute-residual comparisons. The saved `exps.ipynb` instead generates 8,000 training/test observations outside the repetition loop and repeatedly splits them 80%/20%, giving 6,400/1,600. Calibration is regenerated inside the loop.

The archived CSVs have no run-level sampling metadata. This does not prove that this exact notebook generated every CSV, but prevents verifying the declared design. The newer summaries explicitly record redraws and sample counts. Provide the correct generating records, corrected summaries, or authorize replacement runs. **No simulations, numerical CSVs, or figure PDFs were changed during this copyedit.**

### AUTHOR-CHECK MAP: the scalarization is not an invertible map

Locations: Sections 2.4--2.5, around `eq:joint-prediction-scalar` and `eq:residual-transformation`.

For d > 1, taking the maximum of standardized coordinates loses information: with zero location and unit scales, (1,0) and (0,1) both map to 1. The displayed threshold formula is therefore not a literal inverse. Describe a coordinate-wise characterization of sublevel sets, and explicitly require positive scale parameters. This is a terminology/definition correction, not a reason to abandon the construction.

### AUTHOR-CHECK EVENT: final piecewise coverage proof

Location: Theorem `thm:TSCP-coverage`.

Two procedures each having marginal coverage does not alone establish coverage after a data-dependent choice between them. The preceding local-union proof already gives the intended route: use a single oracle-acceptance event, and show that on this event the final set contains the test outcome in either branch. State that common event inclusion explicitly. The earlier erroneous cell-conditional inference was corrected by the author and is no longer an open item.

### AUTHOR-CHECK ENDPOINT: search-lemma contradiction

Location: Lemma `lem:reduction-search`.

The sentence says that all indices greater than or equal to the maximizing index have zero boundary value. This includes the maximizer itself and therefore forces the maximum to be zero. Check whether the intended relation is strictly greater, specify a rightmost maximizer if necessary, and reconcile ties with the proof and search algorithm. The inequality was not silently changed during a language edit.

### AUTHOR-CHECK BOUNDARY: partition and mean-index conventions

Locations: `eq:rectangle-explicit`, `eq:rect-wise-bounds`, and the mean-index definition.

The partition cells are half-open at their upper ends, but the global rectangle is closed, so literal equality of their union with the global rectangle omits its upper boundary. The subsequent local characterization uses closed upper bounds. Also, two weak inequalities around the empirical mean do not define a unique index when the mean equals an order statistic. Pairwise distinct observations alone do not rule out this equality. Choose consistent half-open/closed conventions or justify ignoring the relevant events under explicit assumptions.

### AUTHOR-CHECK ZERO: pairwise distinctness does not imply no atom at zero

Location: end of the link-function proof in Appendix B.

The proof replaces an empty solution set by the singleton residual zero and calls that event probability zero under Assumption 1. Under exchangeability alone, the implication is false. A uniformly random permutation of residuals (0,1,2) is exchangeable, nonnegative, pairwise distinct, and has positive finite marginal variance, but the test residual equals zero with probability 1/3.

Thus the stated assumptions do not justify the asserted equivalence on that event. An appropriate additional condition or a one-sided containment argument may suffice; verify it. This is separate from the practical tied-score issue.

### AUTHOR-CHECK DOMAIN: sample size and degenerate formulas

Locations: Section 2.2, Lemmas `lem:solution-key-ineq`, `lem:finite-prediction-guarantee`, `lem:gwc-rescaling`, and Lemma S3.

The original setup explicitly required n > 1; the revised setup no longer states it. For n = 1 the observed variance is zero and strict standardized-score bounds do not follow from distinctness. Restore the intended n >= 2 domain where needed. Also define the GWC candidate at test coordinate t_j equal to the observed mean; the displayed formula divides by zero there, although the proof separately treats that case.

### AUTHOR-CHECK ALGEBRA: linear coefficient in the link-function proof

Location: Appendix B, immediately after the quadratic is expanded as A x^2 + B x + K.

Expanding A(x-mu)^2 gives B = -2 A mu. The displayed B instead equals +2 A mu. The subsequent root formula is centered at +mu, so check and reconcile the expansion. The degenerate A = 0 branch also deserves checking: if the variance term is zero, the polynomial can vanish identically rather than only at x = mu. The sign was tagged for mathematical confirmation rather than disguised as a purely grammatical change.

### AUTHOR-CHECK DS: data-splitting algorithm's quantile index

Location: Appendix A, Algorithm `alg:std-ds`.

The algorithm defines scores only for indices in I_2, then takes a quantile of scores indexed 1,...,n. Use the final calibration subset and its size consistently. The empirical experiments were not rerun or altered.

### AUTHOR-CHECK TIES: capped-score implementation versus assumptions

Locations: Sections 2.3 and 4.3; standing Assumption 1.

Capping a signed score at zero guarantees nonnegativity, not pairwise distinctness or positive empirical scale. The saved TSCP implementation perturbs scalar upper-bound scores; this is not the coordinate jitter described in the theoretical discussion. Reconcile the convention and degenerate scales with the theorem or keep an explicitly empirical scope. The final copyedit restores the short capped-score qualification in the experiment section.

### AUTHOR-CHECK BALANCE: coverage balance and asymptotic shape claims

Locations: Sections 2.4--2.5 and 3.4.

Matching marginal means and variances does not equalize standardized tail shapes or marginal coverages. A defensible narrower description is coordinate-adaptive width that reduces inefficiency caused by heterogeneous scales. A coverage-balance claim needs additional conditions, and a random data-dependent cutoff needs more care than a fixed cutoff. The statement that the local union becomes rectangular in the large-sample limit also needs stated conditions/proof or an explicitly empirical qualification.

### AUTHOR-CHECK SHIFT: explanation and practical shift selection

Location: Appendix C, shifted CQR scores; response R2.M7.

The saved results support equality between shifted TSCP and shifted GWC within each tested run and near-invariance across the three constants, not a universal shift-invariance theorem or direct observation of the proposed bound-inflation mechanism. The copyedit identifies the mechanism as a proposed explanation.

A shift must ensure nonnegativity for future inputs, not just calibration residuals. For ordered quantile endpoints with width w_j(x), the signed score is bounded below by -w_j(x)/2. A verified uniform bound w_j(x) <= M gives the sufficient choice C >= M/2. The response explains this possibility; the current manuscript has no validated general shift selector. Decide whether to add a justified bound or explicitly retain the study as a fixed-constant sensitivity analysis.

## Submission Checks

### AUTHOR-CHECK REPRO: timing provenance and reproduction details

The diagnostic run manifest records the machine, package versions, serial timing convention, and thread limits. This does not establish the environment of the older six-dataset timing records used for the main comparison. Confirm the relevant run environment and that the public repository includes the files needed to reproduce the final selection. No network publication or repository release was performed here.

### AUTHOR-CHECK COLOR: revision highlighting

The experiments and many author revisions are blue; some non-experimental content is brown and the final grammar/notation corrections also affect previously unmarked passages. Confirm the editor-facing highlighting convention and apply it consistently. An exact source-level diff against v1 is unavailable because only its PDF was supplied.

### AUTHOR-CHECK METADATA: journal template fields

`main.tex` still contains the editor placeholder and provisional volume/year/date/paper-ID fields. The preprint style may suppress some of them. Confirm the appropriate submission values or leave only intentional template metadata; no factual publication information was invented.

## Changes and Verification

- Full language review includes the abstract, introduction, setup, methodology, discussion, algorithms, proofs, both experiment files, and active table captions.
- Fixed spelling, grammar, duplicated words, informal revision-style phrasing, coverage/volume descriptions, CQR outcome-space units, the extra parenthesis in the timing formula, and a figure-versus-appendix reference.
- Corrected the real-data caption's flag description: the red entries indicate undercoverage by more than two Monte Carlo standard errors, not any departure on either side of 0.9. Values and coloring are unchanged.
- Standardized vector multi-indices and parameter vectors while keeping scalar coordinates unbolded; R2.m6 is now addressed.
- Preserved the author's current order, appendix placement, omitted studies, 25 experimental PDF inclusions, all numerical tables, CSVs, and figure PDFs. Only the empirical-copula capitalization changes in the energy table source.
- Reconciled the cover to 12 added studies and 17 added figure assets; original studies and extra panels are not counted again. See `experiment_inventory.json`.
- The earlier dedicated small-calibration stress study, additional heavy-tail stress study, uncertainty section, compact duplicate real-data table, and diagnostic joint-coverage table are no longer claimed to appear in the current manuscript. Their saved files are not deleted.
- The final audit is `../reviewer_update/check_final_editorial.py`. Older integration/publication audits refer to earlier, narrower editing stages and are not current submission certificates.
