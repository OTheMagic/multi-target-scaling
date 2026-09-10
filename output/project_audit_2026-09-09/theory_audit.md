# Manuscript and theory audit — September 9, 2026

Read-only assessment of the current TeX sources, supplementary proofs, response letter, source-side readiness notes, and current signed-envelope derivation. Line references below are to the current source, not the older audit's saved line numbers. This is a source-level mathematical assessment, not a complete independent formal proof verification or a rerun of every experiment.

## Main conclusion

There are two generations of the scientific contribution in this folder. The JMLR manuscript presents the original nonnegative-score TSCP/GWC/LWC shortcut. The newer standalone envelope report contains a materially stronger construction: direct signed scores, exact surface bounds, weaker distributional requirements, explicit boundary/degeneracy rules, a same-score containment theorem relative to the corrected old shortcut, and faster dimension dependence. The most consequential remaining task is to decide and integrate the final scientific story; it is more than figure polishing.

The September 4 editorial status files cannot be treated as current scientific certificates. Some mathematical issues they list have already been repaired in the current source; other manuscript claims, reviewer responses, and embedded figures have not caught up with the September 8–9 methodology and reruns.

## What is already developed

### Original manuscript

- Problem: simultaneous prediction intervals for all d outcomes, with marginal coverage over calibration and test randomness. Independent model training; arbitrary fixed predictive model and coordinate residual maps. The goal is rectangular interpretability, not unrestricted minimum-volume prediction sets (`multi_target_scaling_latex/body.tex:75`, `:95`, `:97`, `:122`).
- Motivation: heterogeneous coordinate error location/scale, avoiding extra calibration splitting while preserving exchangeability. The population oracle, naive plug-in, and split-scale constructions clarify the statistical tradeoff (`body.tex:229`, `:275`, `:284`).
- Main mechanism: augment the residual calibration sample with the unknown candidate, standardize symmetrically, derive an explicit self-score link, then construct computable upper bounds on the conformal quantile (`body.tex:298`, `:310`, `:352`).
- Three coverage theorems: fixed/conditionally exchangeable scalarization (`:207`); GWC coverage (`:434`); final TSCP coverage (`:669`). One main local-enclosure proposition (`:486`), five main lemmas (link, oracle finiteness, GWC formula, LWC formula, row/search reduction), and three auxiliary lemmas in the supplement. Nine pseudocode algorithms total across body and supplement.
- The expensive conceptual cell construction is reduced from (n+1)^d cells to coordinate rows, with backward or binary search (`body.tex:599`, `:639`, `:681`). The stated original best/worst costs are O(d² n log n) and O(d² n²); nonnegative-score boundaries are reusable across test inputs (`:711`, `:713`).
- Written proofs and cost analyses are substantial, not placeholders (`supplementary.tex:145`, `:186`, `:262`).
- Explicit current original-theory assumptions: n≥2, exchangeable observations, independently trained model, nonnegative atom-free coordinate residual distributions with positive finite population variance, and almost surely distinct coordinate observations (`body.tex:82`, `:122`, `:148`, `:163`). Infinite-variance and capped-score empirical experiments therefore require explicit separation from that theorem's scope.

### Signed surface-envelope extension

The mathematical specification is `envelope_method/signed_envelope.tex`; its first part is the September 9 research report and its appendix is the full derivation.

- General scores may be signed and may have ties; the attainable test-score domain can depend on x, provided it is a valid known enclosing domain (`:489`–`:517`). Interval sublevel sets are needed for rectangular output geometry, not for validity itself (`:505`).
- Exact augmented-standardization formula and signed inverse, including ±infinity branches and genuine empty sublevels (`:534`–`:605`). Finite population moments are explicitly unnecessary (`:542`–`:544`).
- A full-conformal reference set with a standard exchangeable-rank coverage proof (`:607`–`:629`).
- Exact coordinate supremum of the entire location/scale ratio, retaining admissible stationary maxima and correct endpoint limits (`:631`–`:658`). This tightens the old separation of numerator/denominator contributions.
- Signed GWC contains full CP pathwise, even with x-dependent attainable domains (`:680`–`:691`).
- Closed coordinate cells retain atoms and equality cases. The surface envelope contains full CP and is contained in GWC; therefore it inherits finite-sample simultaneous coverage (`:693`–`:775`). It can fill gaps and is not the exact full-conformal set or the exact full-cell LWC union.
- Uniform weak improvement over the old mathematical shortcut is proved on **identical nonnegative score representations**, matching cells/weak boundary conventions, and positive calibration scales (`:778`–`:875`). This permits equality. It does not establish signed-vs-capped, CQHR, exact full-cell LWC, or runtime dominance (`:881`–`:903`).
- The historical tied-mean implementation bug is explicitly described and its corrected comparator distinguished from the older executable (`:889`–`:898`). This must survive into final baseline documentation.
- Off-coordinate scores are precomputed, removing the extra dimension factor from each surface quantile: worst-case O(d n² + d n log n), plus a proved certified binary-search range and direct order-statistic localization (`:905`–`:1016`). Backward search remains valid without unproved global monotonicity.
- Direct CQR interpretation allows expansion and contraction, with domain lower endpoint −half the fitted quantile width (`:1018`–`:1040`). A symmetric reference rule plus whole-domain fallback covers zero calibration variance; this can be inefficient for heavily capped scores (`:1047`–`:1059`).
- Reported numerical formula checks supplement the proofs and explicitly do not claim exhaustive software verification (`:426`–`:459`). The standalone report's 2,400 trials are a four-DGP toy/model study, not a comprehensive real-data benchmark (`:478`–`:485`). Other empirical families live outside this particular report.

## What has already been fixed despite older open-item lists

The following are visible in current source and should not be presented as wholly unaddressed:

| Older concern | Current evidence |
|---|---|
| Missing n≥2 domain | `body.tex:82` states it. |
| Distinctness does not imply zero atom probability | `body.tex:154` and `:165` now explicitly assume no point mass. |
| Positive scales not specified for scalarization | `body.tex:243` includes positivity. |
| Nonunique mean index | `body.tex:633` uses a strict upper inequality. |
| Search contradiction includes maximizing index | `body.tex:646` now uses strictly greater indices. |
| Data-split quantile indexed outside final calibration subset | `supplementary.tex:84` uses I₂. |
| Wrong sign of quadratic linear coefficient | `supplementary.tex:322` now has −2 A μ. |
| Final selection proof only combines marginal bounds | `body.tex:674` now includes a common-oracle lower bound, matching the event route already developed in `supplementary.tex:444`–`:466`; still clean up its wrong/undefined h notation and state branchwise containment clearly. |

The current readiness JSON still describes several pre-repair states. Its recorded author-tag lines and scientific decisions are historical snapshots.

## Remaining substantive consistency work in the original manuscript

1. **Scalarization is still called invertible.** A maximum from R^d to R loses information; the affine coordinate threshold is a sublevel-set characterization, not an inverse of the multivariate map (`body.tex:205`, `:245`). The envelope note correctly uses an inverse of the one-dimensional self-score.
2. **Boundary conventions are inconsistent.** Original cells are half-open while their claimed union equals the closed GWC box (`body.tex:504`–`:508`); local bounds later use closed inequalities (`:607`). Reconcile definitions/proofs/implementation; atom-free marginals alone do not automatically eliminate equality with random data-dependent boundaries. The envelope's closed cover is an existing clean solution.
3. **GWC formula has an undefined critical-candidate expression at t equal to the calibration mean** (`body.tex:426`). The envelope's complete supremum formula explicitly treats t=m and separates maxima from minima (`signed_envelope.tex:643`–`:658`).
4. **Capped scores versus standing assumptions and implementation remain unresolved in the manuscript.** Capping creates atoms and possibly zero scale, and scalar-score jitter is not the same procedure as coordinate-residual jitter (`body.tex:148`–`:160`; response `:139`). The extension already supplies tie/zero-scale conventions that can become the unified theorem and implementation contract.
5. **Coverage balance is motivation, not a theorem from matching two moments.** Original statements at `body.tex:225` and `:270` are too broad if interpreted as equal marginal coverage or uniformly optimal widths. Heterogeneous tail shapes remain heterogeneous after standardization. The approximate-rectangle claim at `:483` is already qualified as empirical; do not upgrade it to an asymptotic theorem without conditions/proof.
6. **Complexity narrative needs alignment.** `body.tex:523` promises O(d² n log n) without the qualification made later at `:711`. If the signed extension uses x-specific width domains, the original global reuse claim at `:713` must be requalified rather than copied unchanged.
7. **Related-work distinctions remain substantive.** The blanket nonrectangular characterization (`body.tex:30`) conflicts with including rectangular shape-template competitors; `:45` overgeneralizes additional training requirements; `:47` still does not distinguish Point CHR from native CQHR adequately. Reviewer R2.M1 is explicitly open (`response_to_reviewers.tex:203`), and shape-template related-work integration is partial (`:252`).
8. **Harmless excluded algebra branch still misstated.** At `supplementary.tex:329`, A=0 and observed variance=0 would make the polynomial identically zero, not have only x=μ as its solution. Under the now-stated positive-scale domain this branch is excluded, but remove or correct it when consolidating the proof.

## Manuscript/evidence/reviewer integration boundary

- The manuscript still calls direct signed-score treatment future work (`body.tex:724`). That is now completed method development in the separate envelope source, so the paper's contribution/discussion/abstract and reviewer response must be reconciled.
- `sampling_provenance.md:3`–`:20` records fresh-only reruns and per-trial fitted data. It explicitly says current manuscript PDFs/captions/embedded figures remain historical and must not be represented as fresh reruns or the new method (`:51`–`:58`). This is now primarily an integration/provenance mapping task, not a blanket claim that all current synthetic work lacks fresh sampling.
- The response letter still says “No simulations were rerun” (`response_to_reviewers.tex:74`–`:76`) and reports 14 addressed / 5 partial / 1 open reviewer points (`:70`). Those are the September 4 revision's statuses, not an up-to-date assessment of the entire repository.
- The 22 response blocks consist of 20 reviewer points plus two editor requests. Remaining historically partial topics are intuition/common-event proof, assumptions/ties/signed scores, shape-template literature, shift choice, and coverage balance. Existing newer theory/evidence resolves parts but needs to be connected explicitly to the requested response.
- JMLR submission metadata still has a placeholder editor and provisional publication fields (`main.tex:67`–`:100`); the response identifies manuscript JMLR-25-3153-1 (`response_to_reviewers.tex:33`). Do not assume the publication fields are factual.

## Recommended scientific focus

Use one coherent narrative: calibration reuse via symmetric augmentation → exact ratio bounds → a practical valid rectangle → signed CQR generality and a precise same-score improvement theorem. Retain original TSCP as a clearly defined comparator and stepping stone. Integrate existing fresh data and reviewer-specific studies before commissioning many new experiments.

What still needs understanding is **when the improvement is material**, not merely whether the construction works: small n, heterogeneous scales, heavy/skewed tails, quantile-model quality, conservative base intervals, high output dimension, and the difference between sign representation gains and envelope approximation gains. Separate paired ratios from ratios of mean volumes; finite-reference conditional gains from unconditional behavior; and genuine coverage effects from small Monte Carlo fluctuations. These distinctions are already well articulated in the newer report and follow-up audit.

Robust scaling, conditional coverage, categorical/partially observed outputs, globally monotone search outside the proved region, or exhaustive full-cell LWC benchmarking are optional future work unless needed to support a final explicit claim. They are not automatically publication prerequisites. A useful theoretical addition would be a transparent sufficient-condition/example analysis for appreciable gains or equality, but a broad universal strict-efficiency theorem is neither currently established nor necessary to make a defensible paper.

