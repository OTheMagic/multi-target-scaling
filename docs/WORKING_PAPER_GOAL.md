# Working paper goal

This records the author's September 9, 2026 direction. It is provisional until the author supplies the complete goal document; that document should supersede this working summary.

## Central contribution

Use a simple coordinate-standardization idea to construct interpretable rectangular prediction regions for multiple outcomes. Reuse calibration observations without an additional split for estimating residual shape, preserve finite-sample simultaneous coverage under the stated assumptions, and achieve competitive region volume against strong rectangular comparators.

The scientific emphasis is the standardization idea and the statistical benefit of keeping calibration data available. The surface-envelope construction is the current technical implementation; comparison with the former shortcut explains its approximation quality and can sit mainly in the appendix.

## Claims to evaluate

1. Coordinate adaptation is useful when residual scales differ; homogeneous settings are an honest control.
2. Avoiding extra calibration splitting is particularly useful with limited calibration data. Compare methods at equal total calibration budgets.
3. Volume remains competitive at adequate simultaneous coverage across noise laws, output dependence, heterogeneous scales and heteroskedasticity.
4. Quantile-model experiments compare against native CQHR using the same fitted base intervals. CQHR does not have Point CHR's same extra shape/calibration split, so this comparison tests a separate benefit.
5. Heavy-tail experiments distinguish finite-sample coverage from efficiency, stability and finite-volume behavior. Infinite population moments violate the older manuscript's moment assumptions; the current envelope validity argument does not need those population moments under its stated conventions.
6. Cleanliness is an efficiency consideration. A single influential observation can distort mean/SD-based shape estimation and inflate volume. Exchangeable contamination need not invalidate coverage; temporal distribution shift is a different limitation.

## Presentation boundaries

- A rectangle is interpretable as one interval per output; this does not imply balanced coordinate coverage or optimality among all region shapes.
- Competitive means a fair, coverage-aware comparison with both wins and losses. No universal Point CHR or CQHR dominance is asserted.
- Improvements from standardization, avoiding the additional split, signed residuals, and tighter approximation are distinct mechanisms.
- Do not remove legitimate difficult observations from the primary benchmark. Retained/deleted outlier studies are labeled sensitivity analyses.
- Existing real-data random-split results do not establish time-forward forecasting or conditional-coverage guarantees.

## Selection and remaining work

Use [the experiment plan](PAPER_EXPERIMENT_PLAN.md) for the proposed body/appendix selection and [the retention register](EXPERIMENT_RETENTION.md) for protected experiments. An experiment can remain useful evidence even if it is not displayed in the paper. Nonselection is not deletion authorization.
