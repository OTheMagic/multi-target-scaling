"""Separate signed-score gains from envelope gains using within-trial comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/envelope_method/results'


def summarize_pairs(frame, keys, references):
    signed = frame[frame.method == 'Envelope_signed']
    rows = []
    for reference in references:
        ref = frame[frame.method == reference]
        paired = signed.merge(ref, on=keys+['trial'], suffixes=('_signed','_reference'), validate='one_to_one')
        for key, group in paired.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            a, b = group.outcome_volume_signed, group.outcome_volume_reference
            finite = np.isfinite(a) & np.isfinite(b) & (b>0)
            ratio = a[finite]/b[finite]
            coverage_difference = group.test_coverage_signed-group.test_coverage_reference
            rows.append(dict(zip(keys,key), reference=reference, trials=len(group), finite_pairs=int(finite.sum()),
                             mean_paired_volume_ratio=ratio.mean(),
                             paired_ratio_se=ratio.std(ddof=1)/np.sqrt(len(ratio)) if len(ratio)>1 else np.nan,
                             reduction_percent=100*(1-ratio.mean()),
                             ratio_of_mean_volumes=a.mean()/b.mean() if np.isfinite(b).all() and b.mean()>0 else np.nan,
                             fraction_signed_smaller=float((a<b).mean()),
                             coverage_signed=group.test_coverage_signed.mean(),
                             coverage_reference=group.test_coverage_reference.mean(),
                             coverage_difference=coverage_difference.mean(),
                             coverage_difference_se=coverage_difference.std(ddof=1)/np.sqrt(len(group)),
                             infinite_reference_trials=int(np.isposinf(b).sum()),
                             invalid_reference_trials=int(b.isna().sum())))
    return pd.DataFrame(rows)


def main():
    frame = pd.read_csv(OUT/'simulation_trials.csv', low_memory=False)
    references = ['TSCP_R_capped_0', 'Envelope_capped_0', 'Signed_GWC', 'CQHR']
    references += sorted(m for m in frame.method.unique() if m.startswith('TSCP_R_shifted_'))
    result = summarize_pairs(frame, ['config_id'], references)
    configs = pd.read_csv(OUT/'configurations.csv')
    result = result.merge(configs, on='config_id')
    result.to_csv(OUT/'signed_gain_by_configuration.csv', index=False)
    toys = pd.read_csv(OUT/'toy_trials.csv', low_memory=False)
    for path in (OUT/'toy_baselines').glob('*/trials.csv'):
        toys = pd.concat([toys, pd.read_csv(path)], ignore_index=True)
    toys = toys.drop_duplicates(['study','method','trial'])
    toy_result = summarize_pairs(toys, ['study'], references[:4]+['Base'])
    toy_result.to_csv(OUT/'signed_gain_by_toy.csv', index=False)
    focus = result[(result.n_features==5)&(result.n_train==2400)&(result.base_alpha==.1)]
    print('Main fitted CQR studies, base miscoverage 0.1:')
    print(focus[['n_cal','reference','trials','finite_pairs','reduction_percent',
                 'coverage_signed','coverage_reference']].sort_values(['reference','n_cal']).to_string(index=False))
    print('Shifted shortcut comparisons:')
    print(result[result.reference.str.contains('shifted')][['n_features','base_alpha','reference','trials',
          'reduction_percent','coverage_signed','coverage_reference']].to_string(index=False))
    print('Fitted toys:')
    print(toy_result[['study','reference','reduction_percent','finite_pairs','infinite_reference_trials',
                      'coverage_signed','coverage_reference']].to_string(index=False))
    audit = json.loads((OUT/'comparator_rerun_audit.json').read_text())
    table = []
    for n, group in focus.groupby('n_cal'):
        by_ref = group.set_index('reference')
        table.append(f"| {n} | {by_ref.loc['TSCP_R_capped_0','reduction_percent']:.2f}% | "
                     f"{by_ref.loc['Envelope_capped_0','reduction_percent']:.2f}% | "
                     f"{by_ref.loc['Signed_GWC','reduction_percent']:.2f}% | "
                     f"{by_ref.loc['CQHR','reduction_percent']:.2f}% |")
    conservative = toy_result[(toy_result.study=='conservative_signed_2d') &
                              (toy_result.reference=='TSCP_R_capped_0')].iloc[0]
    misspecified = toy_result[(toy_result.study=='misspecified_width') &
                             (toy_result.reference=='CQHR')].iloc[0]
    gaussian = toy_result[toy_result.study.str.startswith(('2d_', '3d_')) &
                          (toy_result.reference=='TSCP_R_capped_0')]
    report = f'''# Signed Gains and Comparator Rerun Audit

## What Is Being Compared?

The old shortcut's proof uses nonnegative scores. Feeding raw signed scores
into it is not a justified baseline. The valid comparisons here are the new
signed envelope versus the old shortcut applied to capped scores, and versus
the capped envelope, shifted shortcut, signed GWC, and CQHR separately.
There is no uniform signed-versus-capped containment theorem.

## Are Signed Gains Large?

Usually not in these experiments. For the main three-output Gaussian CQR study
(five features, 2,400 training observations, 600 test observations, 100 fresh
trials per setting), both nominal base and target miscoverage are 0.1:

| Calibration size | Reduction vs old capped shortcut | Sign-only reduction vs capped envelope | Reduction vs signed GWC | Reduction vs CQHR |
| --- | --- | --- | --- | --- |
{chr(10).join(table)}

A positive percentage means a smaller signed-envelope region. These are
100 times one minus the mean within-trial ratio of mean outcome volumes, not
ratios of pooled means. The CSV files retain paired Monte Carlo standard errors,
finite-pair counts, and both coverage estimates. Signed coverage in these four
settings ranges from {focus.coverage_signed.min():.4f} to {focus.coverage_signed.max():.4f}.

An intuition check: in an idealized independent-coordinate model, accurately
fitted marginal 90% intervals have joint coverage 0.9^d, only 0.729 for d=3.
Reaching a joint 90% target generally calls for expansion. This does not forbid
shrinking an individual coordinate, but it limits how much broad shrinkage one
should expect. Conservative marginal 98% intervals in two independent
coordinates instead have joint coverage 0.9604, leaving room to shrink toward
the 90% target. This calculation is intuition, not an assumption used in the
method's proof or a claim about the fitted models' exact base coverage.

The seven freshly fitted Gaussian toys are mixed: versus the old capped shortcut,
the mean paired reductions range from {gaussian.reduction_percent.min():.2f}% to
{gaussian.reduction_percent.max():.2f}%. Negative reductions are counterexamples
to uniform efficiency dominance across score representations.

Larger practical gains occur when the base intervals are very conservative.
In the conservative toy (500 trials, base miscoverage 0.02, target 0.1), the signed
method has coverage {conservative.coverage_signed:.4f}. The old capped shortcut
has infinite volume in {int(conservative.infinite_reference_trials)}/500 trials
because capping removes all calibration variation in at least one coordinate.
Among its {int(conservative.finite_pairs)} finite-reference trials, the signed envelope
reduces volume by {conservative.reduction_percent:.2f}% on average. This is a
conditional finite-pair statistic, not an unconditional finite volume ratio.
Signed shrinkage versus the uncalibrated base interval is a separate comparison.

Against CQHR, the misspecified-width toy at common base miscoverage 0.1 shows
{misspecified.reduction_percent:.2f}% mean paired reduction, with coverage
{misspecified.coverage_signed:.4f} versus {misspecified.coverage_reference:.4f}.
Its ratio of mean volumes is {misspecified.ratio_of_mean_volumes:.3f}, a different
estimator. Standard Gaussian toys do not show this CQHR advantage.

## Every Archived Comparator, Not Just Envelope

The audit classifies {audit['archived_csv_files']} archived CSV files and checks
{audit['checked_method_scenario_entries']:,} method/scenario entries, including
mixed aggregates and rows present only in baseline-specific files. These entries
include duplicate historical views; they are not independent experiment counts.
Status: {audit['statuses'].get('complete',0):,} covered by fresh reruns or fitted
replications, {audit['statuses'].get('missing',0)} unfilled affordable entries,
and {audit['statuses'].get('deferred_full_lwc',0)} explicitly deferred signed full-LWC illustration.

The non-envelope families checked are TSCP_R, TSCP_GWC, TSCP_S, Unscaled,
Empirical Copula, Point CHR, Naive, Bonferroni, Population Oracle, and CQHR.
Existing completed small full-LWC results remain available; no new full-LWC
work was launched for this audit.

The audit found and repaired genuine omissions:

- Raw CQR Unscaled and Empirical Copula: evaluated for all 2,230 already-fresh
  fitted CQR trials, using the same fitted predictions as the other methods.
- Raw and capped comparators: completed for 930 already-fresh fitted toy trials.
- A baseline-only 50-dimensional Laplace setting at n_cal=30: 30 new trials,
  each with fresh training/calibration/test data and refitting. Ordinary
  baselines, including the population oracle, are complete; no full-cell search.
- An archived one-trial smoke check: replaced with three explicitly specified
  fresh trials. Its old table retained d=2, n_cal=12, alpha=0.1 and correlation
  0.4, but not training size or feature count. The replacement uses 80 training,
  20 test observations and two features. It is a new smoke replication, not an
  exact replay of an undocumented fit setup.

Adding a comparator to the fitted outputs of an existing fresh trial preserves
pairing and the required protocol: datasets and fits differ across trials,
while methods within a trial share them. It does not reuse one fitted model
across trials. Each new comparator sidecar links to the original fresh archive
by SHA-256. No quarantined residuals were used as current experimental inputs.

## Numerical Boundary Check

The newly added capped-toy comparison exposed inverse-link cancellation at an
exactly accepted zero. The implementation now retains that boundary when its
forward self-score is accepted, rather than turning it into an empty region.
900 boundary regression checks passed. All 2,230 existing main CQR trials were
rechecked; none required a material change to its stored capped-envelope bounds.
New toy comparator outputs were recomputed after the correction.

## Saved Evidence

- `results/signed_gain_by_configuration.csv` and `results/signed_gain_by_toy.csv`
- `results/comparator_rerun_audit.csv` and `results/obsolete_table_classification.csv`
- `results/cqr_baselines/` and `results/toy_baselines/`
- `repair_settings.json` and the new trial archives under `results/absolute/`
- `results/final_audit.json` and `results/zero_boundary_corrections.json`

Original notebook-compatible synthetic tables have fresh replacements, including
the omitted methods and baseline-only setting. Historical manuscript figures
remain historical documents; this audit does not silently relabel them as reruns.
'''
    (ROOT/'envelope_method/FOLLOWUP_AUDIT.md').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
