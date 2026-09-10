# Signed Envelope: Results and Qualifications

## Completed Fitted Studies

- 225 synthetic configurations, 35,463 fresh train/calibration/test trials.
- 1,730 fresh fitted toy trials, with 1,000 training observations each.
- Eight real datasets, 200 splits each; six also have five target-alpha levels.
- All primary and pilot synthetic runs refit within each trial. Shared method
  comparisons use identical data and fits within that trial. The existing shape
  baseline uses its already-fresh, matching deterministic cohort.

## Main Findings

Across 179 absolute-score configurations (33,233 trials), no
coordinate-containment violations were found against the corrected old shortcut.
The median configuration's mean paired volume ratio is 0.9867;
the smallest is 0.6513. Averaging configuration ratios equally gives
0.9621, which is a descriptive summary, not a universal efficiency gain.

There is no uniform CQHR advantage. In the standard fitted Gaussian toys, signed
envelope volumes are generally similar to, or slightly larger than, CQHR.
In the misspecified-width toy (80 trials, common base alpha 0.1), mean volumes
are 14.476 versus 17.531,
with joint coverages 0.9034 versus
0.9064. The ratio of these mean volumes is
0.826; this differs
from the mean within-trial ratio reported in the paired table.

Signed shrinkage is visible in the conservative-interval toy (base alpha 0.02,
explicitly separate from the requested 0.1 comparisons). Mean envelope volume
is 3.622 versus base volume
4.110, with coverage 0.9043.
The plot fixes trial 0 and test covariate 0, rather than selecting a favorable example.

A search of the saved CQR trials found a mixed-sign witness: GWC adjustments
were (5.208, 5.612, 0.622), but envelope adjustments were
(5.208, 5.567, -0.672). Thus expansion of every GWC coordinate does not rule
out shrinkage of one envelope coordinate. This is not evidence that the entire
envelope lies below zero, or that its volume is smaller than the base box.
See `sign_witnesses.json` for
the exact archive and test index. No full-LWC computation was needed.

## Proof Scope and Search

Uniform weak improvement is proved for the same nonnegative score function,
positive calibration scales, and harmonized closed-cell rules. It does not
compare different signed/capped representations, does not dominate CQHR, and
does not say a smaller region has greater coverage. A tied-zero mean-cell bug
in the former shortcut is fixed and documented, including six saved witnesses.

The note proves both a certified binary-search range and a stronger direct
order-statistic localization rule. Optional `search="rank"` matched backward
search in all 9,200 saved formula cases. Backward search remains the default;
extra rank preprocessing need not help when only one surface is inspected.

## Deferred Costly Work

No further full-LWC computation is scheduled. `full_lwc_manual.ipynb` is disabled
by default, runs sequentially when enabled, and checkpoints per trial. Full-LWC
outputs that completed before the pause request are retained, but no claim about
runtime comparability is made: vectorized and historical implementations differ.

## Data and Reading Guide

`signed_envelope.tex` is the full derivation; `signed_envelope.pdf` is the reading
copy. `results/paired_overview.csv` indexes paired comparisons, and
`figures/study_manifest.json` maps study plots to configurations. Trial JSON/NPZ
files contain raw observations, scores, bounds, and seeds. Infinite-volume rates
and invalid numerical outputs must not be mistaken for finite estimates.
`results/signed_vs_capped_summary.csv` separates the score-representation comparison
from the positive-score shortcut comparison and reports the usable finite pairs.
Monte Carlo standard errors are over trials, not individual test points.
Volume-ratio error bars use only finite positive-reference pairs; the exported
summary reports their count. `results/final_audit.json` verifies all 35,463
primary/pilot archive checksums, data shapes, fresh draws, and fit metadata.

Superseded data are in `../quarantine/obsolete_synthetic_2026-09-08/`, with hash
manifests. Fresh tables were restored under the original notebook data paths;
their explicit repetition counts can exceed a smaller historical view because
shared configurations use the full newly generated cohort. Historical manuscript
PDFs/figures are not updated by this standalone-method report.
