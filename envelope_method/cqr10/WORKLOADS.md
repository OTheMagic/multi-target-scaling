# Ten-outcome CQR Extension

1. Measure bounded training and calibration pilots before a full study.
2. Freeze a quality-checked model configuration and timing assumptions.
3. Implement separate fresh-data trial checkpoints and a manual notebook.
4. Test the runner, checkpoint resume, saved arrays, and notebook without running
   the expensive study. Full LWC is excluded throughout this extension.

The ten outcomes use five input features, independent Gaussian noise with
standard deviations 10 through 1, and newly drawn training, validation,
calibration, and test samples in every trial. Only the data-generating
coefficients are fixed across trials. Conformal methods share each trial's fit.
Previous three-outcome cohorts and obsolete-data quarantine remain untouched.

## Completed Checkpoint (2026-09-09)

- Model pilot: six single-quantile fits comparing legacy, stronger plain, and
  linear-initialized gradient boosting. Default chosen from this separate pilot.
- Full runtime/quality pilot: six fresh trials, 120 quantile models; five/ten
  input features and base alphas 0.1/0.5/0.9. All diagnostic thresholds pass.
- Grid: all 22 original primary CQR settings, 1,430 trials. Optional larger
  calibration settings add four configurations / 260 trials. Full study NOT run.
- Runtime estimate: 11.78 single-worker hours for the core, planning allowance
  8.84-18.85 hours. Optional grid adds 2.31 hours. Runtime retains the measured
  full-save pilot overhead; no compact speedup is assumed.
- Storage default is compact. Estimated core disk: 0.16 GB compact, 1.03 GB
  scores, 5.76 GB full. These are projections from saved pilot metadata and
  compressed members, including export allowances; no formal study was run.
- Ready notebook: `cqr10_experiments.ipynb`; full execution disabled by default.
  Per-configuration timing details are in `runtime_estimates.csv`.
- Storage revision verification: 21 runner tests passed, including all-tier
  scientific equivalence, compact resume, metadata corruption, missing required
  artifacts, explicit upgrade rejection and two-process Windows execution.
  All six original full pilot archives/models and method metrics still reproduce;
  no pilot files were rewritten or migrated.
  Six notebook code cells executed with training blocked. Structural checks
  passed; full nbformat schema/live Jupyter validation unavailable in this
  environment. Dependency installation was attempted but packages unavailable.
- Existing protocol regression checks passed: fresh draws/fits for both
  synthetic APIs, 500 tied-score cases and 900 exact-zero boundary checks.
- No full LWC computation performed. Existing shared evaluator gains only an
  optional envelope-search argument; its backward-search default is unchanged.
- CLI/notebook expose `compact`, `scores`, `full` and an output directory.
  Storage stays outside the scientific configuration, preserving IDs and seeds.
  Higher-storage reruns require a separate output directory; richer existing
  checkpoints remain richer when resumed with a compact request.
