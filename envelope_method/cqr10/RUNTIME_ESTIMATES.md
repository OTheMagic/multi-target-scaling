# Ten-Outcome Runtime Estimates

Measured on this Windows machine, one computation worker and one numerical-library thread.
These are projections from six fresh fitted pilot trials, not completed study results.

| Experiment | Input features | Fresh trials | Central time | Planning range |
|---|---:|---:|---:|---:|
| Base alpha 0.1: calibration-size sweep | 5 | 400 | 3.01 h | 2.26-4.82 h |
| Base alpha 0.5: calibration-size sweep | 5 | 400 | 2.86 h | 2.15-4.58 h |
| Base-alpha sweep: additional 0.3, 0.7, 0.9 settings | 5 | 300 | 1.88 h | 1.41-3.01 h |
| Optional n_cal=500,1000 at base alpha 0.1 | 5 | 200 | 1.53 h | 1.15-2.44 h |
| Base alpha 0.1: calibration-size sweep | 10 | 120 | 1.56 h | 1.17-2.49 h |
| Base alpha 0.5: calibration-size sweep | 10 | 120 | 1.52 h | 1.14-2.43 h |
| Base-alpha sweep: additional 0.3, 0.7, 0.9 settings | 10 | 90 | 0.95 h | 0.72-1.53 h |
| Optional n_cal=500,1000 at base alpha 0.1 | 10 | 60 | 0.78 h | 0.59-1.26 h |

**Core total: 1430 trials, 11.78 hours (planning allowance 8.84-18.85 hours).**

Default compact storage: approximately 0.16 GB; scores tier: 1.03 GB; full arrays/models: 5.76 GB. Reserve at least 1.5 times the chosen projection for variability.

Disk projections use saved pilot metadata/member sizes, with allowances for fingerprints and CSV copies. Compact and scores sweeps have not been benchmarked. Runtime still includes the historical full-save overhead; no saving-time speedup is claimed. Storage controls disk retention, not training memory or the number of model fits.

The base-alpha sweep shares its 0.1/0.5, n_cal=100 configurations with the two calibration-size sweeps. They are counted only once. Shift sensitivity reuses the same trial fit and adds seconds overall, not hundreds of new fits. That sharing is within the same experiment trial, never a reused data pool across trials.

The CSV gives one estimate for every individual configuration. Test size is 600; training size is 12,000, with 20% internally held out for early stopping, plus 3,000 separate diagnostic validation observations. All are fresh per trial.

These ranges are engineering allowances, not statistical confidence intervals. Alpha 0.3/0.7 fitting costs are interpolated between measured 0.1/0.5/0.9 pilots. Calibration-size overhead uses measured 30/50/100/200/500/1000 timings at alpha 0.1. Rare difficult fits, thermal throttling, competing work, and different software can change timings. Do not divide the single-worker estimate by the number of logical CPUs. The notebook supports bounded process parallelism but defaults to one worker.

Full LWC is excluded. The inexpensive TSCP_R shortcut is still included.
