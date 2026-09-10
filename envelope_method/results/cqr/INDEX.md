# CQR experiment index

> Numerical files are centralized under root `data/`. This source-side index is included on GitHub; local data is excluded. The earlier staged originals were removed by the author on September 10.

Every listed configuration, trial and saved result remains active under root `data/envelope_method/results/`. Full archives retain observations; scores archives retain every original non-X/y member; compact trials retain measurements in JSON/CSV. Storage counts below are read from the active checkpoints.

[Paper selection](../../../docs/PAPER_EXPERIMENT_PLAN.md) | [Retention register](../../../docs/EXPERIMENT_RETENTION.md) | [Storage guide](../../../docs/ARCHIVE_STORAGE_GUIDE.md) | [Results map](../README.md)

**46 configurations / 2,230 fitted trials.** Active storage: **541 full / 1,689 scores / 0 compact**. Trial NPZs: **178,161,171 bytes**; complete configuration directories: **205,303,089 bytes**. Staged originals under `deletable/` are excluded.

Each trial has fresh observations and refitting. Shared study views and comparator sidecars are not extra fits. Columns show actual settings and original source views. Student-t/Cauchy original generators use unit-scale noise despite their stored scale vectors; Gamma has coordinate-index shapes including zero. Read the paper plan before interpreting law labels.

## Gaussian

| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| [`783592e873de7398`](../../../data/envelope_method/results/cqr/783592e873de7398/config.json) | notebook/exploratory | 2 | 20 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.30 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`ab2bc8d0c58dac55`](../../../data/envelope_method/results/cqr/ab2bc8d0c58dac55/config.json) | notebook/exploratory | 2 | 20 | 0.1 | 0.5 | 800 / 200 | 200 | 1 | 199 | 0 | 4.24 | smoke_test_cqhr.ipynb |
| [`c90f0fcc46997140`](../../../data/envelope_method/results/cqr/c90f0fcc46997140/config.json) | notebook/exploratory | 2 | 20 | 0.1 | 0.1 | 800 / 200 | 200 | 1 | 199 | 0 | 4.30 | smoke_test_cqhr.ipynb; requested_base_alpha_0.1 |
| [`e373d0aaa3d0582f`](../../../data/envelope_method/results/cqr/e373d0aaa3d0582f/config.json) | notebook/exploratory | 2 | 20 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.30 | exps.ipynb: exploratory quantile sweep |
| [`3c3bedb541888294`](../../../data/envelope_method/results/cqr/3c3bedb541888294/config.json) | notebook/exploratory | 2 | 30 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.30 | exps.ipynb: quantile probes; requested_base_alpha_0.1 |
| [`679747124bb82d5d`](../../../data/envelope_method/results/cqr/679747124bb82d5d/config.json) | notebook/exploratory | 2 | 30 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.30 | exps.ipynb: quantile probes; requested_base_alpha_0.1 |
| [`a22ec809a6777341`](../../../data/envelope_method/results/cqr/a22ec809a6777341/config.json) | notebook/exploratory | 2 | 30 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.30 | exps.ipynb: quantile probes |
| [`b29fe3a20a383e6c`](../../../data/envelope_method/results/cqr/b29fe3a20a383e6c/config.json) | notebook/exploratory | 2 | 30 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.30 | exps.ipynb: quantile probes |
| [`08f839ba6a22d1ea`](../../../data/envelope_method/results/cqr/08f839ba6a22d1ea/config.json) | notebook/exploratory | 2 | 50 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.31 | exps.ipynb: exploratory quantile sweep |
| [`4517d8aac3372329`](../../../data/envelope_method/results/cqr/4517d8aac3372329/config.json) | notebook/exploratory | 2 | 50 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.31 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`61f8eb01d84b9ce9`](../../../data/envelope_method/results/cqr/61f8eb01d84b9ce9/config.json) | notebook/exploratory | 2 | 100 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.33 | exps.ipynb: exploratory quantile sweep |
| [`e6d4dd3797e99b49`](../../../data/envelope_method/results/cqr/e6d4dd3797e99b49/config.json) | notebook/exploratory | 2 | 100 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.33 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`6148e6adc624582c`](../../../data/envelope_method/results/cqr/6148e6adc624582c/config.json) | notebook/exploratory | 2 | 300 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.40 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`934e8b0ae96e3d1e`](../../../data/envelope_method/results/cqr/934e8b0ae96e3d1e/config.json) | notebook/exploratory | 2 | 300 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.40 | exps.ipynb: exploratory quantile sweep |
| [`16c55eaf72bdbfc2`](../../../data/envelope_method/results/cqr/16c55eaf72bdbfc2/config.json) | primary | 3 | 30 | 0.1 | 0.1 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.81 | requested_base_alpha_0.1 |
| [`4e113c187ebdadf2`](../../../data/envelope_method/results/cqr/4e113c187ebdadf2/config.json) | primary | 3 | 30 | 0.1 | 0.5 | 2,400 / 600 | 100 | 1 | 99 | 0 | 5.41 | reviewer_exps/cqr/capped_sample_size_check_trial.csv |
| [`5f474269f3ec6318`](../../../data/envelope_method/results/cqr/5f474269f3ec6318/config.json) | primary | 3 | 30 | 0.1 | 0.5 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.83 | reviewer_update/data/cqr_sample_size_trial.csv |
| [`8527e223b376ca56`](../../../data/envelope_method/results/cqr/8527e223b376ca56/config.json) | notebook/exploratory | 3 | 30 | 0.1 | 0.1 | 1,600 / 400 | 100 | 100 | 0 | 0 | 17.57 | smoke_test_cqr.ipynb; requested_base_alpha_0.1 |
| [`d62d892adec04062`](../../../data/envelope_method/results/cqr/d62d892adec04062/config.json) | primary | 3 | 30 | 0.1 | 0.1 | 2,400 / 600 | 100 | 100 | 0 | 0 | 23.05 | requested_base_alpha_0.1 |
| [`f7527d7d8d6b95da`](../../../data/envelope_method/results/cqr/f7527d7d8d6b95da/config.json) | notebook/exploratory | 3 | 30 | 0.1 | 0.9 | 1,600 / 400 | 100 | 1 | 99 | 0 | 4.42 | smoke_test_cqr.ipynb |
| [`928d1456b39f8c8f`](../../../data/envelope_method/results/cqr/928d1456b39f8c8f/config.json) | primary | 3 | 50 | 0.1 | 0.1 | 2,400 / 600 | 100 | 100 | 0 | 0 | 23.22 | requested_base_alpha_0.1 |
| [`96fd6cffe17799df`](../../../data/envelope_method/results/cqr/96fd6cffe17799df/config.json) | primary | 3 | 50 | 0.1 | 0.1 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.86 | requested_base_alpha_0.1 |
| [`c297b54bd9cb34ed`](../../../data/envelope_method/results/cqr/c297b54bd9cb34ed/config.json) | primary | 3 | 50 | 0.1 | 0.5 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.87 | reviewer_update/data/cqr_sample_size_trial.csv |
| [`f8e690ded42e5af5`](../../../data/envelope_method/results/cqr/f8e690ded42e5af5/config.json) | primary | 3 | 50 | 0.1 | 0.5 | 2,400 / 600 | 100 | 1 | 99 | 0 | 5.53 | reviewer_exps/cqr/capped_sample_size_check_trial.csv |
| [`1e011016d30adf10`](../../../data/envelope_method/results/cqr/1e011016d30adf10/config.json) | primary | 3 | 100 | 0.1 | 0.1 | 2,400 / 600 | 100 | 100 | 0 | 0 | 23.73 | requested_base_alpha_0.1 |
| [`3950e70bdd00a3c5`](../../../data/envelope_method/results/cqr/3950e70bdd00a3c5/config.json) | primary | 3 | 100 | 0.1 | 0.7 | 2,400 / 600 | 100 | 1 | 99 | 0 | 5.59 | reviewer_exps/cqr/capped_base_alpha_sweep_trial.csv |
| [`3f67600c90a404a8`](../../../data/envelope_method/results/cqr/3f67600c90a404a8/config.json) | primary | 3 | 100 | 0.1 | 0.5 | 2,400 / 600 | 100 | 1 | 99 | 0 | 7.14 | reviewer_exps/cqr/capped_base_alpha_sweep_trial.csv; reviewer_exps/cqr/capped_sample_size_check_trial.csv; reviewer_exps/cqr/shifted_sensitivity_trial.csv |
| [`4bdeb48d23efcc5d`](../../../data/envelope_method/results/cqr/4bdeb48d23efcc5d/config.json) | primary | 3 | 100 | 0.1 | 0.9 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.91 | reviewer_update/data/cqr_base_alpha_trial.csv |
| [`4d743aa44fbecf01`](../../../data/envelope_method/results/cqr/4d743aa44fbecf01/config.json) | primary | 3 | 100 | 0.1 | 0.1 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.91 | requested_base_alpha_0.1 |
| [`64fe748eadaee93f`](../../../data/envelope_method/results/cqr/64fe748eadaee93f/config.json) | primary | 3 | 100 | 0.1 | 0.3 | 2,400 / 600 | 100 | 1 | 99 | 0 | 5.79 | reviewer_exps/cqr/capped_base_alpha_sweep_trial.csv |
| [`65d93b0249f064be`](../../../data/envelope_method/results/cqr/65d93b0249f064be/config.json) | primary | 3 | 100 | 0.1 | 0.9 | 2,400 / 600 | 100 | 1 | 99 | 0 | 5.55 | reviewer_exps/cqr/capped_base_alpha_sweep_trial.csv |
| [`a860b4ca8ceb3408`](../../../data/envelope_method/results/cqr/a860b4ca8ceb3408/config.json) | primary | 3 | 100 | 0.1 | 0.5 | 2,400 / 600 | 30 | 1 | 29 | 0 | 2.35 | reviewer_update/data/cqr_base_alpha_trial.csv; reviewer_update/data/cqr_sample_size_trial.csv; reviewer_update/data/cqr_shift_trial.csv |
| [`ef72df201e316a43`](../../../data/envelope_method/results/cqr/ef72df201e316a43/config.json) | primary | 3 | 100 | 0.1 | 0.3 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.94 | reviewer_update/data/cqr_base_alpha_trial.csv |
| [`fee95965187727fd`](../../../data/envelope_method/results/cqr/fee95965187727fd/config.json) | primary | 3 | 100 | 0.1 | 0.7 | 2,400 / 600 | 30 | 1 | 29 | 0 | 1.92 | reviewer_update/data/cqr_base_alpha_trial.csv |
| [`04a924ad4b07dcaf`](../../../data/envelope_method/results/cqr/04a924ad4b07dcaf/config.json) | primary | 3 | 200 | 0.1 | 0.1 | 2,400 / 600 | 100 | 100 | 0 | 0 | 24.76 | requested_base_alpha_0.1 |
| [`29459e963d9c82d9`](../../../data/envelope_method/results/cqr/29459e963d9c82d9/config.json) | primary | 3 | 200 | 0.1 | 0.1 | 2,400 / 600 | 30 | 1 | 29 | 0 | 2.05 | requested_base_alpha_0.1 |
| [`995665f168ac9bf9`](../../../data/envelope_method/results/cqr/995665f168ac9bf9/config.json) | primary | 3 | 200 | 0.1 | 0.5 | 2,400 / 600 | 100 | 1 | 99 | 0 | 6.18 | reviewer_exps/cqr/capped_sample_size_check_trial.csv |
| [`e020dda90c999189`](../../../data/envelope_method/results/cqr/e020dda90c999189/config.json) | primary | 3 | 200 | 0.1 | 0.5 | 2,400 / 600 | 30 | 1 | 29 | 0 | 2.08 | reviewer_update/data/cqr_sample_size_trial.csv |
| [`200d36a63f6a1f21`](../../../data/envelope_method/results/cqr/200d36a63f6a1f21/config.json) | notebook/exploratory | 4 | 20 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.42 | exps.ipynb: exploratory quantile sweep |
| [`f76843c357b0bc41`](../../../data/envelope_method/results/cqr/f76843c357b0bc41/config.json) | notebook/exploratory | 4 | 20 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.42 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`a79ccba3ef5e641e`](../../../data/envelope_method/results/cqr/a79ccba3ef5e641e/config.json) | notebook/exploratory | 4 | 50 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.44 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`aea1987c8f3f4550`](../../../data/envelope_method/results/cqr/aea1987c8f3f4550/config.json) | notebook/exploratory | 4 | 50 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.44 | exps.ipynb: exploratory quantile sweep |
| [`a9241983c960e7e1`](../../../data/envelope_method/results/cqr/a9241983c960e7e1/config.json) | notebook/exploratory | 4 | 100 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.48 | exps.ipynb: exploratory quantile sweep |
| [`f2b175d70e3eccce`](../../../data/envelope_method/results/cqr/f2b175d70e3eccce/config.json) | notebook/exploratory | 4 | 100 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.48 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`046859dfdfb1ec94`](../../../data/envelope_method/results/cqr/046859dfdfb1ec94/config.json) | notebook/exploratory | 4 | 300 | 0.1 | 0.1 | 800 / 200 | 10 | 1 | 9 | 0 | 0.61 | exps.ipynb: exploratory quantile sweep; requested_base_alpha_0.1 |
| [`1ee588be4e365124`](../../../data/envelope_method/results/cqr/1ee588be4e365124/config.json) | notebook/exploratory | 4 | 300 | 0.1 | 0.8 | 800 / 200 | 10 | 1 | 9 | 0 | 0.62 | exps.ipynb: exploratory quantile sweep |

## Reading one configuration

`config.json` fixes the design. `trial_*.json` records measurements, seeds, hashes and the declared storage tier. Legacy checkpoints without storage metadata mean full storage. A full `trial_*.npz` includes training/calibration/test observations; a scores NPZ omits only the six X/y arrays. Compact checkpoints intentionally have no NPZ. `trials.csv`/`summary.csv` are reporting exports, and `status.json` records completion. Comparator sidecars may be in adjacent `auxiliary/` or `cqr_baselines/` directories.

Active MiB counts all files in each configuration directory, including JSON/CSV, and uses bytes / 1,048,576. These counts are an index, not a substitute for `final_audit.py` or migration hash verification. Missing scores/full NPZs are errors and are not automatically redrawn. Preserve original source hashes for old comparator links and use the active hash to check the current NPZ bytes.

The author removed the earlier `deletable/` originals. Exact original observations are locally available only for retained full trials; scores archives continue to support their documented comparisons. See the [storage guide](../../../docs/ARCHIVE_STORAGE_GUIDE.md).
