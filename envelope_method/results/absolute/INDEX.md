# ABSOLUTE experiment index

> Numerical files are centralized under root `data/`. This source-side index is included on GitHub; local data is excluded. The earlier staged originals were removed by the author on September 10.

Every listed configuration, trial and saved result remains active under root `data/envelope_method/results/`. Full archives retain observations; scores archives retain every original non-X/y member; compact trials retain measurements in JSON/CSV. Storage counts below are read from the active checkpoints.

[Paper selection](../../../docs/PAPER_EXPERIMENT_PLAN.md) | [Retention register](../../../docs/EXPERIMENT_RETENTION.md) | [Storage guide](../../../docs/ARCHIVE_STORAGE_GUIDE.md) | [Results map](../README.md)

**179 configurations / 33,233 fitted trials.** Active storage: **778 full / 32,455 scores / 0 compact**. Trial NPZs: **3,707,062,235 bytes**; complete configuration directories: **4,170,655,243 bytes**. Staged originals under `deletable/` are excluded.

Each trial has fresh observations and refitting. Shared study views and comparator sidecars are not extra fits. Columns show actual settings and original source views. Student-t/Cauchy original generators use unit-scale noise despite their stored scale vectors; Gamma has coordinate-index shapes including zero. Read the paper plan before interpreting law labels.

## Cauchy

| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| [`898faf279901d34c`](../../../data/envelope_method/results/absolute/898faf279901d34c/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 200 | 0 | 0 | 148.15 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`53f83ed6b5d3df7a`](../../../data/envelope_method/results/absolute/53f83ed6b5d3df7a/config.json) | primary | 2 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.84 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`5966a2bbf52d4905`](../../../data/envelope_method/results/absolute/5966a2bbf52d4905/config.json) | primary | 2 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.99 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`1b3d0761e2c65529`](../../../data/envelope_method/results/absolute/1b3d0761e2c65529/config.json) | primary | 2 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 8.58 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`8d441ace2414ce13`](../../../data/envelope_method/results/absolute/8d441ace2414ce13/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.18 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`117125efaf4e9e4a`](../../../data/envelope_method/results/absolute/117125efaf4e9e4a/config.json) | primary | 2 | 1000 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 10.68 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`c675a2f311ed1346`](../../../data/envelope_method/results/absolute/c675a2f311ed1346/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.70 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`8b54266e1753ceb0`](../../../data/envelope_method/results/absolute/8b54266e1753ceb0/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.99 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`aa06302f21148403`](../../../data/envelope_method/results/absolute/aa06302f21148403/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.73 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`3cdb5988bc81b54a`](../../../data/envelope_method/results/absolute/3cdb5988bc81b54a/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.67 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`7f238136cc462b22`](../../../data/envelope_method/results/absolute/7f238136cc462b22/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.61 | syn_exps/cauchy/tscp_r_cauchy.csv |
| [`008218558ef528cc`](../../../data/envelope_method/results/absolute/008218558ef528cc/config.json) | primary | 10 | 1000 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 42.96 | syn_exps/cauchy/tscp_r_cauchy.csv |

## Gamma

| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| [`ff7e4cf9c72bbf13`](../../../data/envelope_method/results/absolute/ff7e4cf9c72bbf13/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 5.96 | syn_exps/gamma/tscp_r_gamma.csv |
| [`2bf67b2af12faa4f`](../../../data/envelope_method/results/absolute/2bf67b2af12faa4f/config.json) | primary | 2 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 5.99 | syn_exps/gamma/tscp_r_gamma.csv |
| [`938b71880914d823`](../../../data/envelope_method/results/absolute/938b71880914d823/config.json) | primary | 2 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 6.10 | syn_exps/gamma/tscp_r_gamma.csv |
| [`e080eba94ba3420e`](../../../data/envelope_method/results/absolute/e080eba94ba3420e/config.json) | primary | 2 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 6.47 | syn_exps/gamma/tscp_r_gamma.csv |
| [`17b0dfbfe5157558`](../../../data/envelope_method/results/absolute/17b0dfbfe5157558/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 6.84 | syn_exps/gamma/tscp_r_gamma.csv |
| [`e0b24d936f7b8e3a`](../../../data/envelope_method/results/absolute/e0b24d936f7b8e3a/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 27.14 | syn_exps/gamma/tscp_r_gamma.csv |
| [`8c87761e164d4cd3`](../../../data/envelope_method/results/absolute/8c87761e164d4cd3/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 27.41 | syn_exps/gamma/tscp_r_gamma.csv |
| [`ebd6c8ab5ffb0696`](../../../data/envelope_method/results/absolute/ebd6c8ab5ffb0696/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.11 | syn_exps/gamma/tscp_r_gamma.csv |
| [`7bf45a517e6e80f8`](../../../data/envelope_method/results/absolute/7bf45a517e6e80f8/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 30.86 | syn_exps/gamma/tscp_r_gamma.csv |
| [`cdad94cb73d7dbac`](../../../data/envelope_method/results/absolute/cdad94cb73d7dbac/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 33.62 | syn_exps/gamma/tscp_r_gamma.csv |

## Gaussian

| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| [`3d3354a844bc2064`](../../../data/envelope_method/results/absolute/3d3354a844bc2064/config.json) | omission repair | 2 | 12 | 0.1 | — | 80 / 20 | 3 | 3 | 0 | 0 | 0.04 | reviewer_exps/absolute_residual/_smoke/smoke_abs_res_trial.csv |
| [`25c5c7e6e87a0981`](../../../data/envelope_method/results/absolute/25c5c7e6e87a0981/config.json) | primary | 2 | 30 | 0.1 | — | 2,400 / 600 | 100 | 1 | 99 | 0 | 2.36 | reviewer_update/data/shape_template_standard_trial.csv |
| [`7014a1e7e401dbcc`](../../../data/envelope_method/results/absolute/7014a1e7e401dbcc/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.69 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`c7ab3684ea5e6952`](../../../data/envelope_method/results/absolute/c7ab3684ea5e6952/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 200 | 0 | 0 | 148.10 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`b0c7d01322f7625d`](../../../data/envelope_method/results/absolute/b0c7d01322f7625d/config.json) | primary | 2 | 50 | 0.1 | — | 2,400 / 600 | 100 | 1 | 99 | 0 | 2.39 | reviewer_update/data/shape_template_standard_trial.csv |
| [`c131bc26e404b839`](../../../data/envelope_method/results/absolute/c131bc26e404b839/config.json) | primary | 2 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.75 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`fc41a93cd92742a0`](../../../data/envelope_method/results/absolute/fc41a93cd92742a0/config.json) | primary | 2 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.78 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`187f67f0389b1c17`](../../../data/envelope_method/results/absolute/187f67f0389b1c17/config.json) | primary | 2 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.90 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`742c89e20c86c7f0`](../../../data/envelope_method/results/absolute/742c89e20c86c7f0/config.json) | primary | 2 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.93 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`962c86db91c9d73f`](../../../data/envelope_method/results/absolute/962c86db91c9d73f/config.json) | primary | 2 | 100 | 0.1 | — | 2,400 / 600 | 100 | 1 | 99 | 0 | 2.47 | reviewer_update/data/shape_template_standard_trial.csv |
| [`433e3887268f726d`](../../../data/envelope_method/results/absolute/433e3887268f726d/config.json) | primary | 2 | 200 | 0.1 | — | 2,400 / 600 | 100 | 1 | 99 | 0 | 2.62 | reviewer_update/data/shape_template_standard_trial.csv |
| [`d0de72594d6bde4f`](../../../data/envelope_method/results/absolute/d0de72594d6bde4f/config.json) | primary | 2 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 8.49 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`e699775642d132bb`](../../../data/envelope_method/results/absolute/e699775642d132bb/config.json) | primary | 2 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 8.52 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`2c57d009d65a59bf`](../../../data/envelope_method/results/absolute/2c57d009d65a59bf/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.07 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`c11c4d61391712fb`](../../../data/envelope_method/results/absolute/c11c4d61391712fb/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.11 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`ba873459937c95e9`](../../../data/envelope_method/results/absolute/ba873459937c95e9/config.json) | notebook/exploratory | 4 | 30 | 0.2 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 12.70 | smoke_test_coordinate_lengths.ipynb |
| [`244e53e26832d3da`](../../../data/envelope_method/results/absolute/244e53e26832d3da/config.json) | primary | 10 | 10 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.23 | reviewer_exps/absolute_residual/small_calibration_stress_trial.csv |
| [`d87f81a8c98e88c4`](../../../data/envelope_method/results/absolute/d87f81a8c98e88c4/config.json) | primary | 10 | 10 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 16.57 | reviewer_update/data/small_calibration_stress_trial.csv |
| [`200d70a07cc79df8`](../../../data/envelope_method/results/absolute/200d70a07cc79df8/config.json) | primary | 10 | 20 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 16.85 | reviewer_update/data/small_calibration_stress_trial.csv |
| [`b547102870232900`](../../../data/envelope_method/results/absolute/b547102870232900/config.json) | primary | 10 | 20 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.52 | reviewer_exps/absolute_residual/small_calibration_stress_trial.csv |
| [`089d250dfd930525`](../../../data/envelope_method/results/absolute/089d250dfd930525/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.23 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`12f6df5ed0cbf9f1`](../../../data/envelope_method/results/absolute/12f6df5ed0cbf9f1/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.00 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`32c1d010a6e2b1b7`](../../../data/envelope_method/results/absolute/32c1d010a6e2b1b7/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 16.99 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`34f29fdf675f6c46`](../../../data/envelope_method/results/absolute/34f29fdf675f6c46/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.67 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`380c9d403d142300`](../../../data/envelope_method/results/absolute/380c9d403d142300/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 16.99 | reviewer_update/data/partial_heteroskedasticity_trial.csv |
| [`3b03ed6f47fccf41`](../../../data/envelope_method/results/absolute/3b03ed6f47fccf41/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.67 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv; reviewer_exps/absolute_residual/small_calibration_stress_trial.csv |
| [`7d81a51e343d1fe5`](../../../data/envelope_method/results/absolute/7d81a51e343d1fe5/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.00 | reviewer_update/data/small_calibration_stress_trial.csv |
| [`b682c56d570d4fc4`](../../../data/envelope_method/results/absolute/b682c56d570d4fc4/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.67 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`cd7f2e9ea784cef6`](../../../data/envelope_method/results/absolute/cd7f2e9ea784cef6/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.00 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`d79154848910a1f5`](../../../data/envelope_method/results/absolute/d79154848910a1f5/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.67 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`ea6ad50df2f23051`](../../../data/envelope_method/results/absolute/ea6ad50df2f23051/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.67 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`f4a5ec9302c38c8e`](../../../data/envelope_method/results/absolute/f4a5ec9302c38c8e/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.00 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`163cefbee257c5e1`](../../../data/envelope_method/results/absolute/163cefbee257c5e1/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.97 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`2b8d5f88c3701053`](../../../data/envelope_method/results/absolute/2b8d5f88c3701053/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.97 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`5061d053dee84b8b`](../../../data/envelope_method/results/absolute/5061d053dee84b8b/config.json) | primary | 10 | 50 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.30 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`531a1c8417a38fd3`](../../../data/envelope_method/results/absolute/531a1c8417a38fd3/config.json) | primary | 10 | 50 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.29 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`64242769c7577482`](../../../data/envelope_method/results/absolute/64242769c7577482/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.97 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv; reviewer_exps/absolute_residual/small_calibration_stress_trial.csv |
| [`686fa458729fc0ea`](../../../data/envelope_method/results/absolute/686fa458729fc0ea/config.json) | notebook/exploratory | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.54 | smoke_test_dependent_noise.ipynb |
| [`7b72e544a1684a77`](../../../data/envelope_method/results/absolute/7b72e544a1684a77/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.97 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`98849d48594e6f7b`](../../../data/envelope_method/results/absolute/98849d48594e6f7b/config.json) | primary | 10 | 50 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.29 | reviewer_update/data/small_calibration_stress_trial.csv |
| [`aeba469712883769`](../../../data/envelope_method/results/absolute/aeba469712883769/config.json) | primary | 10 | 50 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.30 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`b437ebc4419604b2`](../../../data/envelope_method/results/absolute/b437ebc4419604b2/config.json) | primary | 10 | 50 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.29 | reviewer_update/data/partial_heteroskedasticity_trial.csv |
| [`b8741e3c31c43a32`](../../../data/envelope_method/results/absolute/b8741e3c31c43a32/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.52 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`b9bbb39d50bc6d3f`](../../../data/envelope_method/results/absolute/b9bbb39d50bc6d3f/config.json) | primary | 10 | 50 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.30 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`d6526006d04061b4`](../../../data/envelope_method/results/absolute/d6526006d04061b4/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.97 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`131108560420b102`](../../../data/envelope_method/results/absolute/131108560420b102/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.70 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv; reviewer_exps/absolute_residual/small_calibration_stress_trial.csv |
| [`1b5dae3499baaf54`](../../../data/envelope_method/results/absolute/1b5dae3499baaf54/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.05 | reviewer_update/data/contamination_stress_trial.csv |
| [`20f29f9d3c240799`](../../../data/envelope_method/results/absolute/20f29f9d3c240799/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.87 | reviewer_update/data/heterogeneity_sweep_trial.csv |
| [`32a6b9d9d00f98f1`](../../../data/envelope_method/results/absolute/32a6b9d9d00f98f1/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/small_calibration_stress_trial.csv |
| [`36495667b73626b1`](../../../data/envelope_method/results/absolute/36495667b73626b1/config.json) | primary | 10 | 100 | 0.2 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.04 | reviewer_update/data/alpha_sensitivity_trial.csv |
| [`38465b484ba0c135`](../../../data/envelope_method/results/absolute/38465b484ba0c135/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.59 | reviewer_exps/absolute_residual/heterogeneity_sweep_trial.csv |
| [`387f8c372dea3532`](../../../data/envelope_method/results/absolute/387f8c372dea3532/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/partial_heteroskedasticity_trial.csv |
| [`3af05032c595a9d7`](../../../data/envelope_method/results/absolute/3af05032c595a9d7/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.66 | reviewer_exps/absolute_residual/heterogeneity_sweep_trial.csv |
| [`45eaa881eb8c2209`](../../../data/envelope_method/results/absolute/45eaa881eb8c2209/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.79 | reviewer_update/data/heterogeneity_sweep_trial.csv |
| [`4990f83671588ead`](../../../data/envelope_method/results/absolute/4990f83671588ead/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`4fa819d1e265e012`](../../../data/envelope_method/results/absolute/4fa819d1e265e012/config.json) | primary | 10 | 100 | 0.05 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.70 | reviewer_exps/absolute_residual/alpha_sensitivity_trial.csv |
| [`52be93013ba6fc36`](../../../data/envelope_method/results/absolute/52be93013ba6fc36/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.25 | syn_exps/gaussian/tscp_r_unit_gaussian.csv; reviewer_exps/absolute_residual/heterogeneity_sweep_trial.csv |
| [`54bff25c4b5f8976`](../../../data/envelope_method/results/absolute/54bff25c4b5f8976/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.71 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`555f3d11439c0fd3`](../../../data/envelope_method/results/absolute/555f3d11439c0fd3/config.json) | primary | 10 | 100 | 0.2 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.71 | reviewer_exps/absolute_residual/alpha_sensitivity_trial.csv |
| [`619cf9d39985247a`](../../../data/envelope_method/results/absolute/619cf9d39985247a/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/alpha_sensitivity_trial.csv; reviewer_update/data/dependent_gaussian_trial.csv |
| [`84d53983e368be01`](../../../data/envelope_method/results/absolute/84d53983e368be01/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.01 | reviewer_update/data/heterogeneity_sweep_trial.csv |
| [`9b40adacea0e77c0`](../../../data/envelope_method/results/absolute/9b40adacea0e77c0/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 17.97 | reviewer_update/data/heterogeneity_sweep_trial.csv |
| [`9e3849efe4e2d4c2`](../../../data/envelope_method/results/absolute/9e3849efe4e2d4c2/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`b64f01e2be0135df`](../../../data/envelope_method/results/absolute/b64f01e2be0135df/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`c86c501f006d2b90`](../../../data/envelope_method/results/absolute/c86c501f006d2b90/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.05 | reviewer_update/data/contamination_stress_trial.csv |
| [`ce58cb1d05e133ac`](../../../data/envelope_method/results/absolute/ce58cb1d05e133ac/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/contamination_stress_trial.csv |
| [`d6299b059bdd7008`](../../../data/envelope_method/results/absolute/d6299b059bdd7008/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.70 | reviewer_exps/absolute_residual/heterogeneity_sweep_trial.csv |
| [`d9a64e430c639f07`](../../../data/envelope_method/results/absolute/d9a64e430c639f07/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.04 | reviewer_update/data/contamination_stress_trial.csv |
| [`da2a40fa18b278c6`](../../../data/envelope_method/results/absolute/da2a40fa18b278c6/config.json) | primary | 10 | 100 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/heterogeneity_sweep_trial.csv |
| [`e4e25cbf2c7206c6`](../../../data/envelope_method/results/absolute/e4e25cbf2c7206c6/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.70 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`e95c499b5ed65f54`](../../../data/envelope_method/results/absolute/e95c499b5ed65f54/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.70 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`edab386b7c1db952`](../../../data/envelope_method/results/absolute/edab386b7c1db952/config.json) | primary | 10 | 100 | 0.05 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 18.03 | reviewer_update/data/alpha_sensitivity_trial.csv |
| [`f351aaad2c92423a`](../../../data/envelope_method/results/absolute/f351aaad2c92423a/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.70 | reviewer_exps/absolute_residual/alpha_sensitivity_trial.csv; reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`f6f5f00f46624d90`](../../../data/envelope_method/results/absolute/f6f5f00f46624d90/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.40 | reviewer_exps/absolute_residual/heterogeneity_sweep_trial.csv |
| [`0f673c91315ad987`](../../../data/envelope_method/results/absolute/0f673c91315ad987/config.json) | primary | 10 | 300 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 20.97 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`18b478527752dc69`](../../../data/envelope_method/results/absolute/18b478527752dc69/config.json) | primary | 10 | 300 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 20.97 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`286621d7f96fa012`](../../../data/envelope_method/results/absolute/286621d7f96fa012/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.65 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`3043a9c01bdf5430`](../../../data/envelope_method/results/absolute/3043a9c01bdf5430/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.65 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`5879c5dffa9466f7`](../../../data/envelope_method/results/absolute/5879c5dffa9466f7/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.64 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`724dc17252863a3b`](../../../data/envelope_method/results/absolute/724dc17252863a3b/config.json) | primary | 10 | 300 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 20.97 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`7a4b60f4482017f2`](../../../data/envelope_method/results/absolute/7a4b60f4482017f2/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.64 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`ad69ada210c0f034`](../../../data/envelope_method/results/absolute/ad69ada210c0f034/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.64 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`b6f7b1a50b75a9cd`](../../../data/envelope_method/results/absolute/b6f7b1a50b75a9cd/config.json) | primary | 10 | 300 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 20.97 | reviewer_update/data/partial_heteroskedasticity_trial.csv |
| [`b8209472b686ed1d`](../../../data/envelope_method/results/absolute/b8209472b686ed1d/config.json) | primary | 10 | 300 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 20.97 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`da11d5e891bb15da`](../../../data/envelope_method/results/absolute/da11d5e891bb15da/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.13 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |
| [`0cd587ac15230a3d`](../../../data/envelope_method/results/absolute/0cd587ac15230a3d/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.91 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`18eaa827c2c39c4c`](../../../data/envelope_method/results/absolute/18eaa827c2c39c4c/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.92 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`1df9d08d006b3ec0`](../../../data/envelope_method/results/absolute/1df9d08d006b3ec0/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.58 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`3f24bbb092a6548e`](../../../data/envelope_method/results/absolute/3f24bbb092a6548e/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.59 | syn_exps/gaussian/tscp_r_gaussian.csv |
| [`47023132baf291ef`](../../../data/envelope_method/results/absolute/47023132baf291ef/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.58 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`50ac0b352be4aa67`](../../../data/envelope_method/results/absolute/50ac0b352be4aa67/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.91 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`74d35887601b9113`](../../../data/envelope_method/results/absolute/74d35887601b9113/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.91 | reviewer_update/data/partial_heteroskedasticity_trial.csv |
| [`8b952a52e3e951f7`](../../../data/envelope_method/results/absolute/8b952a52e3e951f7/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.58 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`cae258f653ffcb40`](../../../data/envelope_method/results/absolute/cae258f653ffcb40/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.92 | reviewer_update/data/dependent_gaussian_trial.csv |
| [`d2d8b9deba13c074`](../../../data/envelope_method/results/absolute/d2d8b9deba13c074/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.58 | reviewer_exps/absolute_residual/dependent_gaussian_trial.csv |
| [`ff833dc7a337ed28`](../../../data/envelope_method/results/absolute/ff833dc7a337ed28/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.02 | syn_exps/gaussian/tscp_r_unit_gaussian.csv |

## Laplace

| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| [`e99fd35d52f8ec7b`](../../../data/envelope_method/results/absolute/e99fd35d52f8ec7b/config.json) | primary | 2 | 10 | 0.1 | — | 6,400 / 1,600 | 10 | 1 | 9 | 0 | 1.05 | syn_exps/laplace/tscp_r_laplace_10sample.csv |
| [`4e11385a5254cc2a`](../../../data/envelope_method/results/absolute/4e11385a5254cc2a/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 200 | 0 | 0 | 148.13 | syn_exps/laplace/tscp_r_laplace.csv; syn_exps/laplace/tscp_r_laplace_30sample.csv |
| [`28275fa5c9e50afb`](../../../data/envelope_method/results/absolute/28275fa5c9e50afb/config.json) | primary | 2 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.80 | syn_exps/laplace/tscp_r_laplace.csv |
| [`c68e637a93a76f6f`](../../../data/envelope_method/results/absolute/c68e637a93a76f6f/config.json) | primary | 2 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.95 | syn_exps/laplace/tscp_r_laplace.csv |
| [`4076ab921805e50a`](../../../data/envelope_method/results/absolute/4076ab921805e50a/config.json) | primary | 2 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 8.54 | syn_exps/laplace/tscp_r_laplace.csv |
| [`47bde060b9939225`](../../../data/envelope_method/results/absolute/47bde060b9939225/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.14 | syn_exps/laplace/tscp_r_laplace.csv |
| [`6c3983b11c567ce5`](../../../data/envelope_method/results/absolute/6c3983b11c567ce5/config.json) | primary | 3 | 10 | 0.1 | — | 6,400 / 1,600 | 10 | 1 | 9 | 0 | 1.24 | syn_exps/laplace/tscp_r_laplace_10sample.csv |
| [`716f65960d11b69c`](../../../data/envelope_method/results/absolute/716f65960d11b69c/config.json) | primary | 3 | 30 | 0.1 | — | 6,400 / 1,600 | 30 | 1 | 29 | 0 | 2.20 | syn_exps/laplace/tscp_r_laplace_30sample.csv |
| [`b585714188896178`](../../../data/envelope_method/results/absolute/b585714188896178/config.json) | primary | 4 | 10 | 0.1 | — | 6,400 / 1,600 | 10 | 1 | 9 | 0 | 1.42 | syn_exps/laplace/tscp_r_laplace_10sample.csv |
| [`066cc3656c82fc7c`](../../../data/envelope_method/results/absolute/066cc3656c82fc7c/config.json) | primary | 4 | 30 | 0.1 | — | 6,400 / 1,600 | 30 | 1 | 29 | 0 | 2.65 | syn_exps/laplace/tscp_r_laplace_30sample.csv |
| [`8889452789b32cef`](../../../data/envelope_method/results/absolute/8889452789b32cef/config.json) | primary | 5 | 10 | 0.1 | — | 6,400 / 1,600 | 10 | 1 | 9 | 0 | 1.61 | syn_exps/laplace/tscp_r_laplace_10sample.csv |
| [`c95860972d574424`](../../../data/envelope_method/results/absolute/c95860972d574424/config.json) | primary | 5 | 30 | 0.1 | — | 6,400 / 1,600 | 30 | 1 | 29 | 0 | 3.09 | syn_exps/laplace/tscp_r_laplace_30sample.csv |
| [`f540630a66d49c54`](../../../data/envelope_method/results/absolute/f540630a66d49c54/config.json) | primary | 6 | 10 | 0.1 | — | 6,400 / 1,600 | 10 | 1 | 9 | 0 | 1.79 | syn_exps/laplace/tscp_r_laplace_10sample.csv |
| [`336c0699d8acb5d9`](../../../data/envelope_method/results/absolute/336c0699d8acb5d9/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.68 | syn_exps/laplace/tscp_r_laplace.csv; syn_exps/laplace/tscp_r_laplace_30sample.csv |
| [`bfb0aa69f3fa55a6`](../../../data/envelope_method/results/absolute/bfb0aa69f3fa55a6/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.98 | syn_exps/laplace/tscp_r_laplace.csv |
| [`c48cdf095d54f4f8`](../../../data/envelope_method/results/absolute/c48cdf095d54f4f8/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.72 | syn_exps/laplace/tscp_r_laplace.csv |
| [`659361edd67b47b9`](../../../data/envelope_method/results/absolute/659361edd67b47b9/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.66 | syn_exps/laplace/tscp_r_laplace.csv |
| [`f9a7e70628c18069`](../../../data/envelope_method/results/absolute/f9a7e70628c18069/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.60 | syn_exps/laplace/tscp_r_laplace.csv |
| [`ca3857868423270d`](../../../data/envelope_method/results/absolute/ca3857868423270d/config.json) | primary | 20 | 30 | 0.1 | — | 6,400 / 1,600 | 30 | 1 | 29 | 0 | 9.71 | syn_exps/laplace/tscp_r_laplace_30sample.csv |
| [`878075e078fd0866`](../../../data/envelope_method/results/absolute/878075e078fd0866/config.json) | primary | 30 | 30 | 0.1 | — | 6,400 / 1,600 | 30 | 1 | 29 | 0 | 14.11 | syn_exps/laplace/tscp_r_laplace_30sample.csv |
| [`5415b912e421c470`](../../../data/envelope_method/results/absolute/5415b912e421c470/config.json) | omission repair | 50 | 30 | 0.1 | — | 6,400 / 1,600 | 30 | 1 | 29 | 0 | 22.88 | syn_exps/laplace/tscp_r_laplace_30sample.csv |

## Mixed

| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| [`ae5799e85666f892`](../../../data/envelope_method/results/absolute/ae5799e85666f892/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.72 | syn_exps/mixed/tscp_r_mixed.csv |
| [`d3000264cb61a0f6`](../../../data/envelope_method/results/absolute/d3000264cb61a0f6/config.json) | primary | 2 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.79 | syn_exps/mixed/tscp_r_mixed.csv |
| [`886a9cfb8e55806b`](../../../data/envelope_method/results/absolute/886a9cfb8e55806b/config.json) | primary | 2 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.94 | syn_exps/mixed/tscp_r_mixed.csv |
| [`a143c8f2c10622b6`](../../../data/envelope_method/results/absolute/a143c8f2c10622b6/config.json) | primary | 2 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 8.53 | syn_exps/mixed/tscp_r_mixed.csv |
| [`02654f7544e0b55b`](../../../data/envelope_method/results/absolute/02654f7544e0b55b/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.12 | syn_exps/mixed/tscp_r_mixed.csv |
| [`139172f0f353f58a`](../../../data/envelope_method/results/absolute/139172f0f353f58a/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.68 | syn_exps/mixed/tscp_r_mixed.csv |
| [`382484961998da39`](../../../data/envelope_method/results/absolute/382484961998da39/config.json) | primary | 10 | 50 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.98 | syn_exps/mixed/tscp_r_mixed.csv |
| [`98edfda1f95cd191`](../../../data/envelope_method/results/absolute/98edfda1f95cd191/config.json) | primary | 10 | 100 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 29.72 | syn_exps/mixed/tscp_r_mixed.csv |
| [`bf191a2c58f7bafb`](../../../data/envelope_method/results/absolute/bf191a2c58f7bafb/config.json) | primary | 10 | 300 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 32.66 | syn_exps/mixed/tscp_r_mixed.csv |
| [`067044987bbdd665`](../../../data/envelope_method/results/absolute/067044987bbdd665/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.60 | syn_exps/mixed/tscp_r_mixed.csv |

## t

| Configuration | Role | d | n cal | Target alpha | Base alpha | Train / test | Trials | Full | Scores | Compact | Active MiB | Original source view |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| [`425c0b902f8ee037`](../../../data/envelope_method/results/absolute/425c0b902f8ee037/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.70 | syn_exps/t/tscp_r_t.csv |
| [`68629c3382377452`](../../../data/envelope_method/results/absolute/68629c3382377452/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.74 | syn_exps/t/tscp_r_t.csv |
| [`9b96811d1eab79fe`](../../../data/envelope_method/results/absolute/9b96811d1eab79fe/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.69 | syn_exps/t/tscp_r_t.csv |
| [`9e0e0843a489d22c`](../../../data/envelope_method/results/absolute/9e0e0843a489d22c/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.69 | syn_exps/t/tscp_r_t.csv |
| [`b0b90b9ac9f7cde9`](../../../data/envelope_method/results/absolute/b0b90b9ac9f7cde9/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.71 | syn_exps/t/tscp_r_t.csv |
| [`c011e610e6dd42a4`](../../../data/envelope_method/results/absolute/c011e610e6dd42a4/config.json) | primary | 2 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 7.72 | syn_exps/t/tscp_r_t.csv |
| [`131aa7b53a2a60c1`](../../../data/envelope_method/results/absolute/131aa7b53a2a60c1/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.11 | syn_exps/t/tscp_r_t.csv |
| [`96af6b0f5182ec04`](../../../data/envelope_method/results/absolute/96af6b0f5182ec04/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.08 | syn_exps/t/tscp_r_t.csv |
| [`9a486de0bf74418d`](../../../data/envelope_method/results/absolute/9a486de0bf74418d/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.10 | syn_exps/t/tscp_r_t.csv |
| [`a1801d419e65e648`](../../../data/envelope_method/results/absolute/a1801d419e65e648/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.07 | syn_exps/t/tscp_r_t.csv |
| [`b0c5c77ff616e56c`](../../../data/envelope_method/results/absolute/b0c5c77ff616e56c/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.07 | syn_exps/t/tscp_r_t.csv |
| [`ff58809b8b3e95c9`](../../../data/envelope_method/results/absolute/ff58809b8b3e95c9/config.json) | primary | 2 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 9.14 | syn_exps/t/tscp_r_t.csv |
| [`03c93ff4a0a731e3`](../../../data/envelope_method/results/absolute/03c93ff4a0a731e3/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.42 | syn_exps/t/tscp_r_t.csv |
| [`2de59f88339a4504`](../../../data/envelope_method/results/absolute/2de59f88339a4504/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 16.91 | reviewer_update/data/heavy_tail_stress_trial.csv |
| [`45c8c2bc743f6372`](../../../data/envelope_method/results/absolute/45c8c2bc743f6372/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 16.85 | reviewer_update/data/heavy_tail_stress_trial.csv |
| [`596dc7e3eaaa46cc`](../../../data/envelope_method/results/absolute/596dc7e3eaaa46cc/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.23 | syn_exps/t/tscp_r_t.csv |
| [`5c13f11f6e703f50`](../../../data/envelope_method/results/absolute/5c13f11f6e703f50/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.25 | syn_exps/t/tscp_r_t.csv |
| [`92c5ff720401d003`](../../../data/envelope_method/results/absolute/92c5ff720401d003/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.48 | syn_exps/t/tscp_r_t.csv |
| [`b3ad77b36a7f47a2`](../../../data/envelope_method/results/absolute/b3ad77b36a7f47a2/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.37 | syn_exps/t/tscp_r_t.csv |
| [`ccb901fa786b999a`](../../../data/envelope_method/results/absolute/ccb901fa786b999a/config.json) | primary | 10 | 30 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 16.87 | reviewer_update/data/heavy_tail_stress_trial.csv |
| [`dacc6d36269aa265`](../../../data/envelope_method/results/absolute/dacc6d36269aa265/config.json) | primary | 10 | 30 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 28.27 | syn_exps/t/tscp_r_t.csv |
| [`0fdb2103ef5c3c25`](../../../data/envelope_method/results/absolute/0fdb2103ef5c3c25/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.20 | syn_exps/t/tscp_r_t.csv |
| [`125df6c1299b7a7a`](../../../data/envelope_method/results/absolute/125df6c1299b7a7a/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.07 | syn_exps/t/tscp_r_t.csv |
| [`479e8557793279ec`](../../../data/envelope_method/results/absolute/479e8557793279ec/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.33 | syn_exps/t/tscp_r_t.csv |
| [`61d4b30241101336`](../../../data/envelope_method/results/absolute/61d4b30241101336/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.05 | syn_exps/t/tscp_r_t.csv |
| [`6b80ac242b8ff262`](../../../data/envelope_method/results/absolute/6b80ac242b8ff262/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.02 | syn_exps/t/tscp_r_t.csv |
| [`8c18c31879d5f852`](../../../data/envelope_method/results/absolute/8c18c31879d5f852/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.76 | reviewer_update/data/heavy_tail_stress_trial.csv |
| [`eadacd9e788dea0f`](../../../data/envelope_method/results/absolute/eadacd9e788dea0f/config.json) | primary | 10 | 500 | 0.1 | — | 6,400 / 1,600 | 200 | 1 | 199 | 0 | 35.25 | syn_exps/t/tscp_r_t.csv |
| [`ee5dff295f8147f9`](../../../data/envelope_method/results/absolute/ee5dff295f8147f9/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.67 | reviewer_update/data/heavy_tail_stress_trial.csv |
| [`ef3910ccf31ce8fd`](../../../data/envelope_method/results/absolute/ef3910ccf31ce8fd/config.json) | primary | 10 | 500 | 0.1 | — | 7,200 / 800 | 200 | 1 | 199 | 0 | 23.71 | reviewer_update/data/heavy_tail_stress_trial.csv |

## Reading one configuration

`config.json` fixes the design. `trial_*.json` records measurements, seeds, hashes and the declared storage tier. Legacy checkpoints without storage metadata mean full storage. A full `trial_*.npz` includes training/calibration/test observations; a scores NPZ omits only the six X/y arrays. Compact checkpoints intentionally have no NPZ. `trials.csv`/`summary.csv` are reporting exports, and `status.json` records completion. Comparator sidecars may be in adjacent `auxiliary/` or `cqr_baselines/` directories.

Active MiB counts all files in each configuration directory, including JSON/CSV, and uses bytes / 1,048,576. These counts are an index, not a substitute for `final_audit.py` or migration hash verification. Missing scores/full NPZs are errors and are not automatically redrawn. Preserve original source hashes for old comparator links and use the active hash to check the current NPZ bytes.

The author removed the earlier `deletable/` originals. Exact original observations are locally available only for retained full trials; scores archives continue to support their documented comparisons. See the [storage guide](../../../docs/ARCHIVE_STORAGE_GUIDE.md).
