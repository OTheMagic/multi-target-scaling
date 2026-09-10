# Real-data comparison using envelope TSCP

Target joint coverage: 90%; 200 splits per dataset. Entries are mean (SD across splits). Volume is full outcome-space volume, product of full interval lengths. Baselines for the six cached datasets reuse the same saved fits; air/crime are reconstructed resplits/refits.

| Dataset | Method | Coverage | Volume | Infinite trials |
|---|---|---:|---:|---:|
| stock | TSCP (envelope) | 0.9516 (0.0610) | 8.867e-02 (3.756e-01) | 0/200 |
| stock | Old TSCP shortcut | 0.9559 (0.0589) | 1.192e-01 (5.475e-01) | 0/200 |
| stock | GWC | 0.9616 (0.0566) | 1.333e-01 (5.977e-01) | 0/200 |
| stock | Unscaled Max | 0.9398 (0.0572) | 3.856e+00 (1.007e+01) | 0/200 |
| stock | Empirical copula | 0.7269 (0.1127) | 6.445e-04 (6.274e-04) | 0/200 |
| stock | Point CHR | 1.0000 (0.0000) | ∞ | 200/200 |
| rf2 | TSCP (envelope) | 0.9019 (0.0166) | 1.243e+05 (2.485e+05) | 0/200 |
| rf2 | Old TSCP shortcut | 0.9019 (0.0166) | 1.245e+05 (2.488e+05) | 0/200 |
| rf2 | GWC | 0.9023 (0.0165) | 1.257e+05 (2.514e+05) | 0/200 |
| rf2 | Unscaled Max | 0.8996 (0.0162) | 1.004e+06 (6.585e+05) | 0/200 |
| rf2 | Empirical copula | 0.8932 (0.0164) | 1.362e+04 (1.278e+04) | 0/200 |
| rf2 | Point CHR | 0.9015 (0.0217) | 1.788e+04 (2.482e+04) | 0/200 |
| scm1d | TSCP (envelope) | 0.9015 (0.0152) | 9.973e+43 (1.709e+44) | 0/200 |
| scm1d | Old TSCP shortcut | 0.9015 (0.0152) | 9.981e+43 (1.710e+44) | 0/200 |
| scm1d | GWC | 0.9016 (0.0152) | 1.008e+44 (1.717e+44) | 0/200 |
| scm1d | Unscaled Max | 0.8999 (0.0157) | 1.794e+44 (2.635e+44) | 0/200 |
| scm1d | Empirical copula | 0.8931 (0.0164) | 7.129e+43 (8.775e+43) | 0/200 |
| scm1d | Point CHR | 0.9047 (0.0198) | 1.863e+44 (3.557e+44) | 0/200 |
| scm20d | TSCP (envelope) | 0.9011 (0.0159) | 9.991e+44 (9.787e+44) | 0/200 |
| scm20d | Old TSCP shortcut | 0.9011 (0.0159) | 1.000e+45 (9.797e+44) | 0/200 |
| scm20d | GWC | 0.9013 (0.0159) | 1.010e+45 (9.877e+44) | 0/200 |
| scm20d | Unscaled Max | 0.9007 (0.0157) | 1.767e+45 (1.914e+45) | 0/200 |
| scm20d | Empirical copula | 0.8922 (0.0171) | 7.719e+44 (7.799e+44) | 0/200 |
| scm20d | Point CHR | 0.9022 (0.0227) | 1.967e+45 (5.244e+45) | 0/200 |
| energy | TSCP (envelope) | 0.9212 (0.0488) | 2.742e+01 (1.748e+01) | 0/200 |
| energy | Old TSCP shortcut | 0.9231 (0.0481) | 2.775e+01 (1.765e+01) | 0/200 |
| energy | GWC | 0.9317 (0.0456) | 3.059e+01 (2.075e+01) | 0/200 |
| energy | Unscaled Max | 0.9168 (0.0484) | 6.313e+01 (2.724e+01) | 0/200 |
| energy | Empirical copula | 0.8867 (0.0606) | 2.165e+01 (1.800e+01) | 0/200 |
| energy | Point CHR | 0.8989 (0.0688) | 3.532e+01 (9.738e+01) | 0/200 |
| student | TSCP (envelope) | 0.9091 (0.0569) | 1.556e+03 (1.396e+03) | 0/200 |
| student | Old TSCP shortcut | 0.9121 (0.0563) | 1.587e+03 (1.414e+03) | 0/200 |
| student | GWC | 0.9160 (0.0558) | 1.712e+03 (1.642e+03) | 0/200 |
| student | Unscaled Max | 0.9041 (0.0550) | 1.206e+03 (1.019e+03) | 0/200 |
| student | Empirical copula | 0.8611 (0.0726) | 8.658e+02 (5.753e+02) | 0/200 |
| student | Point CHR | 0.9388 (0.0571) | 5.508e+03 (8.560e+03) | 0/200 |
| air | TSCP (envelope) | 0.9035 (0.0175) | 3.007e+05 (8.634e+04) | 0/200 |
| air | Old TSCP shortcut | 0.9036 (0.0175) | 3.011e+05 (8.644e+04) | 0/200 |
| air | GWC | 0.9040 (0.0175) | 3.035e+05 (8.736e+04) | 0/200 |
| air | Unscaled Max | 0.9019 (0.0189) | 1.183e+09 (4.580e+08) | 0/200 |
| air | Empirical copula | 0.8969 (0.0179) | 2.646e+05 (6.954e+04) | 0/200 |
| air | Point CHR | 0.9030 (0.0221) | 3.060e+05 (1.271e+05) | 0/200 |
| crime | TSCP (envelope) | 0.9077 (0.0371) | 1.521e+66 (1.495e+67) | 0/200 |
| crime | Old TSCP shortcut | 0.9078 (0.0370) | 1.525e+66 (1.499e+67) | 0/200 |
| crime | GWC | 0.9091 (0.0373) | 1.724e+66 (1.698e+67) | 0/200 |
| crime | Unscaled Max | 0.9085 (0.0376) | 1.261e+71 (1.165e+72) | 0/200 |
| crime | Empirical copula | 0.8624 (0.0453) | 1.010e+60 (7.797e+60) | 0/200 |
| crime | Point CHR | 0.9171 (0.0447) | 7.304e+62 (4.886e+63) | 0/200 |

Audit: 1,600 saved real splits verified; six datasets have all five envelope alpha levels (0.1, 0.3, 0.5, 0.7, 0.9), air/crime have alpha=0.1. All 1,200 cached-source hashes match. Saved coverage and volume recomputed from binary bounds/test scores for every envelope-cohort record; 32 envelope formula spot-replays match current code. Baseline coverage checked against paired test scores allowing 1e-12 boundary tolerance for decimal CSV lengths. No full-LWC runs launched.

The historical manuscript table uses residual-space half-width volumes; these full outcome volumes differ by 2^d. Do not mix the two conventions. The old manuscript TSCP row is not the new envelope row.

Scope: the separate 1,430-trial ten-output CQR extension remains unrun by design; full LWC is deferred. Existing synthetic completion evidence is results/final_audit.json (225 configurations, 35,463 archives); that full synthetic audit was not rerun by this table builder.

Point CHR correction (2026-09-09): Air and Crime now use the second calibration half's own conformal rank. Their earlier rows were too narrow. Original rf2 is unchanged; the one-row-deleted rf2 study already used the correct ranks. See rf2_remaining/chr_rank_fix_verification.json and chr_rank_correction_extra_real.csv.
