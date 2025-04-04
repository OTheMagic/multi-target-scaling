import numpy as np
import pandas as pd
from utility.exps import run_synthetic_experiment

dim_list = [10, 100]
alpha_list = [0.2]
sample_list = np.arange(50, 510, 10)
trials_list = [100, 500, 1000]

df_ds = run_synthetic_experiment(dim_list, sample_list, alpha_list, trials_list, method="ds")
df_lpro = run_synthetic_experiment(dim_list, sample_list, alpha_list, trials_list, method="lpr_o")

df_lpro.to_csv("interesting_small_exps/hdln_lpro.csv")
df_ds.to_csv("interesting_small_exps/hdln_ds.csv")