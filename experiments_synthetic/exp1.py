import numpy as np
import pandas as pd
from utility.exps import run_synthetic_experiment

dim_list = [2]
sample_list = [30, 50, 100, 300, 500, 1000]
alpha_list = [0.1]

df_lpr = run_synthetic_experiment(dim_list=dim_list, sample_list=sample_list, alpha_list=alpha_list, method="Scaling-based")
df_lpr.to_csv("exp1/lpr.csv")
df_lpr_unit = run_synthetic_experiment(dim_list=dim_list, sample_list=sample_list, alpha_list=alpha_list, method="Scaling-based")
df_lpr_unit.to_csv("exp1/lpr_unit.csv")