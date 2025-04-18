import numpy as np
import pandas as pd
from utility.exps import run_synthetic_experiment

dim_list = [2, 4, 10, 100]
sample_list = [10, 30, 50, 100, 200, 500]
alpha_list = [0.1]

df_ds = run_synthetic_experiment(dim_list=dim_list, sample_list=sample_list, alpha_list=alpha_list, method="DS-S")
df_ds = run_synthetic_experiment(dim_list=dim_list, sample_list=sample_list, alpha_list=alpha_list, method="DS-S")
df_ds.to_csv("exp1/ds.csv")