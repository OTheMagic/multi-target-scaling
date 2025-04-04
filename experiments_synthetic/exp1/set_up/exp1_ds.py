import numpy as np
import pandas as pd
from utility.exps import run_synthetic_experiment

dim_list = np.arange(2, 11)
sample_list = np.arange(50, 1050, 10)
alpha_list = [0.2]

df_ds = run_synthetic_experiment(dim_list=dim_list, sample_list=sample_list, alpha_list=alpha_list, trials= 100, method="ds")
df_ds.to_csv("exp1/ds.csv")