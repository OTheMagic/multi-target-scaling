import numpy as np
import pandas as pd
from utility.exps import run_synthetic_experiment

dim_list = np.arange(10, 101)
sample_list = [50, 100]
alpha_list = [0.2, 0.1]

df_npr = run_synthetic_experiment(dim_list=dim_list, sample_list=sample_list, alpha_list=alpha_list, trials= 100, method="npr")
df_npr.to_csv("exp2/npr.csv")