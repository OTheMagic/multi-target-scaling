import numpy as np
import pandas as pd
from utility.exps import run_synthetic_experiment

dim_list = [2]
noise_list = [4, 1]
sample_list = [20]
trials_list = [1000]
alpha_list = [0.2]

df_ds = run_synthetic_experiment(dim_list, sample_list, alpha_list, trials_list, method="ds", noise_list=noise_list)
df_lpro = run_synthetic_experiment(dim_list, sample_list, alpha_list, trials_list, method="lpr_o", noise_list=noise_list)
df_npr = run_synthetic_experiment(dim_list, sample_list, alpha_list, trials_list, method="npr", noise_list=noise_list)
df_lpr = run_synthetic_experiment(dim_list, sample_list, alpha_list, trials_list, method="lpr", noise_list=noise_list)


df_ds["method"] = "DS"
df_lpr["method"] = "LPR"
df_lpro["method"] = "LPR-O"
df_npr["method"] = "NPR"

# Reorder columns so 'method' is first
def move_method_first(df):
    cols = df.columns.tolist()
    return df[["method"] + [col for col in cols if col != "method"]]

df_ds = move_method_first(df_ds)
df_lpr = move_method_first(df_lpr)
df_lpr_o = move_method_first(df_lpro)
df_npr = move_method_first(df_npr)

# Combine all into one DataFrame
df_all = pd.concat([df_ds,  df_npr, df_lpr_o, df_lpr], ignore_index=True)

df_all.to_csv("small_samples.csv")