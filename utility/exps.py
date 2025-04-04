# Import general packages
import hashlib
import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Union
from sklearn.linear_model import LinearRegression

# Import utility packages
from utility.data_generator import make_multitarget_regression, calibration_split
from utility.lpr import check_coverage_rate, one_rect_prediction_regions_nD
from utility.npr import no_scaling_prediction_region
from utility.ds import data_splitting_prediction_region
from utility.rectangle import Rectangle

def stable_hash(*args):
    key = "_".join(str(a) for a in args)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

def run_synthetic_experiment(
    dim_list,
    sample_list,
    alpha_list,
    trials_list,
    method: str = "lpr",
    noise_list = None
) -> pd.DataFrame:

    outputs = []
   
    for alpha in alpha_list:
        for index_dim, dim in enumerate(dim_list):
            for index_sample, sample in enumerate(sample_list):
                for trials in trials_list:
                    # Initialize accumulators
                    runtime_lpr = runtime_lpr_o = runtime_npr = runtime_ds = 0
                    c_cal_lpr = c_cal_lpr_o = c_cal_npr = c_cal_ds = 0
                    c_test_lpr = c_test_lpr_o = c_test_npr = c_test_ds = 0
                    c_vol_lpr = c_vol_lpr_o = c_vol_npr = c_vol_ds = 0

                    for i in range(trials):
                        # Use stable hashed seed
                        seed = stable_hash(index_dim, index_sample, i)
                        if noise_list is None:
                            noise_list = (dim - np.arange(dim)) * 5

                        # Generate synthetic data
                        X, y = make_multitarget_regression(
                            n_samples=sample*10,
                            n_features=10,
                            n_informative=10,
                            n_targets=dim,
                            noise_list=noise_list,
                            random_state=seed
                        )

                        # Split the data
                        X_train, X_test, X_cal, y_train, y_test, y_cal = calibration_split(X, y)

                        # Fit model
                        model = LinearRegression()
                        model.fit(X_train, y_train)

                        # Predict and score
                        prediction_cal = model.predict(X_cal)
                        prediction_test = model.predict(X_test)
                        scores_cal = np.abs(prediction_cal - y_cal)
                        scores_test = np.abs(prediction_test - y_test)

                        if method == "lpr":
                            start = time.time()
                            regions_lpr, region = one_rect_prediction_regions_nD(scores=scores_cal, alpha=alpha, short_cut=False)
                            runtime_lpr += time.time() - start

                            c_cal_lpr += check_coverage_rate(scores_cal, regions_lpr, one_rect=False)
                            c_test_lpr += check_coverage_rate(scores_test, regions_lpr, one_rect=False)
                            for reg in regions_lpr:
                                c_vol_lpr += reg.volume()

                        elif method == "lpr_o":
                            start = time.time()
                            region_lpr_o = one_rect_prediction_regions_nD(scores=scores_cal, alpha=alpha, short_cut=True)
                            runtime_lpr_o += time.time() - start

                            c_cal_lpr_o += check_coverage_rate(scores_cal, region_lpr_o, one_rect=True)
                            c_test_lpr_o += check_coverage_rate(scores_test, region_lpr_o, one_rect=True)
                            c_vol_lpr_o += region_lpr_o.volume()

                        elif method == "ds":
                            start = time.time()
                            region_ds = data_splitting_prediction_region(scores=scores_cal, alpha=alpha)
                            runtime_ds += time.time() - start

                            c_cal_ds += check_coverage_rate(scores_cal, region_ds, one_rect=True)
                            c_test_ds += check_coverage_rate(scores_test, region_ds, one_rect=True)
                            c_vol_ds += region_ds.volume()

                        elif method == "npr":
                            start = time.time()
                            region_npr = no_scaling_prediction_region(scores=scores_cal, alpha=alpha)
                            runtime_npr += time.time() - start

                            c_cal_npr += check_coverage_rate(scores_cal, region_npr, one_rect=True)
                            c_test_npr += check_coverage_rate(scores_test, region_npr, one_rect=True)
                            c_vol_npr += region_npr.volume()

                # Average over trials
                    avg = trials
                    if method == "lpr":
                        outputs.append([alpha, dim, sample, trials, c_vol_lpr / avg, c_cal_lpr / avg, c_test_lpr / avg, runtime_lpr / avg])
                    elif method == "lpr_o":
                        outputs.append([alpha, dim, sample, trials, c_vol_lpr_o / avg, c_cal_lpr_o / avg, c_test_lpr_o / avg, runtime_lpr_o / avg])
                    elif method == "ds":
                        outputs.append([alpha, dim, sample, trials, c_vol_ds / avg, c_cal_ds / avg, c_test_ds / avg, runtime_ds / avg])
                    elif method == "npr":
                        outputs.append([alpha, dim, sample, trials, c_vol_npr / avg, c_cal_npr / avg, c_test_npr / avg, runtime_npr / avg])

    # Return results as DataFrame
    columns = ["alpha", "n_dim", "n_scores", "n_trials", "coverage_vol", "coverage_cal", "coverage_test", "runtime"]
    return pd.DataFrame(outputs, columns=columns)