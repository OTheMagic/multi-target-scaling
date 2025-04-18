# Import general packages
import hashlib
import numpy as np
import pandas as pd
import time
from typing import List, Optional
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Import utility packages
from utility.data_generator import make_multitarget_regression
from utility.lpr import check_coverage_rate, one_rect_prediction_regions_nD
from utility.npr import no_scaling_prediction_region
from utility.ds import data_splitting_scaling_prediction_region, data_spliting_CHR_prediction_region
from utility.rectangle import Rectangle

def stable_hash(*args):
    """
    Create a reproducible hash from input arguments, useful for random seeds.

    Parameters
    ----------
    *args : Any
        Sequence of inputs to be hashed.

    Returns
    -------
    int
        A deterministic hash value based on inputs.
    """
    key = "_".join(str(a) for a in args)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

def function_choice(scores, alpha, method):
    """
    Select and run the prediction region method based on input string.

    Parameters
    ----------
    scores : np.ndarray
        Score values used to construct prediction regions.
    alpha : float
        Miscoverage level.
    method : str
        The method name specifying which region construction to use.

    Returns
    -------
    depends on method
        Either a single region (Rectangle) or list of regions.
    """
    if method == "LPR":
        return one_rect_prediction_regions_nD(scores=scores, alpha=alpha, short_cut=False)
    elif method == "LPR-O":
        return one_rect_prediction_regions_nD(scores=scores, alpha=alpha, short_cut=True)
    elif method == "DS-S":
        return data_splitting_scaling_prediction_region(scores=scores, alpha=alpha)
    elif method == "DS-CHR":
        return data_spliting_CHR_prediction_region(scores=scores, alpha=alpha)
    elif method == "NPR":
        return no_scaling_prediction_region(scores=scores, alpha=alpha)
    elif method == "Copula-N":
        pass  # Placeholder for future extension
    elif method == "Copula-DS":
        pass

def run_synthetic_experiment(
    dim_list: List[int],
    sample_list: List[int],
    alpha_list: List[float],
    noises_list: List[List[int]] = None,
    trials: int = 100,
    method: str = "LPR-O",
    log_scale = True
) -> pd.DataFrame:
    """
    Run synthetic experiments using various conformal prediction region methods.

    Parameters
    ----------
    dim_list : List[int]
        List of dimensionalities for target outputs.
    sample_list : List[int]
        List of sample sizes for calibration.
    alpha_list : List[float]
        List of miscoverage rates.
    noises_list : List[List[int]], optional
        Optional noise levels for each target dimension.
    trials : int, optional
        Number of trials to run per configuration. Default is 100.
    method : str, optional
        Prediction region method to use. Default is "LPR-O".

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing average test coverage, volume, length, and timing statistics.
    """

    outputs = []

    for alpha in alpha_list:
        for index_dim, dim in enumerate(dim_list):
            # Specify noise level per dimension
            if noises_list is None:
                noise_list = (dim - np.arange(dim)) * 2
            else:
                noise_list = noises_list[index_dim]

            # Generate training and test data
            X, y, coef_true = make_multitarget_regression(
                n_samples=5000,
                n_features=10,
                n_targets=dim,
                noise_list=noise_list,
                random_state=stable_hash(dim)
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=stable_hash(dim))

            # Fit regression model on training data
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Compute test scores for later evaluation
            prediction_test = model.predict(X_test)
            scores_test = np.abs(prediction_test - y_test)

            for index_sample, sample in enumerate(sample_list):
                # Initialize collection: [coverage_rate, volume, max_length, runtime] per trial
                collection = np.zeros((4, trials))
                for i in range(trials):
                    seed = stable_hash(dim, sample, i)

                    # Generate calibration data using fixed coefficients
                    X_cal, y_cal = make_multitarget_regression(
                        n_samples=sample,
                        n_features=10,
                        n_informative=10,
                        n_targets=dim,
                        noise_list=noise_list,
                        random_state=seed,
                        coef=coef_true
                    )

                    prediction_cal = model.predict(X_cal)
                    scores_cal = np.abs(prediction_cal - y_cal)

                    # Run prediction region method and record performance
                    if method == "LPR":
                        start = time.time()
                        regions, region = function_choice(scores=scores_cal, alpha=alpha, method=method)
                        collection[3][i] = time.time() - start
                        collection[0][i] = check_coverage_rate(scores=scores_test, regions=regions, one_rect=False)
                        lengths = np.zeros(dim)
                        volume = 0
                        for reg in regions:
                            volume += reg.volume()
                            lengths += reg.length_along_dimensions()
                        if log_scale:
                            collection[1][i] = np.log10(volume)
                        else:
                            collection[1][i] = volume
                        collection[2][i] = np.max(lengths)
                    else:
                        start = time.time()
                        region = function_choice(scores=scores_cal, alpha=alpha, method=method)
                        collection[3][i] = time.time() - start
                        collection[0][i] = check_coverage_rate(scores=scores_test, regions=region, one_rect=True)
                        if log_scale:
                            collection[1][i] = np.log10(region.volume())
                        else:
                            collection[1][i] = region.volume()
                        collection[2][i] = np.max(region.length_along_dimensions())

                # Store summary statistics per configuration
                outputs.append([
                    alpha, dim, sample, trials,
                    np.mean(collection[0]), np.std(collection[0], ddof=1),
                    np.mean(collection[1]), np.std(collection[1], ddof=1),
                    np.median(collection[2]),
                    np.mean(collection[3]), np.std(collection[3], ddof=1)
                ])

    columns = [
        "alpha", "n_dim", "n_cals", "n_trials",
        "test_coverage_avg", "test_coverage_1std",
        "coverage_vol_avg", "coverage_vol_1std",
        "coverage_max_length_median",
        "runtime_avg", "runtime_1std"
    ]
    return pd.DataFrame(outputs, columns=columns)