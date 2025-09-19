# Import general packages
import hashlib
import numpy as np
import pandas as pd
import time
from typing import List, Optional

# Import training packages
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Dataset loader
from ucimlrepo import fetch_ucirepo 
from kagglehub import dataset_download
from scipy.io import arff

# Import utility packages
from utility.data_generator import make_multitarget_regression
from utility.res_rescaled import check_coverage_rate, scaled_prediction, standardized_prediction
from utility.unscaled import unscaled_prediction, bonferroni_prediction
from utility.data_splitting import data_splitting_scaled_prediction, data_splitting_standardized_prediction, data_spliting_CHR_prediction
from utility.copula import empirical_copula_prediction
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

    # Scaled methods
    if method == "Scaled (Full)":
        return scaled_prediction(scores=scores, alpha=alpha, short_cut=False)
    elif method == "Scaled (Shortcut)":
        return scaled_prediction(scores=scores, alpha=alpha, short_cut=True)
    elif method == "Scaled (Data Splitting)":
        return data_splitting_scaled_prediction(scores=scores, alpha=alpha)

    # Standardized methods
    elif method == "Standardized (Full)":
        return standardized_prediction(scores=scores, alpha=alpha, short_cut=False)
    elif method == "Standardized (Shortcut)":
        return standardized_prediction(scores=scores, alpha=alpha, short_cut=True)
    elif method == "Standardized (Data Splitting)":
        return data_splitting_standardized_prediction(scores=scores, alpha=alpha)
    
    # Point CHR
    elif method == "Point CHR":
        return data_spliting_CHR_prediction(scores=scores, alpha=alpha)
    
    # Empirical copuls
    elif method == "Empirical copula":
        return empirical_copula_prediction(scores=scores, alpha=alpha)

    # No scaling methods
    elif method == "Unscaled":
        return unscaled_prediction(scores=scores, alpha=alpha)
    elif method == "Bonferroni":
        return bonferroni_prediction(scores=scores, alpha=alpha)

def run_synthetic_experiment(
    dim_list: List[int],
    sample_list: List[int],
    alpha_list: List[float],
    noise_type: str = "Gaussian",
    noises_list: List[int] = None,
    trials: int = 300,
    method: str = "Standardized Shortcut",
    log_scale = False
) -> pd.DataFrame:

    outputs = []

    for alpha in alpha_list:
        for index_dim, dim in enumerate(dim_list):
            # Specify noise level per dimension
            if noises_list is None:
                noise_list = dim - np.arange(dim)
            else:
                noise_list = noises_list[index_dim]

            # Generate training and test data
            X, y, coef_true = make_multitarget_regression(
                n_samples=8000,
                n_features=10,
                n_targets=dim,
                noise_type=noise_type,
                noise_list=noise_list,
                random_state=stable_hash(dim)
            )
            # Initialize model
            model = LinearRegression()

            for index_sample, sample in enumerate(sample_list):

                # Initialize collection: [coverage_rate, volume, max_length, runtime] per trial
                collection = np.zeros((4, trials))

                # Run trials
                for i in range(trials):
                    seed = stable_hash(dim, sample, i)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=seed+42)
                    model.fit(X_train, y_train)
                    # Compute test scores for later evaluation
                    prediction_test = model.predict(X_test)
                    scores_test = np.abs(prediction_test - y_test)

                    # Generate calibration data using fixed coefficients
                    X_cal, y_cal = make_multitarget_regression(
                        n_samples=sample,
                        n_features=10,
                        n_informative=10,
                        n_targets=dim,
                        noise_type=noise_type,
                        noise_list=noise_list,
                        random_state=seed,
                        coef=coef_true
                    )

                    prediction_cal = model.predict(X_cal)
                    scores_cal = np.abs(prediction_cal - y_cal)

                    # Run prediction region method and record performance
                    if method == "Scaled (Full)" or method == "Standardized (Full)":
                        start = time.time()
                        regions, region = function_choice(scores=scores_cal, alpha=alpha, method=method)
                        collection[3][i] = time.time() - start
                        collection[0][i] = check_coverage_rate(scores=scores_test, regions=regions, one_rect=False)
                        volume = 0
                        for reg in regions:
                            volume += reg.volume()
                        if log_scale:
                            collection[1][i] = np.log10(volume)
                        else:
                            collection[1][i] = volume
                        collection[2][i] = np.max(region.length_along_dimensions())
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
                    alpha, dim, sample, trials, noise_type,
                    np.mean(collection[0]), np.std(collection[0], ddof=1),
                    np.mean(collection[1]), np.std(collection[1], ddof=1),
                    np.median(collection[2]),
                    np.mean(collection[3])
                ])

    columns = [
        "alpha", "n_dim", "n_cals", "n_trials", "noise_type",
        "test_coverage_avg", "test_coverage_1std",
        "coverage_vol_avg", "coverage_vol_1std",
        "coverage_max_length_median",
        "runtime_avg"
    ]
    return pd.DataFrame(outputs, columns=columns)

def run_real_experiments(data, num_splits, alpha = 0.1, cal_size = 0.2, test_size = 0.2):
    if data == "student":
        student_performance = fetch_ucirepo(id=320) 
        X = student_performance.data.features 
        y = student_performance.data.targets 
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        target_cols = ['G1', 'G2', 'G3']
        # Preprocessing: One-hot encode categorical columns, pass through numerical ones
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,          
                max_depth=None,            
                min_samples_split=2,       
                min_samples_leaf=1,        
                max_features=1.0,          # Use all features at each split
                bootstrap=True,            
                random_state=77,
                n_jobs=-1
            ))
        ])
    if data == "air":
        air_quality = fetch_ucirepo(id=360) 
        target_cols = ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]
        df = air_quality.data.features.drop(columns=["Date", "Time", "NMHC(GT)"])
        feature_cols = df.columns.difference(target_cols)
        df[feature_cols] = df[feature_cols].replace(-200, np.nan)
        imputer = SimpleImputer(strategy="mean")
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        df = df[(df[target_cols] != -200).all(axis=1)]

        X = df.drop(columns=target_cols)
        y = df[target_cols]
        model = RandomForestRegressor(
                n_estimators=200,          
                max_depth=None,            
                min_samples_split=2,       
                min_samples_leaf=1,        
                max_features=1.0,          # Use all features at each split
                bootstrap=True,            
                random_state=77,
                n_jobs=-1
            )
    if data == "crime":
        communities_and_crime = fetch_ucirepo(id=211) 
        X = communities_and_crime.data.features.drop(columns="State")
        X = X.loc[:, X.isna().mean() < 0.3] 
        y = communities_and_crime.data.targets 
        y = y.dropna()
        X = X.loc[y.index]
        model= RandomForestRegressor(
                n_estimators=300,          
                max_depth=None,            
                min_samples_split=2,       
                min_samples_leaf=1,        
                max_features=1.0,          # Use all features at each split
                bootstrap=True,            
                random_state=77,
                n_jobs=-1
            )
    if data == "energy":
        energy_efficiency = fetch_ucirepo(id=242) 
        X = energy_efficiency.data.features 
        y = energy_efficiency.data.targets 
        model= RandomForestRegressor(
                n_estimators=300,          
                max_depth=None,            
                min_samples_split=2,       
                min_samples_leaf=1,        
                max_features=1.0,          # Use all features at each split
                bootstrap=True,            
                random_state=77,
                n_jobs=-1
            )
    if data == "stock_portfolio":
        stock_portfolio_performance = fetch_ucirepo(id=390) 
        X = stock_portfolio_performance.data.features 
        y = stock_portfolio_performance.data.targets 
        model= RandomForestRegressor(
                n_estimators=300,          
                max_depth=None,            
                min_samples_split=2,       
                min_samples_leaf=1,        
                max_features=1.0,          # Use all features at each split
                bootstrap=True,            
                random_state=77,
                n_jobs=-1
            )

    output = {
        "Scaled Shortcut": np.zeros((3, num_splits)),
        "Standardized Shortcut": np.zeros((3, num_splits)),
        "Point CHR": np.zeros((3, num_splits)),
        "Splitting baseline": np.zeros((3, num_splits)),
        "Unscaled": np.zeros((3, num_splits)),
        "Empirical copula": np.zeros((3, num_splits))
    }
    for i in range(num_splits):
        X_train, X_cal_test, y_train, y_cal_test = train_test_split(X, y, test_size=test_size+cal_size, random_state=stable_hash(i))
        X_cal, X_test, y_cal, y_test = train_test_split(X_cal_test, y_cal_test, test_size=test_size/(cal_size+test_size), random_state=stable_hash(i))
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        prediction_cal = model.predict(X_cal)
        scores_cal = np.asarray(np.abs(prediction_cal-y_cal), dtype=np.float64)
        prediction_test = model.predict(X_test)
        scores_test = np.asarray(np.abs(prediction_test-y_test), dtype=np.float64)

        start = time.time()
        lpr = scaled_prediction(scores_cal, alpha=alpha)
        output["Scaled Shortcut"][2][i] = time.time()-start
        output["Scaled Shortcut"][0][i] = check_coverage_rate(scores_test, lpr)
        output["Scaled Shortcut"][1][i] = lpr.volume()

        start = time.time()
        lpr = standardized_prediction(scores_cal, alpha=alpha)
        output["Standardized Shortcut"][2][i] = time.time()-start
        output["Standardized Shortcut"][0][i] = check_coverage_rate(scores_test, lpr)
        output["Standardized Shortcut"][1][i] = lpr.volume()

        start = time.time()
        ds = data_splitting_scaled_prediction(scores_cal, alpha=alpha)
        output["Splitting baseline"][2][i] = time.time()-start
        output["Splitting baseline"][0][i] = check_coverage_rate(scores_test, ds)
        output["Splitting baseline"][1][i] = ds.volume()

        start = time.time()
        dschr = data_spliting_CHR_prediction(scores_cal, alpha=alpha)
        output["Point CHR"][2][i] = time.time()-start
        output["Point CHR"][0][i] = check_coverage_rate(scores_test, dschr)
        output["Point CHR"][1][i] = dschr.volume()

        start = time.time()
        npr = unscaled_prediction(scores_cal, alpha=alpha)
        output["Unscaled"][2][i] = time.time()-start
        output["Unscaled"][0][i] = check_coverage_rate(scores_test, npr)
        output["Unscaled"][1][i] = npr.volume()

        start = time.time()
        empc = empirical_copula_prediction(scores_cal, alpha=alpha)
        output["Empirical copula"][2][i] = time.time()-start
        output["Empirical copula"][0][i] = check_coverage_rate(scores_test, empc)
        output["Empirical copula"][1][i] = empc.volume()

    results = [
        ["Scaled Shortcut", scores_cal.shape, scores_test.shape, np.mean(output["Scaled Shortcut"][0]), np.std(output["Scaled Shortcut"][0], ddof=1), 
                              np.mean(output["Scaled Shortcut"][1]), np.std(output["Scaled Shortcut"][1], ddof = 1), 
                              np.mean(output["Scaled Shortcut"][2])],
        ["Standardized Shortcut", scores_cal.shape, scores_test.shape, np.mean(output["Standardized Shortcut"][0]), np.std(output["Standardized Shortcut"][0], ddof=1), 
                              np.mean(output["Standardized Shortcut"][1]), np.std(output["Standardized Shortcut"][1], ddof = 1), 
                              np.mean(output["Standardized Shortcut"][2])],                      
        ["Point CHR", scores_cal.shape, scores_test.shape, np.mean(output["Point CHR"][0]), np.std(output["Point CHR"][0], ddof=1), 
                              np.mean(output["Point CHR"][1]), np.std(output["Point CHR"][1], ddof = 1), 
                              np.mean(output["Point CHR"][2])],
        ["Splitting baseline", scores_cal.shape, scores_test.shape, np.mean(output["Splitting baseline"][0]), np.std(output["Splitting baseline"][0], ddof=1), 
                              np.mean(output["Splitting baseline"][1]), np.std(output["Splitting baseline"][1], ddof = 1), 
                              np.mean(output["Splitting baseline"][2])],
        ["Unscaled", scores_cal.shape, scores_test.shape, np.mean(output["Unscaled"][0]), np.std(output["Unscaled"][0], ddof=1), 
                              np.mean(output["Unscaled"][1]), np.std(output["Unscaled"][1], ddof = 1), 
                              np.mean(output["Unscaled"][2])],
        ["Empirical copula", scores_cal.shape, scores_test.shape, np.mean(output["Empirical copula"][0]), np.std(output["Empirical copula"][0], ddof=1), 
                              np.mean(output["Empirical copula"][1]), np.std(output["Empirical copula"][1], ddof = 1), 
                              np.mean(output["Empirical copula"][2])]
    ]
    columns=["Methods","cal_size","test_size", "test_coverage_avg", "test_coverage_1std", "coverage_vol", "coverage_vol_1std", "runtime_avg"]
        
    return pd.DataFrame(results, columns=columns)