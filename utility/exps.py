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
from scipy.io import arff

# Import utility packages
from utility.data_generator import make_multitarget_regression
from utility.res_rescaled import check_coverage_rate, standardized_prediction
from utility.unscaled import unscaled_prediction, bonferroni_prediction
from utility.data_splitting import data_splitting_standardized_prediction, data_spliting_CHR_prediction, data_splitting_oracle_prediction, naive_prediction
from utility.copula import empirical_copula_prediction

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

def function_choice(scores, alpha, method, mu = None, std = None):
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
    #if method == "Scaled (Full)":
    #    return scaled_prediction(scores=scores, alpha=alpha, short_cut=False)
    #elif method == "Scaled (Shortcut)":
    #    return scaled_prediction(scores=scores, alpha=alpha, short_cut=True)
    #elif method == "Scaled (Data Splitting)":
    #    return data_splitting_scaled_prediction(scores=scores, alpha=alpha)

    # Standardized methods
    if method == "TSCP_LWC":
        return standardized_prediction(scores=scores, alpha=alpha, short_cut=False)
    elif method == "TSCP_R":
        return standardized_prediction(scores=scores, alpha=alpha, short_cut=True)
    elif method == "TSCP_GWC":
        return standardized_prediction(scores=scores, alpha=alpha, method= "GWC", short_cut=True)
    elif method == "TSCP_S":
        return data_splitting_standardized_prediction(scores=scores, alpha=alpha)
    elif method == "Population_oracle":
        return data_splitting_oracle_prediction(scores = scores, mu = mu, std = std, alpha=alpha)
    elif method == "Naive":
        return naive_prediction(scores=scores, alpha=alpha)
    
    # Point CHR
    elif method == "Point_CHR":
        return data_spliting_CHR_prediction(scores=scores, alpha=alpha)
    
    # Empirical copuls
    elif method == "Empirical_copula":
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
    method: str = "TSCP_R",
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
                    if method == "TSCP_LWC":
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

                        #Simulate a population oracle
                        if method == "Population_oracle":
                            oracle_X, oracle_y = make_multitarget_regression(
                                n_samples=100000,
                                n_features=10,
                                n_informative=10,
                                n_targets=dim,
                                noise_type=noise_type,
                                noise_list=noise_list,
                                random_state=seed+3,
                                coef=coef_true
                            )

                            oracle_prediction = model.predict(oracle_X)
                            oracle_res = np.abs(oracle_prediction - oracle_y)

                            mu = np.mean(oracle_res, axis=0)
                            std = np.std(oracle_res, axis=0, ddof = 1)

                            start = time.time()
                            region = function_choice(scores = scores_cal, alpha=alpha, method=method, mu = mu, std = std)
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

def heavy_t(
    dim_list: List[int],
    sample_list: List[int],
    alpha_list: List[float],
    df_list: List[int] = [2, 3, 10, 30, 50, 100],
    trials: int = 300,
    method: str = "TSCP_R",
    log_scale = False
) -> pd.DataFrame:

    outputs = []

    for alpha in alpha_list:
        for index_dim, dim in enumerate(dim_list):
            for index_sample, sample in enumerate(sample_list):
                for df in df_list:
                    # Generate training and test data
                    X, y, coef_true = make_multitarget_regression(
                        n_samples=8000,
                        n_features=10,
                        n_targets=dim,
                        noise_type="t",
                        df=df,
                        random_state=stable_hash(dim)
                    )
                    # Initialize model
                    model = LinearRegression()


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
                            noise_type="t",
                            df=df,
                            random_state=seed,
                            coef=coef_true
                        )

                        prediction_cal = model.predict(X_cal)
                        scores_cal = np.abs(prediction_cal - y_cal)

                        # Run prediction region method and record performance
                        if method == "TSCP_LWC":
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

                            #Simulate a population oracle
                            if method == "Population_oracle":
                                oracle_X, oracle_y = make_multitarget_regression(
                                    n_samples=10000,
                                    n_features=10,
                                    n_informative=10,
                                    n_targets=dim,
                                    noise_type="t",
                                    df=df,
                                    random_state=seed,
                                    coef=coef_true
                                )

                                oracle_prediction = model.predict(oracle_X)
                                oracle_res = np.abs(oracle_prediction - oracle_y)

                                mu = np.mean(oracle_res, axis=0)
                                std = np.std(oracle_res, axis=0, ddof = 1)

                                start = time.time()
                                region = function_choice(scores = scores_cal, alpha=alpha, method=method, mu = mu, std = std)
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
                        alpha, dim, sample, df, trials, "t",
                        np.mean(collection[0]), np.std(collection[0], ddof=1),
                        np.mean(collection[1]), np.std(collection[1], ddof=1),
                        np.median(collection[2]),
                        np.mean(collection[3])
                    ])

    columns = [
        "alpha", "n_dim", "n_cals", "df", "n_trials", "noise_type",
        "test_coverage_avg", "test_coverage_1std",
        "coverage_vol_avg", "coverage_vol_1std",
        "coverage_max_length_median",
        "runtime_avg"
    ]
    return pd.DataFrame(outputs, columns=columns)



def run_real_experiments(data, num_splits, alpha = 0.1, cal_size = 0.2, test_size = 0.2):

    methods = [
            "TSCP", 
            "TSCP-S", 
            "TSCP-GWC",
            "Point CHR", 
            "Emp. copula", 
            "Unscaled Max", 
            "Bonferroni"]

    # Load data
    if data == "stock":

        stock_portfolio_performance = fetch_ucirepo(id=390)
        
        X = stock_portfolio_performance.data.features
        y = stock_portfolio_performance.data.targets
        X = X.drop(columns=y.columns)
        y = y.map(lambda x: float(x.strip('%'))/100 if isinstance(x, str) and '%' in x else x)

        model = MultiTaskLasso(alpha=0.0001)
    
    if data == "rf1":

        df = arff.loadarff("real_exps/data/rf1.arff")
        df = pd.DataFrame(df[0]).dropna()

        X = (df.iloc[:, :64])
        y = (df.iloc[:, 64:])

        model = RandomForestRegressor(
                n_estimators=200,          
                max_depth=16,            
                min_samples_split=5,       
                min_samples_leaf=2,        
                max_features="sqrt",          
                bootstrap=True,            
                random_state=42,
                n_jobs=-1
            )

    if data == "rf2":

        df = arff.loadarff("real_exps/data/rf2.arff")
        df = pd.DataFrame(df[0]).dropna()

        X = (df.iloc[:, :576])
        y = (df.iloc[:, 576:])

        model = RandomForestRegressor(
                n_estimators=200,          
                max_depth=16,            
                min_samples_split=5,       
                min_samples_leaf=2,        
                max_features="sqrt",          
                bootstrap=True,            
                random_state=42,
                n_jobs=-1
            )

    if data == "scm1d":

        df = arff.loadarff("real_exps/data/scm1d.arff")
        df = pd.DataFrame(df[0]).dropna()

        X = (df.iloc[:, :280])
        y = (df.iloc[:, 280:])

        model = RandomForestRegressor(
                n_estimators=200,          
                max_depth=16,            
                min_samples_split=5,       
                min_samples_leaf=2,        
                max_features="sqrt",          
                bootstrap=True,            
                random_state=42,
                n_jobs=-1
            )

    if data == "scm20d":

        df = arff.loadarff("real_exps/data/scm20d.arff")
        df = pd.DataFrame(df[0]).dropna()

        X = (df.iloc[:, :61])
        y = (df.iloc[:, 61:])

        model = RandomForestRegressor(
                n_estimators=200,          
                max_depth=16,            
                min_samples_split=5,       
                min_samples_leaf=2,        
                max_features="sqrt",          
                bootstrap=True,            
                random_state=42,
                n_jobs=-1
            )

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
                random_state=42,
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
                max_depth=16,            
                min_samples_split=5,       
                min_samples_leaf=2,        
                max_features="sqrt",          # Use all features at each split
                bootstrap=True,            
                random_state=42,
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
                n_estimators=200,          
                max_depth=16,            
                min_samples_split=5,       
                min_samples_leaf=2,        
                max_features="sqrt",          # Use all features at each split
                bootstrap=True,            
                random_state=42,
                n_jobs=-1
            )

    if data == "energy":

        energy_efficiency = fetch_ucirepo(id=242) 
        
        X = energy_efficiency.data.features 
        y = energy_efficiency.data.targets 

        model= RandomForestRegressor(
                n_estimators=200,          
                max_depth=None,            
                min_samples_split=2,       
                min_samples_leaf=1,        
                max_features=1.0,          # Use all features at each split
                bootstrap=True,            
                random_state=77,
                n_jobs=-1
            )

    results = np.zeros((len(methods), 4, num_splits))

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
        n, d = scores_cal.shape

        # Run the methods
        for index, method in enumerate(methods):

            start = time.time()
            prediction = function_choice(scores_cal, alpha, method)
            results[index][0][i] = time.time()-start
            results[index][1][i] = check_coverage_rate(scores_test, prediction)
            results[index][2][i] = prediction.volume()
            results[index][3][i] = np.log10(prediction.volume())
        
    output = []
    for index, method in enumerate(methods):
        row = [
        method,  # String
        scores_cal.shape,  # Tuple
        scores_test.shape,  # Tuple
        np.mean(results[index][1]),  # Float
        np.std(results[index][1], ddof=1),  # Float
        np.mean(results[index][2]),  # Float
        np.std(results[index][2], ddof=1),  # Float
        np.mean(results[index][3]),  # Float
        np.std(results[index][3], ddof=1),  # Float
        np.mean(results[index][0])  # Float
        ]
        output.append(row)
        
    columns=["Methods","cal_size","test_size", "test_coverage_avg", "test_coverage_1std", "coverage_vol", "coverage_vol_1std", "coverage_vol_log", "coverage_vol_log_1std", "runtime_avg"]
        
    return pd.DataFrame(output, columns=columns)