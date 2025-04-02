import numpy as np
from typing import Optional, List, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def make_multitarget_regression(
    n_samples: int = 100,
    n_features: int = 20,
    n_targets: int = 3,
    n_informative: int = 10,
    noise_list: Optional[List[float]] = None,
    coef: bool = False,
    random_state: Optional[int] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
    """
    Generate a regression problem with multiple targets and separate noise for each target.

    Parameters:
        n_samples (int): Number of samples.
        n_features (int): Total number of input features.
        n_targets (int): Number of target outputs.
        n_informative (int): Number of informative features per target.
        noise_list (List[float] or None): List of noise standard deviations for each target.
        coef (bool): Whether to return the coefficient vectors.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        X (ndarray): Input feature matrix of shape (n_samples, n_features).
        y (ndarray): Target matrix of shape (n_samples, n_targets).
        coefs (List[ndarray], optional): List of coefficient vectors for each target.
    """
    rng = np.random.default_rng(random_state)

    # Generate feature matrix
    X = rng.standard_normal(size=(n_samples, n_features))

    # Output arrays
    y = np.zeros((n_samples, n_targets))
    coef_list = []

    # Set default noise if not provided
    if noise_list is None:
        noise_list = [0.0] * n_targets
    elif len(noise_list) != n_targets:
        raise ValueError("Length of noise_list must match n_targets")

    for i in range(n_targets):
        # Create a sparse coefficient vector
        coef_i = np.zeros(n_features)
        informative_idx = rng.choice(n_features, size=n_informative, replace=False)
        coef_i[informative_idx] = rng.uniform(-100, 100, size=n_informative)
        coef_list.append(coef_i)

        # Generate target with specific noise
        y[:, i] = X @ coef_i + rng.normal(scale=noise_list[i], size=n_samples)

    if coef:
        return X, y, coef_list
    else:
        return X, y


def calibration_split(X, 
    y, test_cal_size=0.2, 
    cal_size=0.5, 
    random_state1=42, 
    random_state2 = 42):
    """
    Split the dataset into training, testing, and calibration sets in specified proportions.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target matrix of shape (n_samples,).
    test_cal_size : float, optional
        Fraction of the dataset to include in the combined (test + calibration) subset. Default: 0.2.
    cal_size : float, optional
        Fraction of the combined (test + calibration) subset to allocate for calibration. Default: 0.5.
    random_state : int, optional
        Seed for random number generation to ensure reproducible results. Default: 42.

    Returns
    -------
    tuple
        X_train, X_test, X_cal, y_train, y_test, y_cal : numpy.ndarray
        Training, test, and calibration splits of the features and targets.
    """
    # Split into training and combined test/calibration set
    X_train, X_test_cal, y_train, y_test_cal = train_test_split(X, y, test_size=test_cal_size, random_state=random_state1)
    
    # Further split the test/calibration set into test and calibration subsets
    X_test, X_cal, y_test, y_cal = train_test_split(X_test_cal, y_test_cal, test_size=cal_size, random_state=random_state2)

    return X_train, X_test, X_cal, y_train, y_test, y_cal

def scores_generator(X, y):

    # Train, calibration, and test split, default ratio: 80/10/10
    X_train, X_test, X_cal, y_train, y_test, y_cal = calibration_split(X, y)

    # Fit model to training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Vectorized calibration scores
    prediction = model.predict(X_cal)
    scores = np.abs(prediction - y_cal)

    # Test scores
    prediction_test = model.predict(X_test)
    scores_test = np.abs(prediction_test - y_test)

    return scores, scores_test