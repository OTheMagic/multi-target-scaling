import numpy as np
from typing import Optional, List, Tuple, Union

def make_multitarget_regression(
    n_samples: int = 100,
    n_features: int = 10,
    n_targets: int = 3,
    n_informative: int = 10,
    noise_type: str = "Gaussian",
    noise_list: Optional[List[float]] = None,
    coef: Optional[List[float]] = None,
    random_state: Optional[int] = 42
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

    if coef is None:
        coef_list = []

    # Set default noise if not provided
    if len(noise_list) != n_targets:
        raise ValueError("Length of noise_list must match n_targets")

    for i in range(n_targets):
        if coef is not None:
            coef_i = coef[i]
            if noise_type == "Gaussian":
                y[:, i] = X @ coef_i + rng.normal(scale=noise_list[i], size=n_samples)
            elif noise_type == "Laplace":
                y[:, i] = X @ coef_i + rng.laplace(scale=noise_list[i], size=n_samples)
            elif noise_type == "Gamma":
                y[:, i] = X @ coef_i + rng.gamma(shape=i, scale=noise_list[i], size=n_samples)
            elif noise_type == "Mixed":
                y[:, i] = X @ coef_i + rng.normal(scale=noise_list[i], size=n_samples) if i < n_targets//2 else X @ coef_i + rng.laplace(scale=noise_list[i], size=n_samples)
        else:
            coef_i = np.zeros(n_features)
            informative_idx = rng.choice(n_features, size=n_informative, replace=False)
            coef_i[informative_idx] = rng.uniform(-10, 10, size=n_informative)
            coef_list.append(coef_i)

            if noise_type == "Gaussian":
                y[:, i] = X @ coef_i + rng.normal(scale=noise_list[i], size=n_samples)
            elif noise_type == "Laplace":
                y[:, i] = X @ coef_i + rng.laplace(scale=noise_list[i], size=n_samples)
            elif noise_type == "Gamma":
                y[:, i] = X @ coef_i + rng.gamma(shape=i, scale=noise_list[i], size=n_samples)
            elif noise_type == "Mixed":
                y[:, i] = X @ coef_i + rng.normal(scale=noise_list[i], size=n_samples) if i < n_targets//2 else X @ coef_i + rng.laplace(scale=noise_list[i], size=n_samples)

    if coef is not None:
        return X, y
    return X, y, coef_list

