import numpy as np
from typing import Optional, List, Tuple, Union


def _validate_regression_inputs(n_features, n_targets, n_informative, noise_list, coef):
    """Validate shared multi-target regression generator inputs."""
    if n_targets <= 0:
        raise ValueError("n_targets must be positive.")
    if n_informative > n_features:
        raise ValueError("n_informative cannot exceed n_features.")
    if noise_list is None:
        noise_list = np.ones(n_targets)
    noise_list = np.asarray(noise_list, dtype=float)
    if len(noise_list) != n_targets:
        raise ValueError("Length of noise_list must match n_targets.")
    if np.any(noise_list < 0):
        raise ValueError("noise_list entries must be nonnegative.")
    if coef is not None and len(coef) != n_targets:
        raise ValueError("Length of coef must match n_targets.")
    return noise_list


def _make_design_and_coefficients(
    rng,
    n_samples,
    n_features,
    n_targets,
    n_informative,
    coef,
):
    """Generate X and either reuse or create one coefficient vector per target."""
    X = rng.standard_normal(size=(n_samples, n_features))
    coef_list = [] if coef is None else None

    for i in range(n_targets):
        if coef is not None:
            coef_i = np.asarray(coef[i])
        else:
            coef_i = np.zeros(n_features)
            informative_idx = rng.choice(n_features, size=n_informative, replace=False)
            coef_i[informative_idx] = rng.uniform(-10, 10, size=n_informative)
            coef_list.append(coef_i)
        yield X, coef_i, coef_list


def _draw_noise(rng, noise_type, scale, n_samples, target_index, n_targets, df):
    """Draw one target's additive noise."""
    if noise_type == "gaussian":
        return rng.normal(scale=scale, size=n_samples)
    if noise_type == "laplace":
        return rng.laplace(scale=scale, size=n_samples)
    if noise_type == "gamma":
        return rng.gamma(shape=target_index, scale=scale, size=n_samples)
    if noise_type == "mixed":
        if target_index < n_targets // 2:
            return rng.normal(scale=scale, size=n_samples)
        return rng.laplace(scale=scale, size=n_samples)
    if noise_type == "cauchy":
        return rng.standard_cauchy(size=n_samples)
    if noise_type == "t":
        if df is None:
            raise ValueError("df must be provided when noise_type='t'.")
        return rng.standard_t(df=df, size=n_samples)
    raise ValueError(f"Unknown noise_type: {noise_type}")


def _dependent_gaussian_correlation(n_targets, correlation, correlation_structure):
    """Build a valid target-target correlation matrix for Gaussian noise."""
    if n_targets <= 0:
        raise ValueError("n_targets must be positive.")

    structure = correlation_structure.lower()
    if structure == "equicorrelated":
        lower = -1 / (n_targets - 1) if n_targets > 1 else -1
        if not lower < correlation < 1:
            raise ValueError(
                "For equicorrelated noise, correlation must be in "
                f"({lower}, 1)."
            )
        corr = np.full((n_targets, n_targets), correlation)
        np.fill_diagonal(corr, 1.0)
        return corr

    if structure == "ar1":
        if not -1 < correlation < 1:
            raise ValueError("For AR(1) noise, correlation must be in (-1, 1).")
        indices = np.arange(n_targets)
        return correlation ** np.abs(indices[:, None] - indices[None, :])

    raise ValueError(
        "correlation_structure must be either 'equicorrelated' or 'ar1'."
    )


def make_multitarget_regression(
    n_samples: int = 100,
    n_features: int = 10,
    n_targets: int = 3,
    n_informative: int = 10,
    noise_type: str = "Gaussian",
    noise_list: Optional[List[float]] = None,
    df: Optional[int] = None,
    coef: Optional[List[float]] = None,
    random_state: Optional[int] = 42,
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
    noise_list = _validate_regression_inputs(
        n_features=n_features,
        n_targets=n_targets,
        n_informative=n_informative,
        noise_list=noise_list,
        coef=coef,
    )

    rng = np.random.default_rng(random_state)
    noise_type_normalized = noise_type.lower()
    y = np.zeros((n_samples, n_targets))

    for i, (X, coef_i, coef_list) in enumerate(_make_design_and_coefficients(
        rng=rng,
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        n_informative=n_informative,
        coef=coef,
    )):
        y[:, i] = X @ coef_i + _draw_noise(
            rng=rng,
            noise_type=noise_type_normalized,
            scale=noise_list[i],
            n_samples=n_samples,
            target_index=i,
            n_targets=n_targets,
            df=df,
        )

    if coef is not None:
        return X, y
    return X, y, coef_list


def make_multitarget_regression_dependent_noise(
    n_samples: int = 100,
    n_features: int = 10,
    n_targets: int = 3,
    n_informative: int = 10,
    noise_type: str = "Gaussian",
    noise_list: Optional[List[float]] = None,
    df: Optional[int] = None,
    coef: Optional[List[float]] = None,
    random_state: Optional[int] = 42,
    correlation: float = 0.5,
    correlation_structure: str = "equicorrelated",
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[np.ndarray]]]:
    """
    Generate multi-target regression data with dependent Gaussian target noise.

    This has the same return contract as `make_multitarget_regression`, but the
    additive noise vector for each sample is drawn jointly across targets:

        epsilon_i ~ N(0, D R D),

    where D is diagonal with entries from `noise_list` and R is either an
    equicorrelated or AR(1) correlation matrix.
    """
    if noise_type.lower() != "gaussian":
        raise ValueError("Dependent-noise generator currently supports Gaussian noise only.")
    if df is not None:
        raise ValueError("df is not used for dependent Gaussian noise.")

    noise_list = _validate_regression_inputs(
        n_features=n_features,
        n_targets=n_targets,
        n_informative=n_informative,
        noise_list=noise_list,
        coef=coef,
    )
    corr = _dependent_gaussian_correlation(
        n_targets=n_targets,
        correlation=correlation,
        correlation_structure=correlation_structure,
    )
    covariance = np.outer(noise_list, noise_list) * corr

    rng = np.random.default_rng(random_state)
    noise = rng.multivariate_normal(
        mean=np.zeros(n_targets),
        cov=covariance,
        size=n_samples,
    )
    y = np.zeros((n_samples, n_targets))

    for i, (X, coef_i, coef_list) in enumerate(_make_design_and_coefficients(
        rng=rng,
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        n_informative=n_informative,
        coef=coef,
    )):
        y[:, i] = X @ coef_i + noise[:, i]

    if coef is not None:
        return X, y
    return X, y, coef_list


# Alias with "multi_target" spelling for easier discovery from reviewer notes.
make_multi_target_regression_dependent_noise = make_multitarget_regression_dependent_noise

