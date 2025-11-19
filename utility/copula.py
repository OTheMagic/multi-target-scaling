import numpy as np
from scipy.stats import rankdata
from utility.rectangle import Rectangle
from sklearn.model_selection import train_test_split


class EmpiricalCopula:
    """
    Empirical copula estimator based on pseudo-observations in [0, 1]^d.

    Attributes
    ----------
    U : np.ndarray or None
        Pseudo-observations of shape (n_samples, d) after rank transform.
    n : int
        Number of samples.
    d : int
        Number of dimensions (features).
    """
    def __init__(self):
        self.U = None  # Pseudo-observations
        self.n = 0
        self.d = 0

    def fit(self, X):
        """
        Fit the empirical copula using data X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data from which to construct pseudo-observations.

        Returns
        -------
        None
        """
        self.n, self.d = X.shape
        self.U = self._to_uniform(X)

    def _to_uniform(self, X):
        """
        Transform each column of X to pseudo-observations in [0, 1]
        using the empirical CDF (rank-based transform).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray, shape (n_samples, n_features)
            Pseudo-observations U, where each column has been mapped to
            approximate uniforms in (0, 1) via ranks.
        """
        U = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            ranks = rankdata(X[:, j], method='ordinal')
            U[:, j] = ranks / (self.n + 1)
        return U

    def cdf(self, u):
        """
        Evaluate the empirical copula C_n at a point u in [0, 1]^d.

        Parameters
        ----------
        u : array-like, shape (d,)
            Point in the unit hypercube at which to evaluate the copula C_n.

        Returns
        -------
        float
            Empirical copula value C_n(u) = (1/n) sum_{i=1}^n 1{U_i <= u}
            where the inequality is component-wise.
        """
        u = np.asarray(u)
        assert u.shape[0] == self.d, "Dimensionality mismatch"
        return np.mean(np.all(self.U <= u, axis=1))

    def quantile_box(self, alpha):
        """
        Compute an axis-aligned threshold box [0, tau_1] x ... x [0, tau_d]
        such that at least (1 - alpha) of the pseudo-observations fall inside.

        This searches over candidate thresholds taken from the sorted
        pseudo-observations and returns the first one whose empirical
        copula value exceeds 1 - alpha. If none is found, returns a
        box with tau_j = 1 for all j.

        Parameters
        ----------
        alpha : float
            Miscoverage level in (0, 1). The target is coverage >= 1 - alpha
            under the empirical copula.

        Returns
        -------
        np.ndarray, shape (d,)
            Vector of upper thresholds (tau_1, ..., tau_d) in [0, 1].
        """
        count = 0
        for i in range(self.n):
            sorted_vals = np.sort(self.U, axis=0, kind="mergesort") 
            if self.cdf(sorted_vals[i]) >= 1-alpha:
                return sorted_vals[i]
            count += 1
        if count == self.n:
            return np.repeat(1, self.d)
    

def inverse_ecdf_transform(U_thresholds, scores):
    """
    Map copula (uniform) thresholds back to the original score scale
    via coordinate-wise empirical CDF inversion.

    Parameters
    ----------
    U_thresholds : np.ndarray, shape (d,)
        Thresholds in [0, 1] for each coordinate, typically coming
        from an empirical copula box.
    scores : np.ndarray, shape (n, d)
        Original scores or residuals used to define the empirical margins.

    Returns
    -------
    np.ndarray, shape (d,)
        Upper bounds on the original score scale corresponding to the
        copula thresholds U_thresholds. If the rank index exceeds n,
        the bound for that coordinate is set to +inf.
    """
    n, d = scores.shape
    upper = np.zeros(d)
    for j in range(d):
        scores_sorted = np.sort(scores[:, j])
        idx = int(np.ceil(U_thresholds[j] * (n+1)))
        upper[j] = scores_sorted[idx-1] if idx <= n else np.inf
    return upper


def empirical_copula_prediction(scores, alpha = 0.2, random_state = 42):
    """
    Construct a rectangular prediction region via an empirical copula
    approach on the scores.

    The procedure:
    1. Fit an empirical copula on the scores.
    2. Find a box in copula space [0, tau_1] x ... x [0, tau_d] with
       empirical coverage at least 1 - alpha.
    3. Map these copula thresholds back to the original score scale
       using the marginal empirical CDFs (inverse_ecdf_transform).

    Parameters
    ----------
    scores : np.ndarray, shape (n, d)
        Calibration scores or residuals used to build the empirical copula.
    alpha : float, default=0.2
        Miscoverage level in (0, 1). Target coverage is at least 1 - alpha.
    random_state : int, default=42
        Currently unused (placeholder for potential future data splitting).

    Returns
    -------
    Rectangle
        A Rectangle object with upper bounds defined by the inverse
        empirical CDF transform of the copula quantile box.
    """
    #scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)
    cop = EmpiricalCopula()
    cop.fit(scores)
    thresholds = cop.quantile_box(alpha=alpha)
    return Rectangle(upper=inverse_ecdf_transform(thresholds, scores))
