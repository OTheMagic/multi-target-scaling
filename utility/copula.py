import numpy as np
from scipy.stats import rankdata
from utility.rectangle import Rectangle
from sklearn.model_selection import train_test_split

class EmpiricalCopula:
    def __init__(self):
        self.U = None  # Pseudo-observations
        self.n = 0
        self.d = 0

    def fit(self, X):
        """
        X: ndarray of shape (n_samples, n_features)
        """
        self.n, self.d = X.shape
        self.U = self._to_uniform(X)

    def _to_uniform(self, X):
        """
        ECDF transform: each column of X to pseudo-observations in [0,1]
        """
        U = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            ranks = rankdata(X[:, j], method='ordinal')
            U[:, j] = ranks / (self.n + 1)
        return U

    def cdf(self, u):
        """
        Evaluate empirical copula C_n at u in [0,1]^d
        """
        u = np.asarray(u)
        assert u.shape[0] == self.d, "Dimensionality mismatch"
        return np.mean(np.all(self.U <= u, axis=1))

    def quantile_box(self, alpha):
        """
        Return axis-aligned threshold box [0, tau_1] x ... x [0, tau_d]
        such that at least (1 - alpha) of points fall inside
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
    n, d = scores.shape
    upper = np.zeros(d)
    for j in range(d):
        scores_sorted = np.sort(scores[:, j])
        idx = int(np.ceil(U_thresholds[j] * (n+1)))
        upper[j] = scores_sorted[idx-1] if idx <= n else np.inf
    return upper

def empirical_copula_prediction(scores, alpha = 0.2, random_state = 42):

    #scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)
    cop = EmpiricalCopula()
    cop.fit(scores)
    thresholds = cop.quantile_box(alpha=alpha)
    return Rectangle(upper=inverse_ecdf_transform(thresholds, scores))
