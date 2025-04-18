import numpy as np
from scipy.stats import rankdata

def ecdf_transform(scores):
    U = np.zeros_like(scores)
    for j in range(scores.shape[0]):
        ranks = rankdata(scores[:, j], method="max")
        U[:, j] = ranks/(scores.shape[0]+1)
    return U

def empricial_copula_prediction_region(scores, alpha = 0.2):
    U = ecdf_transform(scores=scores)


def vine_copula_prediction_region(scores, alpha = 0.2):
    pass
