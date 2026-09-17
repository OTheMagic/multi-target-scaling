"""Audited methods and additional baselines; checks are in README.md.

Imports perform no experiments or data loading.
"""

from .unscaled import unscaled_prediction
from .ec import EmpiricalCopula, empirical_copula_prediction, inverse_ecdf_transform
from .point_chr import data_spliting_CHR_prediction
from .cqhr import CQHRResult, cqhr_adjustments
from .gwc import gwc_prediction
from .envelope import EnvelopeCalibration, EnvelopeRegion, envelope_prediction
from .additional import (
    bonferroni_prediction, naive_prediction, data_splitting_oracle_prediction,
    data_splitting_standardized_prediction,
)

__all__ = [
    "unscaled_prediction",
    "EmpiricalCopula", "empirical_copula_prediction", "inverse_ecdf_transform",
    "data_spliting_CHR_prediction",
    "CQHRResult", "cqhr_adjustments",
    "gwc_prediction",
    "EnvelopeCalibration", "EnvelopeRegion", "envelope_prediction",
    "bonferroni_prediction", "naive_prediction",
    "data_splitting_oracle_prediction", "data_splitting_standardized_prediction",
]
