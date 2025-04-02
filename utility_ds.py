import numpy as np
import math
from data_generator import calibration_split
from itertools import combinations, product
from rectangle import Rectangle

def data_splitting_prediction_region(scores, alpha = 0.2):

    # Number of samples, number of dimensions
    n = scores.shape[0]
    d = scores.shape[1]

    