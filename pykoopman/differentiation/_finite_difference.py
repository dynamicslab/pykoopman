import numpy as np
from sklearn.base import BaseEstimator


class FiniteDifference(BaseEstimator):
    def __init__(self, order=1):
        self.order = order

    def __call__(self, x, t=1):
        return np.gradient(x)
