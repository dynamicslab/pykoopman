# from warnings import warn
# import numpy as np
# import scipy
# import tensorflow as tf
# from sklearn.utils.validation import check_is_fitted
# todo: write a lightweight Koopman package in pytorch
from __future__ import annotations


class NNDMD(object):
    def __init__(self):
        pass

    def fit(self, x, y=None, dt=None):
        pass

    def predict(self, x):
        pass

    @property
    def coef_(self):
        pass

    @property
    def state_matrix_(self):
        pass

    @property
    def eigenvalues_(self):
        pass

    @property
    def unnormalized_modes(self):
        pass

    def compute_eigen_phi(self, x):
        pass
