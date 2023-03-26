"""module for base class of regressor"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from sklearn.base import BaseEstimator


class BaseRegressor(BaseEstimator, ABC):
    """
    Wrapper class for PyKoopman regressors.

    This class is inherited from `sklearn.base.BaseEstimator`

    Parameters
    ----------
    regressor : sklearn.base.BaseEstimator
        A regressor object implementing ``fit`` and ``predict`` methods.

    Attributes
    ----------
    regressor : sklearn.base.BaseEstimator
        A regressor object implementing ``fit`` and ``predict`` methods.
    """

    def __init__(self, regressor):
        # check .fit
        if not hasattr(regressor, "fit") or not callable(getattr(regressor, "fit")):
            raise AttributeError("regressor does not have a callable fit method")
        # check .predict
        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            raise AttributeError("regressor does not have a callable predict method")
        self.regressor = regressor

    def fit(self, x, y=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    @abstractmethod
    def coef_(self):
        pass

    @abstractmethod
    def state_matrix_(self):
        pass

    @abstractmethod
    def eigenvectors_(self):
        pass

    @abstractmethod
    def eigenvalues_(self):
        pass

    @abstractmethod
    def _compute_phi(self, x):
        pass

    @abstractmethod
    def _compute_psi(self, x):
        pass

    @abstractmethod
    def ur(self):
        pass

    @abstractmethod
    def unnormalized_modes(self):
        pass
