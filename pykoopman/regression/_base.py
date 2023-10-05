"""module for base class of regressor"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from sklearn.base import BaseEstimator


class BaseRegressor(BaseEstimator, ABC):
    """
    Base class for PyKoopman regressors.

    This class provides a wrapper for regressors used in the PyKoopman package.
    It's designed to be used with any regressor object that implements `fit`
    and `predict` methods following the `sklearn.base.BaseEstimator` interface.

    Note: This is an abstract base class, and should not be instantiated directly.
    Instead, a subclass should be created that implements the required abstract methods.

    Args:
        regressor (BaseEstimator): A regressor object implementing `fit` and `predict`
        methods.

    Attributes:
        regressor (BaseEstimator): The regressor object passed during initialization.

    Abstract methods:
        coef_ : Should return the coefficients of the regression model.

        state_matrix_ : Should return the state matrix of the dynamic system.

        eigenvectors_ : Should return the eigenvectors of the system.

        eigenvalues_ : Should return the eigenvalues of the system.

        _compute_phi(x_col) : Should compute and return the phi function on given data.

        _compute_psi(x_col) : Should compute and return the psi function on given data.

        ur : Should return the u_r of the system.

        unnormalized_modes : Should return the unnormalized modes of the system.
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
    def _compute_phi(self, x_col):
        pass

    @abstractmethod
    def _compute_psi(self, x_col):
        pass

    @abstractmethod
    def ur(self):
        pass

    @abstractmethod
    def unnormalized_modes(self):
        pass
