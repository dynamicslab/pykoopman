"""module for base class of regressor"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np
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

    def _detect_reshape(self, X, offset=True):
        """
        Detect the shape of the input data and reshape it accordingly to return
        both X and Y in the correct shape.
        """
        s1 = -1 if offset else None
        s2 = 1 if offset else None
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            if X.ndim == 2:
                self.n_samples_, self.n_input_features_ = X.shape
                self.n_trials_ = 1
                return X[:s1], X[s2:]
            elif X.ndim == 3:
                self.n_trials_, self.n_samples_, self.n_input_features_ = X.shape
                X, Y = X[:, :s1, :], X[:, s2:, :]
                return X.reshape(-1, X.shape[2]), Y.reshape(
                    -1, Y.shape[2]
                )  # time*trials, features

        elif isinstance(X, list):
            assert all(isinstance(x, np.ndarray) for x in X)
            self.n_trials_tot, self.n_samples_tot, self.n_input_features_tot = (
                [],
                [],
                [],
            )
            X_tot, Y_tot = [], []
            for x in X:
                x, y = self._detect_reshape(x)
                X_tot.append(x)
                Y_tot.append(y)
                self.n_trials_tot.append(self.n_trials_)
                self.n_samples_tot.append(self.n_samples_)
                self.n_input_features_tot.append(self.n_input_features_)
            X = np.concatenate(X_tot, axis=0)
            Y = np.concatenate(Y_tot, axis=0)

            self.n_trials_ = sum(self.n_trials_tot)
            self.n_samples_ = sum(self.n_samples_tot)
            self.n_input_features_ = sum(self.n_input_features_tot)

            return X, Y

    def _return_orig_shape(self, X):
        """
        X will be a 2d array of shape (n_samples * n_trials, n_features).
        This function will return the original shape of X.
        """
        if not hasattr(self, "n_trials_tot"):
            X = X.reshape(self.n_trials_, -1, self.n_input_features_)
            if X.shape[0] == 1:
                X = X[0]
            return X

        else:
            X_tot = []
            prev_t = 0
            for i in range(len(self.n_trials_tot)):
                X_i = X[prev_t : prev_t + self.n_trials_tot[i] * self.n_samples_tot[i]]
                X_i = X_i.reshape(
                    self.n_trials_tot[i], -1, self.n_input_features_tot[i]
                )
                X_tot.append(X_i)
                prev_t += self.n_trials_tot[i] * self.n_samples_tot[i]
            return X_tot

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
