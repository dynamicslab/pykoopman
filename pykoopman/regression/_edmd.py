# from warnings import warn
from __future__ import annotations

import numpy as np
import scipy
from pydmd.dmdbase import DMDTimeDict
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class EDMD(BaseRegressor):
    """Extended DMD (EDMD) regressor.

    Aims to determine the system matrices A,C
    that satisfy y' = Ay and x = Cy, where y' is the time-shifted
    observable with y0 = phi(x0). C is the measurement matrix that maps back to the
    state.

    The objective functions,
    :math:`\\|Y'-AY\\|_F`,
    are minimized using least-squares regression and singular value
    decomposition.

    See the following reference for more details:
        `M.O. Williams , I.G. Kevrekidis, C.W. Rowley
        "A Data–Driven Approximation of the Koopman Operator:
        Extending Dynamic Mode Decomposition."
        Journal of Nonlinear Science, Vol. 25, 1307-1346, 2015.
        <https://link.springer.com/article/10.1007/s00332-015-9258-5>`_

    Attributes
    ----------
    _coef_ : numpy.ndarray, shape (n_input_features_, n_input_features_) or
        (n_input_features_, n_input_features_ + n_control_features_)
        Weight vectors of the regression problem. Corresponds to either [A] or [A,B]

    _state_matrix_ : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Identified state transition matrix A of the underlying system.

    _eigenvalues_ : numpy.ndarray, shape (n_input_features_,)
        Identified Koopman lamda

    eigenvectors_ : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Identified Koopman eigenvectors

    _unnormalized_modes_ : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Identified Koopman eigenvectors

    n_samples_ : int
        Number of samples

    n_input_features_ : int
        Number of input features

    C : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Matrix that maps psi to the input features
    """

    def __init__(self):
        pass

    def fit(self, x, y=None, dt=None):
        """
        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
            Measurement data to be fit.

        y : numpy.ndarray, shape (n_samples, n_features)
            Time-shifted measurement data to be fit

        dt : scalar
            Discrete time-step

        Returns
        -------
        self: returns a fitted ``EDMD`` instance
        """
        self.n_samples_, self.n_input_features_ = x.shape

        if y is None:
            X1 = x[:-1, :]
            X2 = x[1:, :]
        else:
            X1 = x
            X2 = y

        # X1, X2 are row-wise data, so there is a transpose in the end.
        self._coef_ = np.linalg.lstsq(X1, X2)[0].T  # [0:Nlift, 0:Nlift]
        self._state_matrix_ = self._coef_
        [self._eigenvalues_, self._eigenvectors_] = scipy.linalg.eig(self.state_matrix_)
        self._unnormalized_modes = self._eigenvectors_
        self._ur = np.eye(self.n_input_features_)
        self._tmp_compute_psi = np.linalg.inv(self._eigenvectors_)

        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to base prediction.

        Returns
        -------
        y: numpy.ndarray, shape (n_samples, n_features)
            Prediction of x one timestep in the future.

        """
        check_is_fitted(self, "coef_")
        y = x @ self.state_matrix_.T
        return y

    def _compute_phi(self, x):
        """Returns `phi(x)` given `x`"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        phi = self._ur.T @ x.T
        return phi

    def _compute_psi(self, x):
        """Returns `psi(x)` given `x`

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to compute psi values.

        Returns
        -------
        phi : numpy.ndarray, shape (n_samples, n_input_features_)
            value of Koopman psi at x
        """
        # compute psi - one column if x is a row
        if x.ndim == 1:
            x = x.reshape(1, -1)
        psi = self._tmp_compute_psi @ x.T
        return psi

    def _set_initial_time_dictionary(self, time_dict):
        """Set the initial values for the class fields `time_dict` and
        `original_time`. This is usually called in `fit()` and never again.

        Parameters
        ----------
        time_dict : dict
            Initial time dictionary for this DMD instance.
        """
        if not ("t0" in time_dict and "tend" in time_dict and "dt" in time_dict):
            raise ValueError('time_dict must contain the keys "t0", "tend" and "dt".')
        if len(time_dict) > 3:
            raise ValueError(
                'time_dict must contain only the keys "t0", "tend" and "dt".'
            )

        self._original_time = DMDTimeDict(dict(time_dict))
        self._dmd_time = DMDTimeDict(dict(time_dict))

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def eigenvectors_(self):
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def unnormalized_modes(self):
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        check_is_fitted(self, "_ur")
        return self._ur
