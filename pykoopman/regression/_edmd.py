"""module for extended dmd"""
# from warnings import warn
from __future__ import annotations

import numpy as np
import scipy
from pydmd.dmdbase import DMDTimeDict
from pydmd.utils import compute_svd
from pydmd.utils import compute_tlsq
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class EDMD(BaseRegressor):
    """Extended DMD (EDMD) regressor.

    Aims to determine the system matrices A,C that satisfy y' = Ay and x = Cy,
    where y' is the time-shifted observable with y0 = phi(x0). C is the measurement
    matrix that maps back to the state.

    The objective functions, \\|Y'-AY\\|_F, are minimized using least-squares regression
    and singular value decomposition.

    See the following reference for more details:
        `M.O. Williams, I.G. Kevrekidis, C.W. Rowley
        "A Data–Driven Approximation of the Koopman Operator:
        Extending Dynamic Mode Decomposition."
        Journal of Nonlinear Science, Vol. 25, 1307-1346, 2015.
        <https://link.springer.com/article/10.1007/s00332-015-9258-5>`_

    Attributes:
        _coef_ (numpy.ndarray): Weight vectors of the regression problem. Corresponds
            to either [A] or [A,B].
        _state_matrix_ (numpy.ndarray): Identified state transition matrix A of the
            underlying system.
        _eigenvalues_ (numpy.ndarray): Identified Koopman lambda.
        _eigenvectors_ (numpy.ndarray): Identified Koopman eigenvectors.
        _unnormalized_modes_ (numpy.ndarray): Identified Koopman eigenvectors.
        n_samples_ (int): Number of samples.
        n_input_features_ (int): Number of input features.
        C (numpy.ndarray): Matrix that maps psi to the input features.
    """

    def __init__(self, svd_rank=1.0, tlsq_rank=0):
        """Initialize the EDMD regressor.

        Args:
            svd_rank (float): Rank parameter for singular value decomposition.
                Default is 1.0.
            tlsq_rank (int): Rank parameter for total least squares. Default is 0.
        """
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank

    def fit(self, x, y=None, dt=None):
        """Fit the EDMD regressor to the given data.

        Args:
            x (numpy.ndarray): Measurement data to be fit.
            y (numpy.ndarray, optional): Time-shifted measurement data to be fit.
                Defaults to None.
            dt (scalar, optional): Discrete time-step. Defaults to None.

        Returns:
            self: Fitted EDMD instance.
        """
        self.n_samples_, self.n_input_features_ = x.shape

        if y is None:
            X1 = x[:-1, :]
            X2 = x[1:, :]
        else:
            X1 = x
            X2 = y

        # perform SVD
        X1T, X2T = compute_tlsq(X1.T, X2.T, self.tlsq_rank)
        U, s, V = compute_svd(X1T, self.svd_rank)

        # X1, X2 are row-wise data, so there is a transpose in the end.
        self._coef_ = U.conj().T @ X2T @ V @ np.diag(np.reciprocal(s))
        # self._coef_ = np.linalg.lstsq(X1, X2)[0].T  # [0:Nlift, 0:Nlift]
        self._state_matrix_ = self._coef_
        [self._eigenvalues_, self._eigenvectors_] = scipy.linalg.eig(self.state_matrix_)
        # self._ur = np.eye(self.n_input_features_)
        self._ur = U
        # self._unnormalized_modes = self._eigenvectors_
        self._unnormalized_modes = self._ur @ self._eigenvectors_

        # np.linalg.pinv(self._unnormalized_modes)
        self._tmp_compute_psi = np.linalg.inv(self._eigenvectors_) @ self._ur.conj().T

        return self

    def predict(self, x):
        """Predict the next timestep based on the given data.

        Args:
            x (numpy.ndarray): Measurement data upon which to base prediction.

        Returns:
            y (numpy.ndarray): Prediction of x one timestep in the future.
        """
        check_is_fitted(self, "coef_")
        y = x @ self.ur.conj() @ self.state_matrix_.T @ self.ur.T
        return y

    def _compute_phi(self, x_col):
        """Compute phi(x) given x.

        Args:
            x_col (numpy.ndarray): Input data x.

        Returns:
            phi (numpy.ndarray): Value of phi(x).
        """
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        phi = self._ur.conj().T @ x_col
        return phi

    def _compute_psi(self, x_col):
        """Compute psi(x) given x.

        Args:
            x_col (numpy.ndarray): Input data x.

        Returns:
            psi (numpy.ndarray): Value of psi(x).
        """
        # compute psi - one column if x is a row
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        psi = self._tmp_compute_psi @ x_col
        return psi

    def _set_initial_time_dictionary(self, time_dict):
        """Set the initial values for the class fields time_dict and original_time.

        Args:
            time_dict (dict): Initial time dictionary for this DMD instance.
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
        """
        Weight vectors of the regression problem. Corresponds to either [A] or
        [A,B].

        """
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        """
        The EDMD state transition matrix.

        This method checks if the regressor is fitted before returning the state matrix.

        Returns:
            numpy.ndarray: The state transition matrix.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    @property
    def eigenvalues_(self):
        """
        The identified Koopman eigenvalues.

        This method checks if the regressor is fitted before returning the eigenvalues.

        Returns:
            numpy.ndarray: The Koopman eigenvalues.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def eigenvectors_(self):
        """
        The identified Koopman eigenvectors.

        This method checks if the regressor is fitted before returning the eigenvectors.

        Returns:
            numpy.ndarray: The Koopman eigenvectors.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def unnormalized_modes(self):
        """
        The raw EDMD V with each column as one EDMD mode.

        This method checks if the regressor is fitted before returning the unnormalized
            modes. Note that this will combined with the measurement matrix from the
            observer to give you the true Koopman modes

        Returns:
            numpy.ndarray: The unnormalized modes.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        """
        The left singular vectors 'U'.

        This method checks if the regressor is fitted before returning 'U'.

        Returns:
            numpy.ndarray: The left singular vectors 'U'.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_ur")
        return self._ur
