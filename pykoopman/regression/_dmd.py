"""module for dmd"""
# from warnings import warn
from __future__ import annotations

import numpy as np
from pydmd import DMDBase
from pydmd.dmdbase import DMDTimeDict
from pydmd.utils import compute_svd
from pydmd.utils import compute_tlsq
from scipy.linalg import eig
from scipy.linalg import sqrtm
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class PyDMDRegressor(BaseRegressor):
    """
    PyDMDRegressor is a wrapper for `pydmd` regressors.

    This class provides a wrapper for the `pydmd` regressor. The details about
    `pydmd` can be found in the reference:

    Demo, N., Tezzele, M., & Rozza, G. (2018). PyDMD: Python dynamic mode decomposition.
    Journal of Open Source Software, 3(22), 530.
    <https://joss.theoj.org/papers/10.21105/joss.00530.pdf>`_

    Args:
        regressor (DMDBase): A regressor instance from DMDBase in `pydmd`.
        tikhonov_regularization (bool or None, optional): Indicates if Tikhonov
        regularization should be applied. Defaults to None.

    Attributes:
        tlsq_rank (int): Rank for truncation in TLSQ method. If 0, no noise reduction
            is computed. If positive, it is used for SVD truncation.
        svd_rank (int): Rank for truncation. If 0, optimal rank is computed and used
            for truncation. If positive integer, it is used for truncation. If float
            between 0 and 1, the rank is the number of the biggest singular values
            that are needed to reach the 'energy' specified by `svd_rank`. If -1, no
            truncation is computed.
        forward_backward (bool): If True, the low-rank operator is computed like in
            fbDMD.
        tikhonov_regularization (bool or None, optional): If None, no regularization
            is applied. If float, it is used as the Tikhonov regularization parameter.
        flag_xy (bool): If True, the regressor is operating on multiple trajectories
            instead of just one.
        n_samples_ (int): Number of samples.
        n_input_features_ (int): Number of features, i.e., the dimension of phi.
        _unnormalized_modes (ndarray): Raw DMD V with each column as one DMD mode.
        _state_matrix_ (ndarray): DMD state transition matrix.
        _reduced_state_matrix_ (ndarray): Reduced DMD state transition matrix.
        _eigenvalues_ (ndarray): Identified Koopman lambda.
        _eigenvectors_ (ndarray): Identified Koopman eigenvectors.
        _coef_ (ndarray): Weight vectors of the regression problem. Corresponds to
            either [A] or [A,B].
        C (ndarray): Matrix that maps psi to the input features.
    """

    def __init__(self, regressor, tikhonov_regularization=None):
        """
        Initializes a PyDMDRegressor instance.

        Args:
            regressor (DMDBase): A regressor instance from DMDBase in `pydmd`.
            tikhonov_regularization (bool or None, optional): Indicates if Tikhonov
                regularization should be applied. Defaults to None.

        Raises:
            ValueError: If regressor is not a subclass of DMDBase from pydmd.
        """
        if not isinstance(regressor, DMDBase):
            raise ValueError("regressor must be a subclass of DMDBase from pydmd.")
        self.regressor = regressor
        # super(PyDMDRegressor, self).__init__(regressor)
        self.tlsq_rank = regressor._tlsq_rank
        self.svd_rank = regressor._Atilde._svd_rank
        self.forward_backward = regressor._Atilde._forward_backward
        self.tikhonov_regularization = tikhonov_regularization
        self.flag_xy = False
        self._ur = None

    def fit(self, x, y=None, dt=1):
        """
        Fit the PyDMDRegressor model according to the given training data.

        Args:
            x (np.ndarray): Measurement data input. Should be of shape (n_samples,
                n_features).
            y (np.ndarray, optional): Measurement data output to be fitted. Should be
                of shape (n_samples, n_features). Defaults to None.
            dt (float, optional): Time interval between `x` and `y`. Defaults to 1.

        Returns:
            self : Returns the instance itself.
        """

        self.n_samples_, self.n_input_features_ = x.shape

        if y is None:
            # single trajectory
            self.flag_xy = False
            X = x[:-1].T
            Y = x[1:].T
        else:
            # multiple segments of trajectories
            self.flag_xy = True
            X = x.T
            Y = y.T

        X, Y = compute_tlsq(X, Y, self.tlsq_rank)
        U, s, V = compute_svd(X, self.svd_rank)

        if self.tikhonov_regularization is not None:
            _norm_X = np.linalg.norm(X)
        else:
            _norm_X = 0

        atilde = self._least_square_operator(
            U, s, V, Y, self.tikhonov_regularization, _norm_X
        )
        if self.forward_backward:
            # b stands for "backward"
            bU, bs, bV = compute_svd(Y, svd_rank=len(s))
            atilde_back = self._least_square_operator(
                bU, bs, bV, X, self.tikhonov_regularization, _norm_X
            )
            atilde = sqrtm(atilde @ np.linalg.inv(atilde_back))

        # - V, lamda, eigenvectors
        self._coef_ = atilde
        self._state_matrix_ = atilde
        [self._eigenvalues_, self._eigenvectors_] = eig(self._state_matrix_)

        # self._coef_ = U @ atilde @ U.conj().T
        # self._state_matrix_ = self._coef_
        # self._reduced_state_matrix_ = atilde
        # [self._eigenvalues_, self._eigenvectors_] = eig(self._reduced_state_matrix_)
        self._ur = U
        self._unnormalized_modes = self._ur @ self._eigenvectors_

        self._tmp_compute_psi = np.linalg.pinv(self.unnormalized_modes)

        # self.C = np.linalg.inv(self._eigenvectors_) @ U.conj().T
        # self._modes_ = U.dot(self._eigenvectors_)

        return self

    def predict(self, x):
        """
        Predict the future values based on the input measurement data.

        Args:
            x (np.ndarray): Measurement data upon which to base the prediction.
                Should be of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values of `x` one timestep in the future. The shape
                is (n_samples, n_features).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        check_is_fitted(self, "coef_")
        y = np.linalg.multi_dot([self.ur, self._coef_, self.ur.conj().T, x.T]).T
        return y

    def _compute_phi(self, x_col):
        """
        Compute the `phi(x)` value given `x`.

        Args:
            x_col (np.ndarray): Input data, if one-dimensional it will be reshaped
                to (-1, 1).

        Returns:
            np.ndarray: Computed `phi(x)` value.
        """
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        phi = self.ur.T @ x_col
        return phi

    def _compute_psi(self, x_col):
        """
        Compute the `psi(x)` value given `x`.

        Args:
            x_col (np.ndarray): Input data, if one-dimensional it will be reshaped
                to (-1, 1).

        Returns:
            np.ndarray: Value of Koopman eigenfunction psi at x.
        """

        # compute psi - one column if x is a row
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        psi = self._tmp_compute_psi @ x_col
        return psi

    def _set_initial_time_dictionary(self, time_dict):
        """
        Sets the initial values for `time_dict` and `original_time`.
        Typically called in `fit()` and not used again afterwards.

        Args:
            time_dict (dict): Initial time dictionary for this DMD instance. Must
                contain the keys "t0", "tend", and "dt".

        Raises:
            ValueError: If the time_dict does not contain the keys "t0", "tend" and
                "dt" or if it contains more than these keys.
        """

        if not ("t0" in time_dict and "tend" in time_dict and "dt" in time_dict):
            raise ValueError(
                'time_dict must contain the keys "t0", ' '"tend" and "dt".'
            )
        if len(time_dict) > 3:
            raise ValueError(
                'time_dict must contain only the keys "t0", ' '"tend" and "dt".'
            )

        self._original_time = DMDTimeDict(dict(time_dict))
        self._dmd_time = DMDTimeDict(dict(time_dict))

    def _least_square_operator(self, U, s, V, Y, tikhonov_regularization, _norm_X):
        """
        Calculates the least square estimation 'A' using the provided parameters.

        Args:
            U (numpy.ndarray): Left singular vectors, shape (n_features, svd_rank).
            s (numpy.ndarray): Singular values, shape (svd_rank, ).
            V (numpy.ndarray): Right singular vectors, shape (n_features, svd_rank).
            Y (numpy.ndarray): Measurement data for prediction, shape (n_samples,
                n_features).
            tikhonov_regularization (bool or NoneType): Tikhonov parameter for
                regularization. If `None`, no regularization is applied, if `float`,
                it is used as the :math:`\\lambda` tikhonov parameter.
            _norm_X (numpy.ndarray): Norm of `X` for Tikhonov regularization, shape
                (n_samples, n_features).

        Returns:
            numpy.ndarray: The least square estimation 'A', shape (svd_rank, svd_rank).
        """

        if tikhonov_regularization is not None:
            s = (s**2 + tikhonov_regularization * _norm_X) * np.reciprocal(s)
        A = np.linalg.multi_dot([U.T.conj(), Y, V]) * np.reciprocal(s)
        return A

    @property
    def coef_(self):
        """
        The weight vectors of the regression problem.

        This method checks if the regressor is fitted before returning the coefficient.

        Returns:
            numpy.ndarray: The coefficient matrix.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        """
        The DMD state transition matrix.

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
        The raw DMD V with each column as one DMD mode.

        This method checks if the regressor is fitted before returning the unnormalized
            modes.

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
