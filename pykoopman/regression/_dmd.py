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
    Wrapper for `pydmd` regressors.

    See the following reference for more details on `pydmd`

        Demo, N., Tezzele, M., & Rozza, G. (2018). PyDMD: Python
        dynamic mode decomposition. Journal of Open Source
        Software, 3(22), 530.
        <https://joss.theoj.org/papers/10.21105/joss.00530.pdf>`_

    Parameters
    ----------
    regressor:
        A regressor instance from DMDBase in `pydmd`

    tikhonov_regularization: bool or NoneType
        Whether or not to choose tikhonov regularization

    Attributes
    ----------
    tlsq_rank : int
        the rank for the truncation; If 0, the method
        does not compute any noise reduction; if positive number, the
        method uses the argument for the SVD truncation used in the TLSQ
        method.

    svd_rank : int
        the rank for the truncation; If 0, the method computes
        the optimal rank and uses it for truncation; if positive interger,
        the method uses the argument for the truncation; if float between 0
        and 1, the rank is the number of the biggest singular values that
        are needed to reach the 'energy' specified by `svd_rank`; if -1,
        the method does not compute truncation. Default is 0.

    forward_backward : bool
        If `True`, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.

    tikhonov_regularization : bool or NoneType, default=None
        Tikhonov parameter for the regularization.
        If `None`, no regularization is applied, if `float`, it is used as the
        :math:`\\lambda` tikhonov parameter.

    flag_xy : bool
        If `True`, the regressor is operating on multiple trajectories instead
        of just one.

    n_samples_ : int
        Number of samples

    n_input_features_ : int
        Number of features, i.e., the dimension of :math:`\\phi`

    _unnormalized_modes : numpy.ndarray, shape (n_input_features_, svd_rank)
        Raw DMD V with each column as one DMD mode

    _state_matrix_ : numpy.ndarray, shape (n_input_features_, n_input_features_)
        DMD state transition matrix

    _reduced_state_matrix_ : numpy.ndarray, shape (svd_rank, svd_rank)
        Reduced DMD state transition matrix

    _eigenvalues_ : numpy.ndarray, shape (n_input_features_,)
        Identified Koopman lamda

    _eigenvectors_ : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Identified Koopman eigenvectors

    _coef_ : numpy.ndarray, shape (n_input_features_, n_input_features_) or
        (n_input_features_, n_input_features_ + n_control_features_)
        Weight vectors of the regression problem. Corresponds to either [A] or [A,B]

    C : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Matrix that maps psi to the input features

    """

    def __init__(self, regressor, tikhonov_regularization=None):
        if not isinstance(regressor, DMDBase):
            raise ValueError("regressor must be a subclass of DMDBase from pydmd.")
        self.regressor = regressor
        # super(PyDMDRegressor, self).__init__(regressor)
        self.tlsq_rank = regressor.tlsq_rank
        self.svd_rank = regressor.svd_rank
        self.forward_backward = regressor.forward_backward
        self.tikhonov_regularization = tikhonov_regularization
        self.flag_xy = False
        self._ur = None

    def fit(self, x, y=None, dt=1):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
            Measurement data input

        y: numpy ndarray, shape (n_samples, n_features), default=None
            Measurement data output to be fitted

        dt : float
            Time interval between `x` and `y`

        Returns
        -------
        self : PyDMDRegressor
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
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
            Measurement data upon which to base prediction.

        Returns
        -------
        y: numpy ndarray, shape (n_samples, n_features)
            Prediction of x one timestep in the future.

        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        check_is_fitted(self, "coef_")
        y = np.linalg.multi_dot([self.ur, self._coef_, self.ur.conj().T, x.T]).T
        return y

    def _compute_phi(self, x):
        """Returns `pji(x)` given `x`"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        phi = self.ur.T @ x.T
        return phi

    def _compute_psi(self, x):
        """Returns `psi(x)` given `x`

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to compute psi values.

        Returns
        -------
        psi : numpy.ndarray, shape (n_samples, n_input_features_)
            value of Koopman eigenfunction psi at x
        """

        # compute psi - one column if x is a row
        if x.ndim == 1:
            x = x.reshape(1, -1)
        psi = self._tmp_compute_psi @ x.T
        return psi

    def _set_initial_time_dictionary(self, time_dict):
        """
        Set the initial values for the class fields `time_dict` and
        `original_time`. This is usually called in `fit()` and never again.

        Parameters
        ----------
        time_dict : dict
            Initial time dictionary for this DMD instance.
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
        Parameters
        ----------
        U : numpy.ndarray, shape (n_features, svd_rank)
            Left singular vectors

        s : numpy.ndarray, shape (svd_rank, )
            Singular values

        V : numpy.ndarray, shape (n_features, svd_rank)
            Right singular vectors

        Y : numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to base prediction

        tikhonov_regularization : bool or NoneType
            Tikhonov parameter for the regularization.
            If `None`, no regularization is applied, if `float`, it is used as the
            :math:`\\lambda` tikhonov parameter.

        _norm_X : numpy.ndarray, shape (n_samples, n_features)
            Norm of `X` for Tikhonov regularization

        Returns
        -------
        A : numpy.ndarray, shape (svd_rank, svd_rank)
            the least square estimation
        """

        if tikhonov_regularization is not None:
            s = (s**2 + tikhonov_regularization * _norm_X) * np.reciprocal(s)
        A = np.linalg.multi_dot([U.T.conj(), Y, V]) * np.reciprocal(s)
        return A

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    # @property
    # def reduced_state_matrix_(self):
    #     check_is_fitted(self, "_reduced_state_matrix_")
    #     return self._reduced_state_matrix_

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
