# from warnings import warn
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
    Wrapper for PyDMD regressors.

       x' = Ax
       A = self.coef_

    Note that
       Ur^T A_tilde U_r = A

    But we will not use A_tilde in any case
    Parameters
    ----------
    DMDRegressor: DMDBase subclass
        Regressor from PyDMD. Must extend the DMDBase class.
    """

    def __init__(self, regressor, tikhonov_regularization=None):
        if not isinstance(regressor, DMDBase):
            raise ValueError("regressor must be a subclass of DMDBase from pydmd.")
        super(PyDMDRegressor, self).__init__(regressor)
        self.tlsq_rank = regressor.tlsq_rank
        self.svd_rank = regressor.svd_rank
        self.forward_backward = regressor.forward_backward
        self.tikhonov_regularization = tikhonov_regularization

        self.flag_xy = False

    def fit(self, x, y=None, dt=1):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_examples, n_features)
            Measurement data to be fit.

        Returns
        -------
        self: returns a fit ``DMDRegressor`` instance
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

        # - modes, eigenvalues, eigenvectors
        self._coef_ = U @ atilde @ U.conj().T
        self._state_matrix_ = U @ atilde @ U.conj().T
        self._reduced_state_matrix_ = atilde
        [self._eigenvalues_, self._eigenvectors_] = eig(self._reduced_state_matrix_)
        self._unnormalized_modes = U @ self._eigenvectors_
        self.C = np.linalg.inv(self._eigenvectors_) @ U.conj().T
        # self._modes_ = U.dot(self._eigenvectors_)

        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_examples, n_features)
            Measurement data upon which to base prediction.

        Returns
        -------
        y: numpy ndarray, shape (n_examples, n_features)
            Prediction of x one timestep in the future.

        """
        check_is_fitted(self, "coef_")
        return np.linalg.multi_dot([self._coef_, x.T]).T

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    @property
    def reduced_state_matrix_(self):
        check_is_fitted(self, "_reduced_state_matrix_")
        return self._reduced_state_matrix_

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def unnormalized_modes(self):
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    def compute_eigen_phi(self, x):
        """
        input data x is a row-wise data
        """
        # compute eigenfunction - one column if x is a row
        return self.C @ x.T

    def _set_initial_time_dictionary(self, time_dict):
        """
        Set the initial values for the class fields `time_dict` and
        `original_time`. This is usually called in `fit()` and never again.

        :param time_dict: Initial time dictionary for this DMD instance.
        :type time_dict: dict
        """
        if not ("t0" in time_dict and "tend" in time_dict and "dt" in time_dict):
            raise ValueError('time_dict must contain the keys "t0", "tend" and "dt".')
        if len(time_dict) > 3:
            raise ValueError(
                'time_dict must contain only the keys "t0", "tend" and "dt".'
            )

        self._original_time = DMDTimeDict(dict(time_dict))
        self._dmd_time = DMDTimeDict(dict(time_dict))

    def _least_square_operator(self, U, s, V, Y, tikhonov_regularization, _norm_X):
        if tikhonov_regularization is not None:
            s = (s**2 + tikhonov_regularization * _norm_X) * np.reciprocal(s)
        return np.linalg.multi_dot([U.T.conj(), Y, V]) * np.reciprocal(s)
