from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class EDMDc(BaseRegressor):
    """
    Extended DMD with control (EDMDc) regressor.

    Aims to determine the system matrices A,B,C
    that satisfy y' = Ay + Bu and x = Cy, where y' is the time-shifted
    observable with y0 = phi(x0) and u is the control input. B and C
    are the unknown control and measurement matrices, respectively.

    The objective functions,
    :math:`\\|Y'-AY-BU\\|_F`
    and
    :math:`\\|X-CY\\|_F`,
    are minimized using least-squares regression and singular value
    decomposition.

    See the following reference for more details:

        `Korda, M. and Mezic, I.
        "Linear predictors for nonlinear dynamical systems:
        Koopman operator meets model predictive control."
        Automatica, Vol. 93, 149–160.
        <https://www.sciencedirect.com/science/article/abs/pii/S000510981830133X>`_

    Parameters
    ----------

    Attributes
    ----------
    coef_ : numpy.ndarray, shape (n_input_features_, n_input_features_) or
        (n_input_features_, n_input_features_ + n_control_features_)
        Weight vectors of the regression problem. Corresponds to either [A] or [A,B]

    state_matrix_ : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Identified state transition matrix A of the underlying system.

    control_matrix_ : numpy.ndarray, shape (n_input_features_, n_control_features_)
        Identified control matrix B of the underlying system.

    projection_matrix_ : numpy.ndarray, shape (n_input_features_+
    n_control_features_, svd_rank)
        Projection matrix into low-dimensional subspace.

    projection_matrix_output_ : numpy.ndarray, shape (n_input_features_+
    n_control_features_, svd_output_rank)
        Projection matrix into low-dimensional subspace.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None, u=None, dt=None):
        """
        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_features)
            Measurement data to be fit.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        y: numpy.ndarray, shape (n_samples, n_features)
            Time-shifted measurement data to be fit

        dt: scalar
            Discrete time-step

        Returns
        -------
        self: returns a fitted ``EDMDc`` instance
        """
        self.n_samples_, self.n_input_features_ = x.shape
        if y is None:
            X1 = x[:-1, :]
            X2 = x[1:, :]
        else:
            X1 = x
            X2 = y

        if u.ndim == 1:
            if len(u) > X1.shape[0]:
                u = u[:-1]
            C = u[np.newaxis, :]
        else:
            if u.shape[0] > X1.shape[0]:
                u = u[:-1, :]
            C = u
        self.n_control_features_ = C.shape[1]

        self._fit_with_unknown_b(X1, X2, C)
        return self

    def _fit_with_unknown_b(self, X1, X2, U):
        """
        Parameters
        ----------
        X1: numpy.ndarray, shape (n_samples, n_features)
            Measurement data given as input.

        X2: numpy.ndarray, shape (n_samples, n_features)
            Measurement data given as target.

        U: numpy.ndarray, shape (n_samples, n_control_features)
            Time series of external actuation/control.

        """
        Nlift = X1.shape[1]
        W = X2.T
        V = np.vstack([X1.T, U.T])
        VVt = V @ V.T
        WVt = W @ V.T
        M = WVt @ np.linalg.pinv(VVt)  # Matrix [A B]
        self._state_matrix_ = M[0:Nlift, 0:Nlift]
        self._control_matrix_ = M[0:Nlift, Nlift:]
        self._coef_ = M

        # Compute Koopman V, eigenvectors, lamda
        [self._eigenvalues_, self._eigenvectors_] = np.linalg.eig(self.state_matrix_)
        self._unnormalized_modes = self._eigenvectors_
        self._ur = np.eye(self.n_input_features_)
        self._tmp_compute_psi = np.linalg.inv(self._eigenvectors_)

    def predict(self, x, u):
        """
        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to base prediction.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        Returns
        -------
        y: numpy.ndarray, shape (n_samples, n_features)
            Prediction of x one timestep in the future.

        """
        check_is_fitted(self, "coef_")
        y = x @ self.state_matrix_.T + u @ self.control_matrix_.T
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
        psi = self._tmp_compute_psi @ x.T
        return psi

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    @property
    def control_matrix_(self):
        check_is_fitted(self, "_control_matrix_")
        return self._control_matrix_

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
