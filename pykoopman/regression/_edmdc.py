"""module for extended dmd with control"""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor

# TODO: add support for time delay observables, so we will
#       have n_consumption_.


class EDMDc(BaseRegressor):
    """Module for Extended DMD with control (EDMDc) regressor.

    Aims to determine the system matrices A, B, C that satisfy y' = Ay + Bu and x = Cy,
    where y' is the time-shifted observable with y0 = phi(x0) and u is the control
    input. B and C are the unknown control and measurement matrices, respectively.

    The objective functions, \\|Y'-AY-BU\\|_F and \\|X-CY\\|_F, are minimized using
    least-squares regression and singular value decomposition.

    See the following reference for more details:
        Korda, M. and Mezic, I. "Linear predictors for nonlinear dynamical systems:
        Koopman operator meets model predictive control." Automatica, Vol. 93, 149–160.
        <https://www.sciencedirect.com/science/article/abs/pii/S000510981830133X>

    Attributes:
        coef_ (numpy.ndarray):
            Weight vectors of the regression problem. Corresponds to either [A] or
            [A,B].
        state_matrix_ (numpy.ndarray):
            Identified state transition matrix A of the underlying system.
        control_matrix_ (numpy.ndarray):
            Identified control matrix B of the underlying system.
        projection_matrix_ (numpy.ndarray):
            Projection matrix into low-dimensional subspace of shape (n_input_features
            +n_control_features, svd_rank).
        projection_matrix_output_ (numpy.ndarray):
            Projection matrix into low-dimensional subspace of shape (n_input_features
            +n_control_features, svd_output_rank).
    """

    def __init__(self):
        """Initialize the EDMDc regressor."""
        pass

    def fit(self, x, y=None, u=None, dt=None):
        """Fit the EDMDc regressor to the given data.

        Args:
            x (numpy.ndarray):
                Measurement data to be fit.
            y (numpy.ndarray, optional):
                Time-shifted measurement data to be fit. Defaults to None.
            u (numpy.ndarray, optional):
                Time series of external actuation/control. Defaults to None.
            dt (scalar, optional):
                Discrete time-step. Defaults to None.

        Returns:
            self: Fitted EDMDc instance.
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
        """Fit the EDMDc regressor with unknown control matrix B.

        Args:
            X1 (numpy.ndarray):
                Measurement data given as input.
            X2 (numpy.ndarray):
                Measurement data given as target.
            U (numpy.ndarray):
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
        """Predict the next timestep based on the given data.

        Args:
            x (numpy.ndarray):
                Measurement data upon which to base prediction.
            u (numpy.ndarray):
                Time series of external actuation/control.

        Returns:
            y (numpy.ndarray):
                Prediction of x one timestep in the future.
        """
        check_is_fitted(self, "coef_")
        y = x @ self.state_matrix_.T + u @ self.control_matrix_.T
        return y

    def _compute_phi(self, x_col):
        """Compute psi(x) given x.

        Args:
            x_col (numpy.ndarray):
                Input data x.

        Returns:
            psi (numpy.ndarray):
                Value of psi(x).
        """
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        phi = self._ur.T @ x_col
        return phi

    def _compute_psi(self, x_col):
        """Compute psi(x) given x.

        Args:
            x_col (numpy.ndarray):
                Input data x.

        Returns:
            psi (numpy.ndarray):
                Value of psi(x).
        """
        # compute psi - one column if x is a row
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        psi = self._tmp_compute_psi @ x_col
        return psi

    @property
    def coef_(self):
        """Weight vectors of the regression problem. Corresponds to either [A] or
        [A,B]."""
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        """Identified state transition matrix A of the underlying system.

        Returns:
            state_matrix (numpy.ndarray):
                State transition matrix A.
        """
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    @property
    def control_matrix_(self):
        """Identified control matrix B of the underlying system.

        Returns:
            control_matrix (numpy.ndarray):
                Control matrix B.
        """
        check_is_fitted(self, "_control_matrix_")
        return self._control_matrix_

    @property
    def eigenvalues_(self):
        """Identified Koopman lambda.

        Returns:
            eigenvalues (numpy.ndarray):
                Koopman eigenvalues.
        """
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def eigenvectors_(self):
        """Identified Koopman eigenvectors.

        Returns:
            eigenvectors (numpy.ndarray):
                Koopman eigenvectors.
        """
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def unnormalized_modes(self):
        """Identified Koopman eigenvectors.

        Returns:
            unnormalized_modes (numpy.ndarray):
                Koopman eigenvectors.
        """
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        """Matrix U that is part of the SVD.

        Returns:
            ur (numpy.ndarray):
                Matrix U.
        """
        check_is_fitted(self, "_ur")
        return self._ur
