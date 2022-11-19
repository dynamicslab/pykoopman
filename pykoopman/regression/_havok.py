from __future__ import annotations

from warnings import warn

import numpy as np
from optht import optht
from scipy.signal import lsim
from scipy.signal import lti
from sklearn.utils.validation import check_is_fitted

from ..common import drop_nan_rows
from ..differentiation._derivative import Derivative
from ._base import BaseRegressor


class HAVOK(BaseRegressor):
    """
    Hankel Alternative View of Koopman (HAVOK) regressor.

    Aims to determine the system matrices A,B
    that satisfy d/dt v = Av + Bu, where v is the vector of the leading delay
    coordinates and u is a low-energy delay coordinate acting as forcing.
    A and B are the unknown system and control matrices, respectively.
    The delay coordinates are obtained by computing the SVD from a Hankel matrix.

    The objective function,
    :math:`\\|dV-AV-BU\\|_F`,
    is minimized using least-squares regression.

    See the following reference for more details:

        `Brunton, S.L., Brunton, B.W., Proctor, J.L., Kaiser, E. & Kutz, J.N.
        "Chaos as an intermittently forced linear system."
        Nature Communications, Vol. 8(19), 2017.
        <https://www.nature.com/articles/s41467-017-00030-8>`_

    Parameters
    ----------

    Attributed
    ----------
    coef_ : array, shape (n_input_features_, n_input_features_) or
        (n_input_features_, n_input_features_ + n_control_features_)
        Weight vectors of the regression problem. Corresponds to either [A] or [A,B]

    state_matrix_ : array, shape (n_input_features_, n_input_features_)
        Identified state transition matrix A of the underlying system.

    control_matrix_ : array, shape (n_input_features_, n_control_features_)
        Identified control matrix B of the underlying system.

    projection_matrix_ : array, shape (n_input_features_+n_control_features_, svd_rank)
        Projection matrix into low-dimensional subspace.

    projection_matrix_output_ : array, shape (n_input_features_+n_control_features_,
                                              svd_output_rank)
        Projection matrix into low-dimensional subspace.
    """

    def __init__(
        self,
        svd_rank=None,
        differentiator=Derivative(kind="finite_difference", k=1),
    ):
        self.svd_rank = svd_rank
        self.differentiator = differentiator

    def fit(self, x, y=None, dt=None):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
            Measurement data to be fit.

        y: not used

        dt: scalar
            Discrete time-step

        Returns
        -------
        self: returns a fitted ``HAVOK`` instance
        """

        if y is not None:
            warn("havok regressor does not require the y argument when fitting.")

        if dt is None:
            raise ValueError("havok regressor requires a timestep dt when fitting.")

        self.dt_ = dt
        self.n_samples_, self.n_input_features_ = x.shape
        self.n_control_features_ = 1

        # Create time vector
        t = np.arange(0, self.dt_ * self.n_samples_, self.dt_)

        # SVD to calculate intrinsic observables
        U, s, Vh = np.linalg.svd(x.T, full_matrices=False)

        # calculate rank using optimal hard threshold by Gavish & Donoho
        if self.svd_rank is None:
            self.svd_rank = optht(x, sv=s, sigma=None)

        # calculate time derivative dxdt & normalize
        dVh = self.differentiator(Vh[: self.svd_rank - 1, :].T, t)
        dVh, t, Vh = drop_nan_rows(
            dVh, t, Vh.T
        )  # this line actually makes vh and dvh transposed

        # regression on intrinsic variables v
        xi = np.zeros((self.svd_rank - 1, self.svd_rank))
        for i in range(self.svd_rank - 1):
            xi[i, :] = np.linalg.lstsq(Vh[:, : self.svd_rank], dVh[:, i], rcond=None)[0]

        self.forcing_signal = Vh[:, self.svd_rank - 1]
        self._reduced_state_matrix_ = xi[:, : self.svd_rank - 1]
        self._reduced_control_matrix_ = xi[:, self.svd_rank - 1]

        self.svals = s
        self.measurement_matrix_ = U[:, : self.svd_rank - 1] @ np.diag(
            s[: self.svd_rank - 1]
        )

        self._coef_ = np.hstack(
            [self.reduced_state_matrix_, self.reduced_control_matrix_.reshape(-1, 1)]
        )
        self._projection_matrix_ = self.measurement_matrix_
        self._projection_matrix_output_ = self.measurement_matrix_

        [eigenvalues_, self.eigenvectors_] = np.linalg.eig(self.reduced_state_matrix_)
        # because we fit the model in continuous time,
        # so we need to convert to discrete time
        self.eigenvalues_ = np.exp(eigenvalues_ * dt)

        self._unnormalized_modes = self.measurement_matrix_ @ self.eigenvectors_

        self.C = np.linalg.multi_dot(
            [
                np.linalg.inv(self.eigenvectors_),
                np.diag(np.reciprocal(s[: self.svd_rank - 1])),
                U[:, : self.svd_rank - 1].T,
            ]
        )

    def predict(self, x, u, t):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
            Measurement data upon which to base prediction.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control, which is sampled at time
            instances in t.

        t: numpy.ndarray, shape (n_samples)
            Time vector. Instances at which solution vector shall be provided.
            Must start at 0.


        Returns
        -------
        y: numpy ndarray, shape (n_samples, n_features)
            Prediction of x at time instances provided in t.

        """
        if t[0] != 0:
            raise ValueError("the time vector must start at 0.")

        check_is_fitted(self, "coef_")
        y0 = (
            # np.linalg.inv(np.diag(self.svals[: self.svd_rank - 1]))
            # @
            np.linalg.pinv(self.projection_matrix_)
            @ x.T
        )
        sys = lti(
            self.reduced_state_matrix_,
            self.reduced_control_matrix_[:, np.newaxis],
            self.measurement_matrix_,
            np.zeros((self.n_input_features_, self.n_control_features_)),
        )
        tout, ypred, xpred = lsim(sys, U=u, T=t, X0=y0.T)
        return ypred

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def reduced_state_matrix_(self):
        check_is_fitted(self, "_reduced_state_matrix_")
        return self._reduced_state_matrix_

    @property
    def reduced_control_matrix_(self):
        check_is_fitted(self, "_reduced_control_matrix_")
        return self._reduced_control_matrix_

    @property
    def projection_matrix_(self):
        check_is_fitted(self, "_projection_matrix_")
        return self._projection_matrix_

    @property
    def projection_matrix_output_(self):
        check_is_fitted(self, "_projection_matrix_output_")
        return self._projection_matrix_output_

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
