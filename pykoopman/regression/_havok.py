from warnings import warn

import numpy as np
from derivative import dxdt
from optht import optht
from scipy.signal import lsim
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class HAVOK(BaseRegressor):
    def __init__(self, svd_rank=None,
                 derivative={'kind': 'finite_difference', 'k': 1}):
        self.svd_rank = svd_rank
        self.derivative = derivative

    def fit(self, x, y=None, dt=None):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
            Measurement data to be fit.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        Returns
        -------
        self: returns a fitted ``DMDc`` instance
        """

        if y is not None:
            warn("havok regressor does not require the y argument when fitting.")

        if dt is None:
            raise ValueError("havok regressor requires a timestep dt when fitting.")

        self.dt_ = dt
        self.n_samples_, self.n_input_features_ = x.shape

        # Create time vector
        t = np.arange(0, self.dt_ * self.n_samples_, self.dt_)

        # SVD to calculate intrinsic observables
        U, s, Vh = np.linalg.svd(x.T, full_matrices=False)

        # calculate rank using optimal hard threshold by Gavish & Donoho
        if self.svd_rank is None:
            self.svd_rank = optht(x, sv=s, sigma=None)

        # calculate time derivative dxdt & normalize
        dVh = np.zeros((self.svd_rank-1, self.n_samples_))
        for i in range(self.svd_rank-1):
            dVh[i, :] = dxdt(Vh[i, :],
                             t,
                             kind=self.derivative['kind'],
                             k=self.derivative['k'])


        # regression on intrinsic variables v
        Vh_norm = np.linalg.norm(Vh, axis=1)
        for i in range(self.svd_rank):
            Vh[i, :] = Vh[i, :] / Vh_norm[i]
        xi = np.zeros((self.svd_rank-1, self.svd_rank))
        for i in range(self.svd_rank-1):
            xi[i, :] = np.linalg.lstsq(Vh[:self.svd_rank, :].T, dVh[i, :], rcond=None)[0]
        for i in range(self.svd_rank):
            xi[:, i] = xi[:, i] / Vh_norm[i]

        self.state_matrix_ = xi[:, :self.svd_rank-1]
        self.control_matrix_ = xi[:, self.svd_rank-1]

        self.coef_ = self.state_matrix_
        self.projection_matrix_ = U
        [self.eigenvalues_, self.eigenvectors_] = np.linalg.eig(self.state_matrix_)

    def predict(self, x):
        t = np.arange(0, self.dt_ * self.n_samples_, self.dt_)

        # lsim((A, B, C, D), U=u, T=t, X0=x0)

        pass