import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class DMDc(BaseRegressor):
    """
    DMD with control (DMDc) regressor.

    Parameters
    ----------
    """

    def __init__(self, svd_rank=None, svd_output_rank=0, control_matrix=None):
        self.svd_rank = svd_rank
        self.svd_output_rank = svd_output_rank
        self.control_matrix_ = control_matrix

    def fit(self, x, u, y=None):
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
        elif u.ndim == 2:
            if u.shape[0] > X1.shape[0]:
                u = u[:-1, :]
            C = u
        self.n_control_features_ = C.shape[1]

        r = self.svd_rank
        if r is None:
            r = self.n_input_features_ + self.n_control_features_

        if self.control_matrix_ is None:
            Omega = np.vstack([X1.T, C.T])
            U, s, Vh = np.linalg.svd(Omega, full_matrices=False)
            Ur = U[:, 0:r]
            sr = s[0:r]
            Vr = Vh[0:r, :].T
            G = np.dot(X2.T, np.dot(Vr * (sr ** (-1)), Ur.T))
            self.state_matrix_ = G[:, 0 : self.n_input_features_]
            self.control_matrix_ = G[:, self.n_input_features_ :]
        else:
            if self.n_input_features_ in self.control_matrix_.shape is False:
                raise TypeError("Control vector/matrix B has wrong shape.")
            if self.control_matrix_.shape[1] == self.n_input_features_:
                self.control_matrix_ = self.control_matrix_.T
            if self.control_matrix_.shape[1] != self.n_control_features_:
                raise TypeError(
                    "The control matrix B must have the same number of inputs as the "
                    "control variable u."
                )

            U, s, Vh = np.linalg.svd(X1.T, full_matrices=False)
            A = np.dot(
                X2.T - np.dot(self.control_matrix_, C.T),
                np.dot(Vh.T * (s ** (-1)), U.T),
            )
            self.state_matrix_ = A
            G = A

        self.coef_ = G

        return self

    def predict(self, x, u):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
            Measurement data upon which to base prediction.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        Returns
        -------
        y: numpy ndarray, shape (n_samples, n_features)
            Prediction of x one timestep in the future.

        """
        check_is_fitted(self, "coef_")
        # TODO: this needs to be modified
        # y = np.dot(self.state_matrix, x.T) + np.dot(self.control_matrix, u.T)
        # return y.T
        return self.regressor.predict(x.T).T
