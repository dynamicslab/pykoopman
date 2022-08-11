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
    decomposition/

    See the following reference for more details:

        `Korda, M. and Mezic, I.
        "Linear predictors for nonlinear dynamical systems:
        Koopman operator meets model predictive control."
        Automatica, Vol. 93, 149–160.
        <https://www.sciencedirect.com/science/article/abs/pii/S000510981830133X>`_

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

     Examples
    --------
    TODO
    """

    def __init__(self):
        pass

    def fit(self, x, y=None, u=None, dt=None):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
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
        elif u.ndim == 2:
            if u.shape[0] > X1.shape[0]:
                u = u[:-1, :]
            C = u
        self.n_control_features_ = C.shape[1]

        self._fit(X1, X2, C)
        return self

    def _fit(self, X1, X2, U):
        Nlift = X1.shape[1]
        W = X2.T
        V = np.vstack([X1.T, U.T])
        VVt = V @ V.T
        WVt = W @ V.T
        M = WVt @ np.linalg.pinv(VVt)  # Matrix [A B]
        self.state_matrix_ = M[0:Nlift, 0:Nlift]
        self.control_matrix_ = M[0:Nlift, Nlift:]
        self.coef_ = M

        # Compute Koopman modes, eigenvectors, eigenvalues
        [self.eigenvalues_, self.eigenvectors_] = np.linalg.eig(self.state_matrix_)
        #TODO
        # self.modes_ =

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
        y = x @ self.state_matrix_.T + u @ self.control_matrix_.T
        return y
