import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class EDMD(BaseRegressor):
    """
    Extended DMD (EDMD) regressor.

    Aims to determine the system matrices A,C
    that satisfy y' = Ay and x = Cy, where y' is the time-shifted
    observable with y0 = phi(x0). C is the measurement matrix that maps back to the
    state.

    The objective functions,
    :math:`\\|Y'-AY\\|_F`,
    are minimized using least-squares regression and singular value
    decomposition.

    See the following reference for more details:
        `M.O. Williams , I.G. Kevrekidis, C.W. Rowley
        "A Data–Driven Approximation of the Koopman Operator:
        Extending Dynamic Mode Decomposition."
        Journal of Nonlinear Science, Vol. 25, 1307-1346, 2015.
        <https://link.springer.com/article/10.1007/s00332-015-9258-5>`_

    Parameters
    ----------

    Attributed
    ----------
    coef_ : array, shape (n_input_features_, n_input_features_) or
        (n_input_features_, n_input_features_ + n_control_features_)
        Weight vectors of the regression problem. Corresponds to either [A] or [A,B]

    state_matrix_ : array, shape (n_input_features_, n_input_features_)
        Identified state transition matrix A of the underlying system.

    projection_matrix_ : array, shape (n_input_features_+n_control_features_, svd_rank)
        Projection matrix into low-dimensional subspace.

    projection_matrix_output_ : array, shape (n_input_features_+n_control_features_,
                                              svd_output_rank)
        Projection matrix into low-dimensional subspace.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None, dt=None):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_samples, n_features)
            Measurement data to be fit.

        y: numpy.ndarray, shape (n_samples, n_features)
            Time-shifted measurement data to be fit

        dt: scalar
            Discrete time-step

        Returns
        -------
        self: returns a fitted ``EDMD`` instance
        """
        self.n_samples_, self.n_input_features_ = x.shape
        if y is None:
            X1 = x[:-1, :]
            X2 = x[1:, :]
        else:
            X1 = x
            X2 = y

        self._fit(X1, X2)
        return self

    def _fit(self, X1, X2):
        Nlift = X1.shape[1]
        W = X2.T
        V = X1.T
        VVt = V @ V.T
        WVt = W @ V.T
        M = WVt @ np.linalg.pinv(VVt)
        self.state_matrix_ = M[0:Nlift, 0:Nlift]
        self.coef_ = M

        # Compute Koopman modes, eigenvectors, eigenvalues
        # [
        #     self.eigenvalues_,
        #     self.left_eigenvectors_,
        #     self.eigenvectors_,
        # ] = scipy.linalg.eig(self.state_matrix_, left=True)
        self.eigenvalues_, self.eigenvectors_ = np.linalg.eig(self.state_matrix_)
        _, self.left_eigenvectors_ = np.linalg.eig(self.state_matrix_.T)

        sort_idx = np.argsort(self.eigenvalues_)
        sort_idx = sort_idx[::-1]
        self.eigenvalues_ = self.eigenvalues_[sort_idx]
        self.modes_ = X1 @ self.eigenvectors_[:, sort_idx]
        self.kef_ = X1 @ self.left_eigenvectors_[:, sort_idx]
        self.left_evecs = self.left_eigenvectors_[:, sort_idx]
        self.right_evecs = self.eigenvectors_[:, sort_idx]

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
        check_is_fitted(self, "coef_")
        y = x @ self.state_matrix_.T
        return y
