import numpy as np
import scipy
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class KEF(BaseRegressor):
    """
    Regressor for Koopman eigenfunction form.

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
        VVt = X1.T @ X1
        WVt = X2.T @ X1
        M = WVt @ np.linalg.pinv(VVt)

        # Compute Koopman modes, eigenvectors, eigenvalues
        [evals, left_evecs, right_evecs] = \
            scipy.linalg.eig(M, left=True)

        sort_idx = np.argsort(evals)
        sort_idx = sort_idx[::-1]

        evals = evals[sort_idx]
        left_evecs = left_evecs[:, sort_idx]
        right_evecs = right_evecs[:, sort_idx]

        self.eigenvalues_ = evals
        self.modes_ = X1 @ right_evecs
        self.kef_ = X1 @ left_evecs

        self.projection_matrix_ = right_evecs
        self.state_matrix_ = np.real(self.projection_matrix_ @
                                     np.diag(evals) @
                                     np.linalg.pinv(self.projection_matrix_))
        self.coef_ = self.state_matrix_


    def reduce(self, t, x, y, rank=None):
        self.test_data = {
            'time': t,
            'state': x,
            'obsv': y
        }

        # Select valid Koopman eigenfunctions
        if rank is None:
            rank = self._evaluate_efuns()

        self.projection_matrix_ = self.projection_matrix_[:, :rank]
        self.eigenvalues_ = self.eigenvalues_[:rank]
        self.state_matrix_ = np.real(self.projection_matrix_[:, :rank] @
                                     np.diag(self.eigenvalues_[:rank]) @
                                     np.linalg.pinv(self.projection_matrix_[:, :rank]))

    def _evaluate_efuns(self):
        """
        Validity check of eigenfunctions
        phi(x(t)) == phi(x(0))*exp(lambda*t)
        """
        for i in range(len(self.eigenvalues_)):
            self.kef_[:, i]

        r = 5
        return r

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
        # y = x @ self.state_matrix_.T
        # y = b @ np.diag(self.eigenvalues_) @ self.modes_
        # y = np.real(x @ np.linalg.pinv(self.projection_matrix_) @ np.diag(self.eigenvalues_) @  \
        #     self.projection_matrix_)
        y = np.linalg.multi_dot(
            [self.projection_matrix_, np.diag(self.eigenvalues_), scipy.linalg.pinv(
                self.projection_matrix_), x.T]).T
        return y
