# from warnings import warn
import numpy as np
import scipy
from pydmd.dmdbase import DMDTimeDict
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

        self._state_matrix_ = np.linalg.lstsq(X1, X2)[0].T  # [0:Nlift, 0:Nlift]
        self._coef_ = self._state_matrix_
        [self._eigenvalues_, self.eigenvectors_] = scipy.linalg.eig(self.state_matrix_)

        self._unnormalized_modes = self.eigenvectors_
        self.C = np.linalg.inv(self.eigenvectors_)

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
        check_is_fitted(self, "coef_")
        y = x @ self.state_matrix_.T
        return y

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

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
