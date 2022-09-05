from warnings import warn

import numpy as np
from pydmd.dmdbase import DMDTimeDict
from pydmd.utils import compute_svd
from pydmd.utils import compute_tlsq
from scipy.linalg import sqrtm
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor

# from numpy import identity


class KDMD(BaseRegressor):
    """
    Kernel Dynamic Mode Decomposition.

    Parameters
    ----------

    """

    def __init__(
        self,
        svd_rank=1.0,  # 1.0 means keeping all ranks
        tlsq_rank=0,
        forward_backward=False,
        tikhonov_regularization=None,
        kernel=RBF(),
    ):
        """
        Kernel DMD class

        """
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.forward_backward = forward_backward
        self.tikhonov_regularization = tikhonov_regularization
        self.kernel = kernel

        if not isinstance(self.kernel, Kernel):
            raise ValueError(
                "kernel must be a subclass of sklearn.gaussian_process.kernel"
            )

    def fit(self, x, y=None):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_examples, n_features)
            Measurement data to be fit.

        Returns
        -------
        self: retucrns a fit ``DMDRegressor`` instance
        """
        if y is not None:
            warn("pydmd regressors do not require the y argument when fitting.")
        self.n_samples_, self.n_input_features_ = x.shape
        # We transpose x because PyDMD assumes examples are columns, not rows
        self._regressor_fit(x.T)

        self._coef_ = self._regressor_atilde.T

        # Get Koopman modes, eigenvectors, eigenvalues from pydmd
        # self._amplitudes_ = self._regressor_amplitudes
        # self._eigenvalues_ = self._regressor_eigs
        # self._modes_ = self._regressor_modes
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
        return self._regression_predict(x.T).T

    def _regressor_fit(self, x):
        """
        here we assume data snapshots is column-wise collected:
        i.e., X = [x_1, x_2, ... , x_M]
        """
        # 0. transpose the matrix
        self._snapshots, self._snapshots_shape = _col_major_2darray(x)
        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-1]
        Y = self._snapshots[:, 1:]

        # 1. tlsq on X and Y
        self._X, self._Y = compute_tlsq(X, Y, self.tlsq_rank)

        # 2. compute KDMD operators, eigenvalues, and koopman modes
        # note that this method is built by considering row-wise collected data
        (
            self._eigenvalues_,
            self._eigenvectors_,
            self._modes_,
            self._unnormalized_modes,
            self._amplitudes_,
        ) = self._regressor_compute_kdmdoperator(self._X.T, self._Y.T)

        # Default timesteps
        self._set_initial_time_dictionary({"t0": 0, "tend": n_samples - 1, "dt": 1})

        # 6. get _coef_ as the transpose
        self._coef_ = self._regressor_atilde.T

        return self

    def _regression_predict(self, x):
        """x is column-wise collected (n_features, n_samples)"""

        KXx = self.kernel(self._X.T, x.T)
        phi_x_T = self.C @ KXx

        x_next = self._unnormalized_modes @ np.diag(self.eigenvalues_) @ phi_x_T
        x_next = np.real(x_next)
        return x_next

    def _regressor_compute_kdmdoperator(self, X, Y):
        """
        here we assume X,Y are rowwise collected. so we can directly use existing
        formulas.
        """
        # 2. compute kernel K(X,X)
        # since sklearn kernel function takes rowwise collected data.
        KXX = self.kernel(X, X)
        KYX = self.kernel(Y, X)

        # 3. compute eig of PD matrix, so it is SVD
        U, s2, _ = compute_svd(KXX, self.svd_rank)
        s = np.sqrt(
            s2
        )  # remember that we need sigma, but svd or eig only gives you the s^2

        # 4. optional compute tiknoiv reg
        if self.tikhonov_regularization is not None:
            s = (
                s**2 + self.tikhonov_regularization * np.linalg.norm(X)
            ) * np.reciprocal(s)

        # 5. compute k_kdmd
        atilde = np.linalg.multi_dot(
            [np.diag(np.reciprocal(s)), U.T.conj(), KYX, U, np.diag(np.reciprocal(s))]
        )

        # 5. optional compute fb
        if self.forward_backward:
            KYY = self.kernel(Y, Y)
            KXY = KYX.T
            bU, bs2, _ = compute_svd(KYY, self.svd_rank)
            bs = np.sqrt(bs2)
            if self.tikhonov_regularization is not None:
                bs = (
                    bs**2 + self.tikhonov_regularization * np.linalg.norm(Y)
                ) * np.reciprocal(bs)

            atilde_back = np.linalg.multi_dot(
                [
                    np.diag(np.reciprocal(bs)),
                    bU.T.conj(),
                    KXY,
                    bU,
                    np.diag(np.reciprocal(bs)),
                ]
            )
            atilde = sqrtm(atilde.dot(np.linalg.inv(atilde_back)))

        self._regressor_atilde = atilde

        # 6. compute eigenquantities
        koopman_eigvals, koopman_eigenvectors = np.linalg.eig(atilde)

        # 7. compute modes
        # modes = np.linalg.pinv(np.linalg.multi_dot([
        #     KXX,
        #     U,
        #     np.diag(np.reciprocal(s)),
        #     koopman_eigenvectors
        # ])) @ X

        A_ = np.linalg.multi_dot(
            [KXX, U, np.diag(np.reciprocal(s)), koopman_eigenvectors]
        )
        b_ = X
        unnormalized_modes = np.linalg.lstsq(A_, b_, rcond=None)[0]

        # consistent with pydmd fashion so that row is the system dimension
        unnormalized_modes = unnormalized_modes.T
        # 7.5 make sure each column is normalized.
        normalized_modes = unnormalized_modes @ np.diag(
            1.0 / np.linalg.norm(unnormalized_modes, axis=0)
        )

        # 8. amplititute
        # follow pydmd fashion, use least square to get it.
        a = np.linalg.lstsq(normalized_modes, self._snapshots.T[0], rcond=None)[0]

        # 9. compute C matrix for prediction
        # x_test_next = np.real(B^T * Lambda^k * C * kernel(X, x_test))
        self.C = koopman_eigenvectors.T @ np.diag(np.reciprocal(s)) @ U.T

        return (
            koopman_eigvals,
            koopman_eigenvectors,
            normalized_modes,
            unnormalized_modes,
            a,
        )

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

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def eigenvectors_(self):
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def amplitudes_(self):
        check_is_fitted(self, "_amplitudes_")
        return self._amplitudes_

    @property
    def modes_(self):
        check_is_fitted(self, "_modes_")
        return self._modes_.T  # so that it will be (n_modes, n_system_dim)

    @property
    def unnormalized_modes(self):
        check_is_fitted(self, "_modes_")
        return self._unnormalized_modes

    @property
    def regressor_atilde(self):
        return self._regressor_atilde


# get the data in 2D shape. referred to pydmd
def _col_major_2darray(X):
    """
    Private method that takes as input the snapshots and stores them into a
    2D matrix, by column. If the input data is already formatted as 2D
    array, the method saves it, otherwise it also saves the original
    snapshots shape and reshapes the snapshots.

    :param X: the input snapshots.
    :type X: int or numpy.ndarray
    :return: the 2D matrix that contains the flatten snapshots, the shape
        of original snapshots.
    :rtype: numpy.ndarray, tuple
    """

    # If the data is already 2D ndarray
    if isinstance(X, np.ndarray) and X.ndim == 2:
        snapshots = X
        snapshots_shape = None
    else:
        input_shapes = [np.asarray(x).shape for x in X]

        if len(set(input_shapes)) != 1:
            raise ValueError("Snapshots have not the same dimension.")

        snapshots_shape = input_shapes[0]
        snapshots = np.transpose([np.asarray(x).flatten() for x in X])

    # check condition number of the data passed in
    cond_number = np.linalg.cond(snapshots)
    if cond_number > 10e4:
        warn(
            "Input data matrix X has condition number {}. "
            """Consider preprocessing data, passing in augmented
            data matrix, or regularization methods.""".format(
                cond_number
            )
        )

    return snapshots, snapshots_shape
