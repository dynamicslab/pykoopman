"""module for kernel dmd"""
from __future__ import annotations

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


class KDMD(BaseRegressor):
    """
    Kernel Dynamic Mode Decomposition.

    See the following reference for more details:

    `Williams, M. O., Rowley, C. W., & Kevrekidis, I. G. (2014).
    "A kernel-based approach to data-driven Koopman spectral analysis."
    arXiv preprint arXiv:1411.2260. <https://arxiv.org/pdf/1411.2260.pdf>`

    Args:
        svd_rank (int): The rank for the truncation. If 0, the method computes
            the optimal rank and uses it for truncation. If positive integer,
            the method uses the argument for the truncation. If float between 0
            and 1, the rank is the number of the biggest singular values that
            are needed to reach the 'energy' specified by `svd_rank`. If -1,
            the method does not compute truncation. Default is 0.
        tlsq_rank (int): The rank for the truncation. If 0, the method does not
            compute any noise reduction. If positive number, the method uses the
            argument for the SVD truncation used in the TLSQ method.
        forward_backward (bool): If True, the low-rank operator is computed
            like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
            Default is False.
        tikhonov_regularization (bool or None): Tikhonov parameter for the
            regularization. If None, no regularization is applied. If float,
            it is used as the λ Tikhonov parameter.
        kernel (sklearn.gaussian_process.Kernel): An instance of kernel from sklearn.

    Attributes:
        svd_rank (int): The rank for the truncation.
        tlsq_rank (int): The rank for the truncation.
        forward_backward (bool): If True, the low-rank operator is computed
            like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
        tikhonov_regularization (bool or None): Tikhonov parameter for the
            regularization.
        kernel (sklearn.gaussian_process.Kernel): An instance of kernel from sklearn.
        n_samples_ (int): Number of samples in KDMD.
        n_input_features_ (int): Dimension of input features, i.e., the dimension
            of each sample.
        _snapshots (numpy.ndarray): Column-wise data matrix of shape
            (n_input_features_, n_samples_).
        _snapshots_shape (tuple): Shape of column-wise data matrix.
        _X (numpy.ndarray): Training features column-wise arranged, needed for
            prediction. Shape is (n_input_features_, n_samples).
        _Y (numpy.ndarray): Training target, column-wise arranged. Shape is
            (n_input_features_, n_samples).
        _coef_ (numpy.ndarray): Reduced Koopman state transition matrix of shape
            (svd_rank, svd_rank).
        _eigenvalues_ (numpy.ndarray): Koopman lambda of shape (svd_rank,).
        _eigenvectors_ (numpy.ndarray): Koopman eigenvectors of shape
            (svd_rank, svd_rank).
        _unnormalized_modes (numpy.ndarray): Koopman V of shape
            (svd_rank, n_input_features_).
        _state_matrix_ (numpy.ndarray): Reduced Koopman state transition matrix
            of shape (svd_rank, svd_rank).
        self.C (numpy.ndarray): Linear matrix that maps kernel product features
            to eigenfunctions of shape (svd_rank, n_samples_).
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
        Kernel Dynamic Mode Decomposition.

        Args:
            svd_rank (int, optional): The rank for the truncation.
                If set to 0, the method computes the optimal rank
                and uses it for truncation. If set to a positive integer,
                the method uses the specified rank for truncation.
                If set to a float between 0 and 1, the rank is determined
                based on the specified energy level. If set to -1, no
                truncation is performed. Default is 1.0.
            tlsq_rank (int, optional): The rank for the truncation used
                in the total least squares preprocessing. If set to 0,
                no noise reduction is performed. If set to a positive integer,
                the method uses the specified rank for the SVD truncation
                in the TLSQ method. Default is 0.
            forward_backward (bool, optional): Whether to compute the
                low-rank operator using the forward-backward method similar
                to fbDMD. If set to True, the low-rank operator is computed
                with forward-backward DMD. If set to False, standard DMD is used.
                Default is False.
            tikhonov_regularization (float or None, optional): Tikhonov
                regularization parameter for regularization. If set to None,
                no regularization is applied. If set to a float, it is used
                as the regularization parameter. Default is None.
            kernel (Kernel, optional): An instance of the kernel class from
                sklearn.gaussian_process. Default is RBF().
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

    def fit(self, x, y=None, dt=1):
        """
        Fits the KDMD model to the provided training data.

        Args:
            x: numpy.ndarray, shape (n_samples, n_features)
                Measurement data input.

            y: numpy.ndarray, shape (n_samples, n_features), optional
                Measurement data output to be fitted. Defaults to None.

            dt: float, optional
                Time interval between `x` and `y`. Defaults to 1.

        Returns:
            KDMD:
                The fitted KDMD instance.
        """

        # if y is not None:
        #    warn("pydmd regressors do not require the y argument when fitting.")
        self.n_samples_, self.n_input_features_ = x.shape
        n_samples = self.n_samples_
        if y is None:
            self._snapshots, self._snapshots_shape = _col_major_2darray(x.T)
            # self._snapshots.shape[1]
            X = self._snapshots[:, :-1]
            Y = self._snapshots[:, 1:]
        else:
            # if we have pairs of data
            X = x.T
            Y = y.T

        # total least square preprocessing on X and Y - features, samples
        self._X, self._Y = compute_tlsq(X, Y, self.tlsq_rank)

        # compute KDMD operators, lamda, and koopman V
        # note that this method is built by considering row-wise collected data
        [
            self._coef_,
            self._eigenvalues_,
            self._eigenvectors_,
            self._unnormalized_modes,
        ] = self._regressor_compute_kdmdoperator(self._X.T, self._Y.T)

        # Default timesteps
        self._set_initial_time_dictionary({"t0": 0, "tend": n_samples - 1, "dt": 1})

        # _coef_ as the transpose
        # self._coef_ = self._regressor_atilde.T

        return self

    def predict(self, x):
        """
        Predicts the future states based on the given input data.

        Args:
            x: numpy.ndarray, shape (n_samples, n_features)
                Measurement data upon which to base the prediction.

        Returns:
            numpy.ndarray, shape (n_samples, n_features)
                Prediction of the future states.
        """

        check_is_fitted(self, "coef_")

        phi = self._compute_psi(x_col=x.T)
        phi_next = np.diag(self.eigenvalues_) @ phi
        x_next_T = self._unnormalized_modes @ phi_next
        return np.real(x_next_T).T

    def _compute_phi(self, x_col):
        """
        Computes the phi(x) given x.

        Args:
            x_col: numpy.ndarray, shape (n_samples, n_features)
                Measurement data upon which to compute phi values.

        Returns:
            numpy.ndarray, shape (n_samples, n_input_features_)
                Value of phi at x.
        """
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        psi = self._compute_psi(x_col)
        phi = np.real(self.eigenvectors_ @ psi)
        return phi

    def _compute_psi(self, x_col):
        """
        Computes the psi(x) given x.

        Args:
            x_col: numpy.ndarray, shape (n_samples, n_features)
                Measurement data upon which to compute psi values.

        Returns:
            numpy.ndarray, shape (n_samples, n_input_features_)
                Value of psi at x.
        """
        # compute psi - one column if x is a row
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        return self._tmp_compute_psi_kdmd @ self.kernel(self._X.T, x_col.T)

    @property
    def coef_(self):
        """
        Getter property for the coef_ attribute.

        Returns:
            numpy.ndarray, shape (svd_rank, svd_rank)
                Reduced Koopman state transition matrix.
        """
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        """
        Getter property for the state_matrix_ attribute.

        Returns:
            numpy.ndarray, shape (svd_rank, svd_rank)
                Reduced Koopman state transition matrix.
        """
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    @property
    def eigenvalues_(self):
        """
        Getter property for the eigenvalues_ attribute.

        Returns:
            numpy.ndarray, shape (svd_rank,)
                Koopman eigenvalues.
        """
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def eigenvectors_(self):
        """
        Getter property for the eigenvectors_ attribute.

        Returns:
            numpy.ndarray, shape (svd_rank, svd_rank)
                Koopman eigenvectors.
        """
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def unnormalized_modes(self):
        """
        Getter property for the unnormalized_modes attribute.

        Returns:
            numpy.ndarray, shape (svd_rank, n_input_features_)
                Koopman unnormalized modes.
        """
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        """
        Getter property for the ur attribute.

        Returns:
            numpy.ndarray, shape (n_samples_, n_input_features_)
                Linear matrix that maps kernel product features to eigenfunctions.
        """
        check_is_fitted(self, "_ur")
        return self._ur

    def _regressor_compute_kdmdoperator(self, X, Y):
        """
        Computes the KDMD operator given input data X and target data Y.

        Args:
            X: numpy.ndarray, shape (n_samples_, n_input_features_)
                Training data input.
            Y: numpy.ndarray, shape (n_samples_, n_input_features_)
                Training data target output.

        Returns:
            list
                A list containing the following elements:
                - koopman_matrix: numpy.ndarray, shape (svd_rank, svd_rank)
                    Reduced Koopman state transition matrix.
                - koopman_eigvals: numpy.ndarray, shape (svd_rank,)
                    Koopman eigenvalues.
                - koopman_eigenvectors: numpy.ndarray, shape (svd_rank, svd_rank)
                    Koopman eigenvectors.
                - unnormalized_modes: numpy.ndarray, shape (svd_rank, n_input_features_)
                    Koopman unnormalized modes.
        """
        # compute kernel K(X,X)
        # since sklearn kernel function takes rowwise collected data.
        KXX = self.kernel(X, X)
        KYX = self.kernel(Y, X)

        # compute eig of PD matrix, so it is SVD
        U, s2, _ = compute_svd(KXX, self.svd_rank)
        s = np.sqrt(s2)
        # remember that we need sigma, but svd or eig only gives you the s^2

        # optional compute tiknoiv reg
        if self.tikhonov_regularization is not None:
            s = (
                s**2 + self.tikhonov_regularization * np.linalg.norm(X)
            ) * np.reciprocal(s)

        koopman_matrix = (
            np.diag(np.reciprocal(s))
            @ U.T.conj()
            @ KYX.T
            @ U
            @ np.diag(np.reciprocal(s))
        )

        # optional compute fb
        if self.forward_backward:
            KYY = self.kernel(Y, Y)
            KXY = KYX.T
            bU, bs2, _ = compute_svd(KYY, self.svd_rank)
            bs = np.sqrt(bs2)
            if self.tikhonov_regularization is not None:
                bs = (
                    bs**2 + self.tikhonov_regularization * np.linalg.norm(Y)
                ) * np.reciprocal(bs)

            atilde_back = (
                np.diag(np.reciprocal(bs))
                @ bU.T.conj()
                @ KXY.T
                @ bU
                @ np.diag(np.reciprocal(bs))
            )
            koopman_matrix = sqrtm(koopman_matrix @ np.linalg.inv(atilde_back))

        # self._regressor_atilde = atilde
        self._state_matrix_ = koopman_matrix

        # compute eigenquantities
        koopman_eigvals, koopman_eigenvectors = np.linalg.eig(koopman_matrix)

        # compute unnormalized V
        BV = np.linalg.lstsq(U @ np.diag(s), X, rcond=None)[0].T
        unnormalized_modes = BV @ koopman_eigenvectors

        # compute psi
        self._ur = BV  # U @ np.diag(s)
        self._tmp_compute_psi_kdmd = (
            np.linalg.inv(koopman_eigenvectors) @ np.diag(np.reciprocal(s)) @ U.T
        )

        return [
            koopman_matrix,
            koopman_eigvals,
            koopman_eigenvectors,
            unnormalized_modes,
        ]

    def _set_initial_time_dictionary(self, time_dict):
        """
        Sets the initial time dictionary.

        Args:
            time_dict: dict
                Dictionary containing the time information with keys 't0', 'tend',
                and 'dt'.
        """
        if not ("t0" in time_dict and "tend" in time_dict and "dt" in time_dict):
            raise ValueError('time_dict must contain the keys "t0", "tend" and "dt".')
        if len(time_dict) > 3:
            raise ValueError(
                'time_dict must contain only the keys "t0", "tend" and "dt".'
            )

        self._original_time = DMDTimeDict(dict(time_dict))
        self._dmd_time = DMDTimeDict(dict(time_dict))


def _col_major_2darray(X):
    def _col_major_2darray(X):
        """
        Converts the input snapshots into a 2D matrix by column-major ordering.

        Args:
            X: int or numpy.ndarray
                The input snapshots.

        Returns:
            snapshots: numpy.ndarray
                The 2D matrix that contains the flattened snapshots.

            snapshots_shape: tuple
                The shape of the original snapshots.
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
