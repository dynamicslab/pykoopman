"""module for dmd with control"""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class DMDc(BaseRegressor):
    """
    Implements Dynamic Mode Decomposition with Control (DMDc) regressor.

    This class provides an implementation for DMDc, a variant of Dynamic Mode
    Decomposition, which is a dimensionality reduction technique used to analyze
    dynamical systems. The goal of DMDc is to compute matrices A and B that satisfy
    the equation x' = Ax + Bu, where x' is the time-shifted state w.r.t. x and u is
    the control input.

    Attributes:
        svd_rank (int): Rank of SVD for the input space, i.e., the space of `X` and
            input `U`. This determines the dimensionality of the projected state and
            control matrices. Defaults to None.
        svd_output_rank (int): Rank of SVD for the output space, i.e., the space of `Y`.
            Defaults to None.
        input_control_matrix (numpy.ndarray): The known input control matrix B. Defaults
            to None.
        n_samples_ (int): Total number of one step evolution samples.
        n_input_features_ (int): Dimension of input features.
        n_control_features_ (int): Dimension of input control signal.
        coef_ (numpy.ndarray): Weight vectors of the regression problem. Corresponds
            to either [A] or [A,B].
        state_matrix_ (numpy.ndarray): Identified state transition matrix A of the
            underlying system.
        control_matrix_ (numpy.ndarray): Identified control matrix B of the underlying
            system.
        reduced_state_matrix_ (numpy.ndarray): Reduced state transition matrix.
        reduced_control_matrix_ (numpy.ndarray): Reduced control matrix.
        eigenvalues_ (numpy.ndarray): DMD lamda.
        unnormalized_modes (numpy.ndarray): DMD V.
        projection_matrix_ (numpy.ndarray): Projection matrix into low-dimensional
            subspace.
        projection_matrix_output_ (numpy.ndarray): Projection matrix into
            low-dimensional subspace.
        eigenvectors_ (numpy.ndarray): DMD eigenvectors.

    Example:
        >>> import numpy as np
        >>> import pykoopman as pk
        >>> A = np.matrix([[1.5, 0],[0, 0.1]])
        >>> B = np.matrix([[1],[0]])
        >>> x0 = np.array([4,7])
        >>> u = np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 3, 5])
        >>> n = len(u)+1
        >>> x = np.zeros([n,len(x0)])
        >>> x[0,:] = x0
        >>> for i in range(n-1):
        >>>     x[i+1,:] = A.dot(x[i,:]) + B.dot(u[np.newaxis,i])
        >>> X1 = x[:-1,:]
        >>> X2 = x[1:,:]
        >>> C = u[:,np.newaxis]
        >>> DMDc = pk.regression.DMDc(svd_rank=3, input_control_matrix=B)
        >>> model = pk.Koopman(regressor=DMDc)
        >>> model.fit(x,C)
        >>> Aest = model.A
        >>> Best = model.B
        >>> print(Aest)
        >>> np.allclose(A,Aest)
        [[ 1.50000000e+00 -1.36609474e-17]
         [-1.58023594e-17  1.00000000e-01]]
        True
    """

    def __init__(self, svd_rank=None, svd_output_rank=None, input_control_matrix=None):
        """
        Initialize a DMDc class object.

        Args:
            svd_rank (int, optional): Rank of SVD for the input space. This determines
                the dimensionality of the projected state and control matrices.
                Defaults to None.
            svd_output_rank (int, optional): Rank of SVD for the output space.
                Defaults to None.
            input_control_matrix (numpy.ndarray, optional): The known input control
                matrix B. Defaults to None.

        Raises:
            ValueError: If svd_rank is not an integer.
            ValueError: If svd_output_rank is not an integer.
            ValueError: If input_control_matrix is not a numpy array.
        """
        self.svd_rank = svd_rank
        self.svd_output_rank = svd_output_rank
        self._input_control_matrix = input_control_matrix

    def fit(self, x, y=None, u=None, dt=None):
        """
        Fit the DMDc model to the provided data.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_features)
            Measurement data to be fit.
            Can be of shape (n_samples, n_features), or (n_trials, n_samples,
                n_features), where
                n_trials is the number of independent trials.
            Can also be of a list of arrays, where each array is a trajectory
                or a 2- or 3-d array of trajectories, provided they have the
                same last dimension.

        y : numpy.ndarray, shape (n_samples, n_features), default=None
            Measurement data output to be fitted.

        u : numpy.ndarray, shape (n_samples, n_control_features), optional, default=None
            Time series of external actuation/control.

        dt : float, optional
            Time interval between `X` and `Y`

        Returns
        -------
        self: returns a fitted ``DMDc`` instance
        """

        if y is None:
            X1, X2 = self._detect_reshape(x)
        else:
            X1, _ = self._detect_reshape(x, offset=False)
            X2, _ = self._detect_reshape(y, offset=False)
        if u is not None:
            offset = u.shape[0] > X1.shape[0]
            u, _ = self._detect_reshape(u, offset=offset)
        self.n_control_features_ = u.shape[1]
        self.n_input_features_ = X1.shape[1]
        C = u

        self.n_control_features_ = C.shape[1]

        if self.svd_rank is None:
            self.svd_rank = self.n_input_features_ + self.n_control_features_
            if self.svd_output_rank is None:
                self.svd_output_rank = self.n_input_features_
        else:
            if self.svd_output_rank is None:
                self.svd_output_rank = self.svd_rank

        rout = self.svd_output_rank
        r = self.svd_rank

        if self._input_control_matrix is None:
            self._fit_unknown_B(X1, X2, C, r, rout)
        else:
            self._fit_known_B(X1, X2, C, r)

        return self

    def _fit_unknown_B(self, X1, X2, C, r, rout):
        """
        Fits the DMDc model when the control matrix B is unknown. It computes
        the state matrix `A` and control matrix `B` using the Dynamic Mode
        Decomposition with control (DMDc) algorithm.

        Args:
            X1 (numpy.ndarray): The state matrix at time t.
            X2 (numpy.ndarray): The state matrix at time t+1.
            C (numpy.ndarray): The control input matrix.
            r (int): Rank for truncation of singular value decomposition.
            rout (int): Rank for truncation of singular value decomposition on X2
                transpose.

        Returns:
            None. Updates the instance variables _state_matrix_, _control_matrix_,
            _coef_, _eigenvectors_, _eigenvalues_, _ur, _tmp_compute_psi,
            _unnormalized_modes.

        Raises:
            ValueError: If the dimensions of X1, X2, and C are not compatible.
        """

        assert rout <= r
        Omega = np.vstack([X1.T, C.T])
        # SVD of input space
        U, s, Vh = np.linalg.svd(Omega, full_matrices=False)
        Ur = U[:, 0:r]
        Sr = np.diag(s[0:r])
        Vr = Vh[0:r, :].T

        Uhat, _, _ = np.linalg.svd(X2.T, full_matrices=False)
        Uhatr = Uhat[:, 0:rout]

        U1 = Ur[: self.n_input_features_, :]
        U2 = Ur[self.n_input_features_ :, :]

        # this is reduced A_r
        self._state_matrix_ = Uhatr.T @ X2.T @ Vr @ np.linalg.inv(Sr) @ U1.T @ Uhatr
        self._control_matrix_ = Uhatr.T @ X2.T @ Vr @ np.linalg.inv(Sr) @ U2.T

        # self._state_matrix_ = self._reduced_state_matrix_
        # self._control_matrix_ = self._reduced_control_matrix_
        # self._state_matrix_ = Uhatr @ self._reduced_state_matrix_ @ Uhatr.T
        # self._control_matrix_ = Uhatr @ self._reduced_control_matrix_

        # pack [A full, B full] as self.coef_
        self._coef_ = np.concatenate(
            (self._state_matrix_, self._control_matrix_), axis=1
        )

        # self._projection_matrix_ = Ur
        # self._projection_matrix_output_ = Uhatr

        # eigenvectors, lamda
        [self._eigenvalues_, self._eigenvectors_] = np.linalg.eig(self._state_matrix_)

        # Koopman modes V
        self._unnormalized_modes = Uhatr @ self._eigenvectors_
        self._ur = Uhatr
        self._tmp_compute_psi = np.linalg.inv(self._eigenvectors_) @ Uhatr.T

    def _fit_known_B(self, X1, X2, C, r):
        """
        Fits the DMDc model when the control matrix B is known. It computes
        the state matrix `A` using the Dynamic Mode Decomposition with control
        (DMDc) algorithm.

        Args:
            X1 (numpy.ndarray): The state matrix at time t.
            X2 (numpy.ndarray): The state matrix at time t+1.
            C (numpy.ndarray): The control input matrix.
            r (int): Rank for truncation of singular value decomposition.

        Returns:
            None. Updates the instance variables _state_matrix_, _coef_,
            _eigenvectors_, _eigenvalues_, _ur, _tmp_compute_psi, _unnormalized_modes.

        Raises:
            ValueError: If the dimensions of X1, X2, and C are not compatible.
        """
        if self.n_input_features_ in self._input_control_matrix.shape is False:
            raise TypeError("Control vector/matrix B has wrong shape.")
        if self._input_control_matrix.shape[1] == self.n_input_features_:
            self._input_control_matrix = self._input_control_matrix.T
        if self._input_control_matrix.shape[1] != self.n_control_features_:
            raise TypeError(
                "The control matrix B must have the same "
                "number of inputs as the control variable u."
            )

        U, s, Vh = np.linalg.svd(X1.T, full_matrices=False)
        Ur = U[:, :r]
        sr = s[:r]
        Vhr = Vh[:r, :]

        self._state_matrix_ = np.linalg.multi_dot(
            [
                Ur.T,
                X2.T - self._input_control_matrix @ C.T,
                Vhr.T,
                np.diag(np.reciprocal(sr)),
            ]
        )
        self._control_matrix_ = Ur.T @ self._input_control_matrix
        # self._state_matrix_ = Ur @ self._reduced_state_matrix_ @ Ur.T

        self._coef_ = np.concatenate(
            (self._state_matrix_, self.control_matrix_), axis=1
        )
        # self._coef_ = Ur @ self._state_matrix_ @ Ur.T
        # self._projection_matrix_ = Ur
        # self._projection_matrix_output_ = Ur

        # Compute , eigenvectors, lamda
        [self._eigenvalues_, self._eigenvectors_] = np.linalg.eig(self._state_matrix_)

        # Koopman V
        self._unnormalized_modes = Ur @ self._eigenvectors_
        self._ur = Ur
        self._tmp_compute_psi = np.linalg.inv(self._eigenvectors_) @ Ur.T

        # compute psi
        # self.C = np.linalg.inv(self._eigenvectors_) @ Ur.T

    def predict(self, x, u):
        """
        Predicts the future state of the system based on the current state and the
        current value of control input, using the fitted DMDc model.

        Args:
            x (numpy.ndarray): The current state of the system.
            u (numpy.ndarray): The current value of the input.

        Returns:
            numpy.ndarray: The predicted future state of the system.

        Raises:
            NotFittedError: If the model is not fitted, raise this error to prevent
                misuse of the model.
        """
        check_is_fitted(self, "coef_")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        u, _ = self._detect_reshape(u, offset=False)
        x, _ = self._detect_reshape(x, offset=False)
        # y = self.coef_ @ np.vstack([x.reshape(1, -1).T, u.reshape(1, -1).T])
        y = (
            x @ self.ur @ self.state_matrix_.T @ self.ur.T
            + u @ self.control_matrix_.T @ self.ur.T
        )
        # y = x @ self.state_matrix_.T + u @ self.control_matrix_.T
        # y = y.T
        y = self.return_orig_shape(y)
        return y

    def _compute_phi(self, x_col):
        """
        Returns the transformed matrix `phi(x)` given `x`.

        The method takes a column vector or a 1-D numpy array and computes its
        transformation using the `_ur` matrix. If the input `x_col` is a 1-D array,
        it reshapes it into a column vector before the computation.

        Args:
            x_col (numpy.ndarray): A column vector or a 1-D numpy array
                representing `x`.

        Returns:
            numpy.ndarray: The transformed matrix `phi(x)`.
        """
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        phi = self._ur.T @ x_col
        return phi

    def _compute_psi(self, x_col):
        """
        Returns `psi(x)` given `x`

        Args:
            x: numpy.ndarray, shape (n_samples, n_features)
                Measurement data upon which to compute psi values.

        Returns
            phi : numpy.ndarray, shape (n_samples, n_input_features_) value of
            Koopman psi at x
        """

        # compute psi - one column if x is a row
        if x_col.ndim == 1:
            x_col = x_col.reshape(-1, 1)
        psi = self._tmp_compute_psi @ x_col
        return psi

    @property
    def coef_(self):
        """
        The weight vectors of the regression problem.

        This method checks if the regressor is fitted before returning the coefficient.

        Returns:
            numpy.ndarray: The coefficient matrix.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
        """
        The DMD state transition matrix.

        This method checks if the regressor is fitted before returning the state matrix.

        Returns:
            numpy.ndarray: The state transition matrix.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_state_matrix_")
        return self._state_matrix_

    @property
    def control_matrix_(self):
        check_is_fitted(self, "_control_matrix_")
        return self._control_matrix_

    # @property
    # def reduced_state_matrix_(self):
    #     check_is_fitted(self, "_reduced_state_matrix_")
    #     return self._reduced_state_matrix_
    #
    # @property
    # def reduced_control_matrix_(self):
    #     check_is_fitted(self, "_reduced_control_matrix_")
    #     return self._reduced_control_matrix_

    @property
    def eigenvectors_(self):
        """
        The identified Koopman eigenvectors.

        This method checks if the regressor is fitted before returning the eigenvectors.

        Returns:
            numpy.ndarray: The Koopman eigenvectors.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def eigenvalues_(self):
        """
        The identified Koopman eigenvalues.

        This method checks if the regressor is fitted before returning the eigenvalues.

        Returns:
            numpy.ndarray: The Koopman eigenvalues.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def unnormalized_modes(self):
        """
        The raw DMD V with each column as one DMD mode.

        This method checks if the regressor is fitted before returning the unnormalized
            modes.

        Returns:
            numpy.ndarray: The unnormalized modes.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        """
        The left singular vectors 'U'.

        This method checks if the regressor is fitted before returning 'U'.

        Returns:
            numpy.ndarray: The left singular vectors 'U'.

        Raises:
            NotFittedError: If the regressor is not fitted yet.
        """
        check_is_fitted(self, "_ur")
        return self._ur

    @property
    def input_control_matrix(self):
        return self._input_control_matrix
