from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class DMDc(BaseRegressor):
    """
    DMD with control (DMDc) regressor.

    Aims to determine the system matrices A,B
    that satisfy x' = Ax + Bu, where x' is the time-shifted
    state w.r.t. x und u is the control input, for
    known and unknown B.

    Minimizes the objective function
    :math:`\\|X'-AX-BU\\|_F`
    using least-squares regression and singular value decomposition
    of the input [X,U] and output spaces [X'] to cope with
    high-dimensionality of X.

    See the following reference for more details:

        `Procter, Joshua L., Brunton, Steven L., and Kutz, J. Nathan.
        "Dynamic Mode Decomposition with Control."
        SIAM J. Appl. Dyn. Syst., 15(1), 142–161.
        <https://epubs.siam.org/doi/abs/10.1137/15M1013857?mobileUi=0>`_

    Parameters
    ----------
    svd_rank : int, optional, default=None
        SVD rank of the input data (x,u), which determines the dimensionality
        of the projected state and control matrices.

    svd_output_rank : int, optional, default=0
        Input and output spaces may vary.

    Attributes
    ----------
    coef_ : numpy.ndarray, shape (n_input_features_, n_input_features_) or
        (n_input_features_, n_input_features_ + n_control_features_)
        Weight vectors of the regression problem. Corresponds to either [A] or [A,B]

    state_matrix_ : numpy.ndarray, shape (n_input_features_, n_input_features_)
        Identified state transition matrix A of the underlying system.

    control_matrix_ : numpy.ndarray, shape (n_input_features_, n_control_features_)
        Identified control matrix B of the underlying system.

    reduced_state_matrix_ : numpy.ndarray, shape (svd_output_rank, svd_output_rank)
        Reduced state transition matrix

    reduced_control_matrix_ : numpy.ndarray, shape (svd_output_rank,
    n_control_features_)
        Reduced control matrix

    eigenvalues_ : numpy.ndarray, shape (svd_rank, )
        DMD lamda

    unnormalized_modes : numpy.ndarray, shape (svd_rank, svd_rank)
        DMD V

    projection_matrix_ : numpy.ndarray, shape (n_input_features_+
    n_control_features_, svd_rank)
        Projection matrix into low-dimensional subspace.

    projection_matrix_output_ : numpy.ndarray, shape (n_input_features_+
    n_control_features_, svd_output_rank)
        Projection matrix into low-dimensional subspace.

    eigenvectors_ : numpy.ndarray, shape (svd_rank, svd_rank)
        DMD eigenvectors

    n_control_features_ : int
        Dimension of input control signal

    n_input_features_ : int
        Dimension of input features

    n_samples_ : int
        Total number of one step evolution samples

    svd_rank : int
        SVD rank of the input data (x,u), which determines the dimensionality
        of the projected state and control matrices.

    svd_output_rank : int
        Input and output spaces may vary.

    Examples
    --------
    For known B
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

    For unknown B
    >>> DMDc = pk.regression.DMDc(svd_rank=3)
    >>> model = pk.Koopman(regressor=DMDc)
    >>> model.fit(x,C)
    >>> Aest = model.A
    >>> Best = model.B
    >>> print(Aest)
    >>> print(Best)
    >>> np.allclose(np.concatenate((A,B),axis=1),np.concatenate((Aest,Best),axis=1))
    [[ 1.5000000e+00  4.6891744e-17]
     [-1.3259342e-17  1.0000000e-01]]
    [[1.00000000e+00]
     [6.88569357e-18]]
    True
    """

    def __init__(self, svd_rank=None, svd_output_rank=None, input_control_matrix=None):
        self.svd_rank = svd_rank
        self.svd_output_rank = svd_output_rank
        self._input_control_matrix_ = input_control_matrix

    def fit(self, x, y=None, u=None, dt=None):
        """
        Parameters
        ----------
        x : numpy ndarray, shape (n_samples, n_features)
            Measurement data to be fit.

        y : numpy ndarray, shape (n_samples, n_features), default=None
            Measurement data output to be fitted

        u : numpy.ndarray, shape (n_samples, n_control_features), \
                optional, default=None
            Time series of external actuation/control.

        dt : float
            Time interval between `x` and `y`

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
        else:
            if u.shape[0] > X1.shape[0]:
                u = u[:-1, :]
            C = u
        self.n_control_features_ = C.shape[1]

        if self.svd_rank is None:
            self.svd_rank = self.n_input_features_ + self.n_control_features_
        r = self.svd_rank

        if self.svd_output_rank is None:
            self.svd_output_rank = self.n_input_features_
        rout = self.svd_output_rank

        if self._input_control_matrix_ is None:
            self._fit_unknown_B(X1, X2, C, r, rout)
        else:
            self._fit_known_B(X1, X2, C, r)

        return self

    def _fit_unknown_B(self, X1, X2, C, r, rout):
        """
        Parameters
        ----------
        X1 : numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to make prediction

        X2 : numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to match the prediction

        C : numpy.ndarray, shape (n_samples, n_control_features)
            Time series of external actuation/control.

        r : int
            Rank of SVD for the input space, i.e., the space of
            `X` and input `U`

        rout : int
            Rank of SVD for the output space, i.e., the space of
            `Y`
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
        Parameters
        ----------
        X1 : numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to make prediction

        X2 : numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to match the prediction

        C : numpy.ndarray, shape (n_samples, n_control_features)
            Time series of external actuation/control.

        r : int
            Rank of SVD for the input space, i.e., the space of
            `X` and input `U`
        """

        if self.n_input_features_ in self._input_control_matrix_.shape is False:
            raise TypeError("Control vector/matrix B has wrong shape.")
        if self._input_control_matrix_.shape[1] == self.n_input_features_:
            self._input_control_matrix_ = self._input_control_matrix_.T
        if self._input_control_matrix_.shape[1] != self.n_control_features_:
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
                X2.T - self._input_control_matrix_ @ C.T,
                Vhr.T,
                np.diag(np.reciprocal(sr)),
            ]
        )
        self._control_matrix_ = Ur.T @ self._input_control_matrix_
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
        Parameters
        ----------
        x : numpy ndarray, shape (n_samples, n_features)
            Measurement data upon which to base prediction.

        u : numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        Returns
        -------
        y: numpy ndarray, shape (n_samples, n_features)
            Prediction of x one timestep in the future.

        """
        check_is_fitted(self, "coef_")
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if u.ndim == 1:
            u = u.reshape(1, -1)
        # y = self.coef_ @ np.vstack([x.reshape(1, -1).T, u.reshape(1, -1).T])
        y = (
            x @ self.ur @ self.state_matrix_.T @ self.ur.T
            + u @ self.control_matrix_.T @ self.ur.T
        )
        # y = x @ self.state_matrix_.T + u @ self.control_matrix_.T
        # y = y.T
        return y

    def _compute_phi(self, x):
        """Returns `phi(x)` given `x`"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        phi = self._ur.T @ x.T
        return phi

    def _compute_psi(self, x):
        """Returns `psi(x)` given `x`

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_features)
            Measurement data upon which to compute psi values.

        Returns
        -------
        phi : numpy.ndarray, shape (n_samples, n_input_features_)
            value of Koopman psi at x
        """

        # compute psi - one column if x is a row
        if x.ndim == 1:
            x = x.reshape(1, -1)
        psi = self._tmp_compute_psi @ x.T
        return psi

    @property
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def state_matrix_(self):
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
        check_is_fitted(self, "_eigenvectors_")
        return self._eigenvectors_

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def unnormalized_modes(self):
        check_is_fitted(self, "_unnormalized_modes")
        return self._unnormalized_modes

    @property
    def ur(self):
        check_is_fitted(self, "_ur")
        return self._ur

    #
    # @property
    # def projection_matrix_(self):
    #     check_is_fitted(self, "_projection_matrix_")
    #     return self._projection_matrix_
    #
    # @property
    # def projection_matrix_output_(self):
    #     check_is_fitted(self, "_projection_matrix_output_")
    #     return self._projection_matrix_output_
