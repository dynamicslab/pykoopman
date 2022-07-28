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
    svd_rank : int, optional (default None)
        SVD rank of the input data (x,u), which determines the dimensionality
        of the projected state and control matrices.

    svd_output_rank : int, optional (default 0)
        Input and output spaces may vary.

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
    >>> DMDc = pk.regression.DMDc(svd_rank=3, control_matrix=B)
    >>> model = pk.Koopman(regressor=DMDc)
    >>> model.fit(x,C)
    >>> Aest = model.state_transition_matrix
    >>> Best = model.control_matrix
    >>> print(Aest)
    >>> np.allclose(A,Aest)
    [[ 1.50000000e+00 -1.36609474e-17]
     [-1.58023594e-17  1.00000000e-01]]
    True

    For unknown B
    >>> DMDc = pk.regression.DMDc(svd_rank=3)
    >>> model = pk.Koopman(regressor=DMDc)
    >>> model.fit(x,C)
    >>> Aest = model.state_transition_matrix
    >>> Best = model.control_matrix
    >>> print(Aest)
    >>> print(Best)
    >>> np.allclose(np.concatenate((A,B),axis=1),np.concatenate((Aest,Best),axis=1))
    [[ 1.5000000e+00  4.6891744e-17]
     [-1.3259342e-17  1.0000000e-01]]
    [[1.00000000e+00]
     [6.88569357e-18]]
    True
    """

    def __init__(self, svd_rank=None, svd_output_rank=None, control_matrix=None):
        self.svd_rank = svd_rank
        self.svd_output_rank = svd_output_rank
        self.control_matrix_ = control_matrix

    def fit(self, x, u, y=None, dt=None):
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
        # if dt is None:
        #     self.time_ = dict([ ('tstart', 0),
        #                         ('tend', self.n_samples_ - 1),
        #                         ('dt', 1)])
        # else:
        #     self.time_ = dict([('tstart', 0),
        #                        ('tend', dt*(self.n_samples_ - 1)),
        #                        ('dt', dt)])

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

        if self.svd_rank is None:
            self.svd_rank = self.n_input_features_ + self.n_control_features_
        r = self.svd_rank

        if self.svd_output_rank is None:
            self.svd_output_rank = self.n_input_features_
        rout = self.svd_output_rank

        if self.control_matrix_ is None:
            self.fit_unknown_B(X1, X2, C, r, rout)

        else:
            self.fit_known_B(X1, X2, C, r)
        return self

    def fit_unknown_B(self, X1, X2, C, r, rout):
        Omega = np.vstack([X1.T, C.T])

        # SVD of input space
        U, s, Vh = np.linalg.svd(Omega, full_matrices=False)
        Ur = U[:, 0:r]
        Sr = np.diag(s[0:r])
        Vr = Vh[0:r, :].T

        # SVD of output space
        if rout is not self.n_input_features_:
            Uhat, _, _ = np.linalg.svd(X2.T, full_matrices=False)
            Uhatr = Uhat[:, 0:rout]
        else:
            Uhatr = np.identity(self.n_input_features_)

        U1 = Ur[: self.n_input_features_, :]
        U2 = Ur[self.n_input_features_ :, :]
        self.state_matrix_ = np.dot(
            Uhatr.T,
            np.dot(X2.T, np.dot(Vr, np.dot(np.linalg.inv(Sr), np.dot(U1.T, Uhatr)))),
        )
        self.control_matrix_ = np.dot(
            Uhatr.T, np.dot(X2.T, np.dot(Vr, np.dot(np.linalg.inv(Sr), U2.T)))
        )
        G = np.concatenate((self.state_matrix_, self.control_matrix_), axis=1)

        self.coef_ = G
        self.projection_matrix_ = Ur
        self.projection_matrix_output_ = Uhatr

        # Compute Koopman modes, eigenvectors, eigenvalues
        [self.eigenvalues_, self.eigenvectors_] = np.linalg.eig(self.state_matrix_)
        self.modes_ = np.dot(
            X2.T,
            np.dot(
                Vr,
                np.dot(
                    np.linalg.inv(Sr), np.dot(U1.T, np.dot(Uhatr, self.eigenvectors_))
                ),
            ),
        )

    def fit_known_B(self, X1, X2, C, r):
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
        Ur = np.identity(self.n_input_features_)
        Uhatr = np.identity(self.n_input_features_)

        self.coef_ = G
        self.projection_matrix_ = Ur
        self.projection_matrix_output_ = Uhatr

        # Compute Koopman modes, eigenvectors, eigenvalues
        [self.eigenvalues_, self.eigenvectors_] = np.linalg.eig(self.state_matrix_)
        self.modes_ = np.dot(
            X2.T, np.dot(Vh.T * (s ** (-1)), np.dot(U.T, self.eigenvectors_))
        )

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
        y = np.dot(self.state_matrix_, x.T) + np.dot(self.control_matrix_, u.T)
        return y.T

    # @property
    # def frequencies_(self):
    #     """
    #     Oscillation frequencies of Koopman modes/eigenvectors
    #     """
    #     check_is_fitted(self, "coef_")
    #     dt = self.time_['dt']
    #     return np.imag(np.log(self.eigenvalues_)/dt)/(2*np.pi)

    # @property
    # def eigenvalues_continuous_(self):
    #     """
    #     Continuous-time Koopman eigenvalues obtained from spectral decomposition of
    #     the Koopman matrix
    #     """
    #     check_is_fitted(self, "coef_")
    #     dt = self.time_['dt']
    #     return np.log(self.eigenvalues_) / dt

    # TODO: function to set time information --> in Koopman, here not necessary
