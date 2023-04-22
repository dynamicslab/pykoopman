"""module for pruning Koopman models"""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from pykoopman.koopman import Koopman


class PrunedKoopman:
    """Prune the given original Koopman `model` at `sweep_index`

    Parameters
    ----------
    model : Koopman
        An instance of `pykoopman.koopman.Koopman`

    sweep_index : numpy.ndarray
        selected indices in the original Koopman model

    Attributes
    ----------
    sweep_index : numpy.ndarray
        selected indices in the original Koopman model

    lamda_ : numpy.ndarray
        The diagonal matrix that contains the selected lamda

    original_model : Koopman
        An instance of `pykoopman.koopman.Koopman`

    V_ : numpy.ndarray
        The matrix that maps selected Koopman eigenfunctions
        back to the system state :math:`x = C \\phi`.
    """

    def __init__(self, model: Koopman, sweep_index: np.ndarray, dt):
        # construct lambda
        self.sweep_index = sweep_index
        # self.lamda_ = np.diag(np.diag(model.lamda)[self.sweep_index])
        self.original_model = model
        self.time = {"dt": dt}

        # no support for controllable for now
        if self.original_model.n_control_features_ > 0:
            raise NotImplementedError

        self.A_ = None

    def fit(self, x):
        """Fit the pruned model given data matrix `x`

        Parameters
        ----------
        x : numpy.ndarray
            Training data for refitting the Koopman V

        Returns
        -------
        self : PrunedKoopman
        """

        # pruned V
        selected_eigenphi = self.psi(x.T).T
        result = np.linalg.lstsq(selected_eigenphi, x)
        # print('refit residual = {}'.format(result[1]))
        self.W_ = result[0].T

        # lamda, W = np.linalg.eig(self.original_model.A)

        self.lamda_ = np.diag(np.diag(self.original_model.lamda)[self.sweep_index]) + 0j
        # evecs = self.original_model._regressor_eigenvectors

        return self

    def predict(self, x):
        """Predict system state at the next time stamp given `x`

        Parameters
        ----------
        x : numpy.ndarray
            System state `x` in row-wise

        Returns
        -------
        xnext : numpy.ndarray
            System state at the next time stamp
        """

        if x.ndim == 1:
            x = x.reshape(1, -1)
        gnext = self.lamda @ self.psi(x.T)
        # xnext = self.compute_state_from_psi(gnext)
        xnext = self.W @ gnext
        return np.real(xnext.T)

    def psi(self, x_col):
        """Evaluate the selected psi at given state `x`

        Parameters
        ----------
        x : numpy.ndarray
            System state `x` in column-wise

        Returns
        -------
        eigenphi : numpy.ndarray
            Selected eigenfunctions' value at given state `x`
        """

        # eigenphi_ori = self.original_model.psi(x_col).T
        # eigenphi_selected = eigenphi_ori[:, self.sweep_index]

        eigenphi_ori = self.original_model.psi(x_col)
        eigenphi_selected = eigenphi_ori[self.sweep_index]
        return eigenphi_selected

    def phi(self, x_col):
        # return self.original_model._regressor_eigenvectors @ self.psi(x_col)
        raise NotImplementedError("Pruned model does not have `phi` but only `psi`")

    @property
    def ur(self):
        raise NotImplementedError("Pruned model does not have `ur`")

    @property
    def A(self):
        raise NotImplementedError(
            "Pruning only happen in eigen-space. So no self.A " "but only self.lamda"
        )

    @property
    def B(self):
        raise NotImplementedError(
            "Pruning only for autonomous system rather than " "controlled system"
        )

    @property
    def C(self):
        return NotImplementedError("Pruning model does not have `C`")

    @property
    def W(self):
        check_is_fitted(self, "W_")
        return self.W_

    @property
    def lamda(self):
        return self.lamda_

    @property
    def lamda_array(self):
        return np.diag(self.lamda) + 0j

    @property
    def continuous_lamda_array(self):
        check_is_fitted(self, "_pipeline")
        return np.log(self.lamda_array) / self.time["dt"]

    # TODO: implement `simulate`, `validity_check`, `score`
    # TODO: implement sparsification for controlled case.
