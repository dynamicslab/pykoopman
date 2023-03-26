"""module for pruning Koopman models"""
from __future__ import annotations

import numpy as np
from pykoopman.koopman import Koopman


class PrunedKoopman(object):
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

    def __init__(self, model: Koopman, sweep_index: np.ndarray):
        # construct lambda
        self.sweep_index = sweep_index
        self.lamda_ = np.diag(np.diag(model.lamda)[self.sweep_index])
        self.original_model = model

    def refit_modes(self, x):
        """Refit the Koopman V given data matrix `x`

        Parameters
        ----------
        x : numpy.ndarray
            Training data for refitting the Koopman V

        Returns
        -------
        self : PrunedKoopman
        """

        selected_eigenphi = self.selected_psi(x)
        result = np.linalg.lstsq(selected_eigenphi, x)
        # print('refit residual = {}'.format(result[1]))
        self.V_ = result[0].T
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

        gnext = self.selected_psi(x) @ self.lamda_
        xnext = self.compute_state_from_psi(gnext)
        return xnext

    def selected_psi(self, x):
        """Evaluate the selected psi at given state `x`

        Parameters
        ----------
        x : numpy.ndarray
            System state `x` in row-wise

        Returns
        -------
        eigenphi : numpy.ndarray
            Selected eigenfunctions' value at given state `x`
        """

        eigenphi_ori = self.original_model.psi(x).T
        eigenphi = eigenphi_ori[:, self.sweep_index]
        return eigenphi

    def compute_state_from_psi(self, g):
        """Inverse selected observables from system state

        We use the just refitted V to achieve that.

        Parameters
        ----------
        g : numpy.ndarray
            Row-wise values of selected Koopman eigenfunctions

        Returns
        -------
        x : numpy.ndarray
            System state recovered
        """

        x = g @ self.V.T
        return x

    @property
    def lamda(self):
        return self.lamda_

    @property
    def V(self):
        return self.V_
