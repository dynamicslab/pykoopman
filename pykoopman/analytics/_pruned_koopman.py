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

    Lambda_ : numpy.ndarray
        The diagonal matrix that contains the selected eigenvalues

    original_model : Koopman
        An instance of `pykoopman.koopman.Koopman`

    C_ : numpy.ndarray
        The matrix that maps selected Koopman eigenfunctions
        back to the system state :math:`x = C \\phi`.
    """

    def __init__(self, model: Koopman, sweep_index: np.ndarray):
        # construct lambda
        self.sweep_index = sweep_index
        self.Lambda_ = np.diag(model.eigenvalues[self.sweep_index])
        self.original_model = model

    def refit_modes(self, x):
        """Refit the Koopman modes given data matrix `x`

        Parameters
        ----------
        x : numpy.ndarray
            Training data for refitting the Koopman modes

        Returns
        -------
        self : PrunedKoopman
        """

        eigenphi = self.compute_eigen_phi(x)
        result = np.linalg.lstsq(eigenphi, x)
        # print('refit residual = {}'.format(result[1]))
        self.C_ = result[0].T
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

        gnext = self.compute_eigen_phi(x) @ self.Lambda_
        xnext = self.compute_state_from_observables(gnext)
        return xnext

    def compute_eigen_phi(self, x):
        """Evaluate the selected eigenfunction at given state `x`

        Parameters
        ----------
        x : numpy.ndarray
            System state `x` in row-wise

        Returns
        -------
        eigenphi : numpy.ndarray
            Selected eigenfunctions' value at given state `x`
        """

        eigenphi_ori = self.original_model.compute_eigenfunction(x)
        eigenphi = eigenphi_ori[:, self.sweep_index]
        return eigenphi

    def compute_state_from_observables(self, g):
        """Inverse selected observables from system state

        We use the just refitted modes to achieve that.

        Parameters
        ----------
        g : numpy.ndarray
            Row-wise values of selected Koopman eigenfunctions

        Returns
        -------
        x : numpy.ndarray
            System state recovered
        """

        x = g @ self.C.T
        return x

    @property
    def Lambda(self):
        return self.Lambda_

    @property
    def C(self):
        return self.C_
