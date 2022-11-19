from __future__ import annotations

import abc

import numpy as np

from pykoopman.koopman import Koopman


class BaseAnalyzer(object):
    """Base class for Koopman model analyzer.

    Aims to perform modes selection.

    Attributes
    ----------
    model : Koopman
        An instance of `pykoopman.koopman.Koopman`

    eigenfunction : Koopman.compute_eigenfunction
        A function that evaluates Koopman eigenfunction

    eigenvalues_cont : numpy.ndarray
        Koopman eigenvalues in continuous-time

    eigenvalues_discrete : numpy.ndarray
        Koopman eigenvalues in discrete-time
    """

    def __init__(self, model: Koopman):
        self.model = model
        self.eigenfunction = self.model.compute_eigenfunction
        self.eigenvalues_cont = self.model.eigenvalues_continuous
        self.eigenvalues_discrete = self.model.eigenvalues

    def _compute_phi_minus_phi_evolved(self, t, validate_data_one_traj):
        """Compute the difference between eigenfunction evolved and
        eigenfunction observed

        Parameters
        ----------
        t : numpy.ndarray
            Time stamp of this validation trajectory

        validate_data_one_traj : numpy.ndarray
            Data matrix of this validation trajectory

        Returns
        -------
        linear_residual_list : list
            returns the residual for each mode
        """

        phi = self.eigenfunction(validate_data_one_traj)
        linear_residual_list = []
        for i in range(len(self.eigenvalues_cont)):
            linear_residual_list.append(
                phi[:, i] - np.exp(self.eigenvalues_cont[i] * t) * phi[0:1, i]
            )
        return linear_residual_list

    def validate(self, t, validate_data_one_traj):
        """validate Koopman eigenfunction

        Given a single trajectory, compute the norm of the difference
        between observed eigenfunction and evolved eigenfunction for
        each mode.

              :math:`$\\| \\phi(x(t+q)) - \\lambda * e^{q * \\lambda} \\phi(x(t)) \\|$`

        Parameters
        ----------
        t : numpy.ndarray
            Time stamp of this validation trajectory

        validate_data_one_traj : numpy.ndarray
            Data matrix of this validation trajectory

        Returns
        -------
        linear_residual_norm_list : list
            returns the difference in norm for each mode
        """

        linear_residual_list = self._compute_phi_minus_phi_evolved(
            t, validate_data_one_traj
        )
        linear_residual_norm_list = [
            np.linalg.norm(tmp) for tmp in linear_residual_list
        ]
        return linear_residual_norm_list

    @abc.abstractmethod
    def prune_model(self, *params, **kwargs):
        raise NotImplementedError
