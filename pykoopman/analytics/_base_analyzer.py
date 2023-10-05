"""module for implement modes analyzer for Koopman approximation"""
from __future__ import annotations

import abc

import numpy as np


class BaseAnalyzer(object):
    """Base class for Koopman model analyzer.

    Attributes:
        model (Koopman): An instance of `pykoopman.koopman.Koopman`.
        eigenfunction (Koopman.compute_eigenfunction): A function that evaluates Koopman
            psi.
        eigenvalues_cont (numpy.ndarray): Koopman lamda in continuous-time.
        eigenvalues_discrete (numpy.ndarray): Koopman lamda in discrete-time.
    """

    def __init__(self, model):
        """Initialize the BaseAnalyzer object.

        Args:
            model (Koopman): An instance of `pykoopman.koopman.Koopman`.
        """
        self.model = model
        self.eigenfunction = self.model.psi
        self.eigenvalues_cont = self.model.continuous_lamda_array
        self.eigenvalues_discrete = self.model.lamda_array

    def _compute_phi_minus_phi_evolved(self, t, validate_data_one_traj):
        """Compute the difference between psi evolved and psi observed.

        Args:
            t (numpy.ndarray): Time stamp of this validation trajectory.
            validate_data_one_traj (numpy.ndarray): Data matrix of this validation
                trajectory.

        Returns:
            list: Linear residual for each mode.
        """

        # shape of phi = (num_samples, num_modes)
        psi = self.eigenfunction(validate_data_one_traj.T).T

        linear_residual_list = []
        for i in range(len(self.eigenvalues_cont)):
            linear_residual_list.append(
                psi[:, i] - np.exp(self.eigenvalues_cont[i] * t) * psi[0:1, i]
            )
        return linear_residual_list

    def validate(self, t, validate_data_one_traj):
        """Validate Koopman psi.

        Given a single trajectory, compute the norm of the difference
        between observed psi and evolved psi for each mode.

        Args:
            t (numpy.ndarray): Time stamp of this validation trajectory.
            validate_data_one_traj (numpy.ndarray): Data matrix of this validation
                trajectory.

        Returns:
            list: Difference in norm for each mode.
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
        """Prune the model.

        This method should be implemented by the derived classes.

        Args:
            *params: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented by the derived class.
        """
        raise NotImplementedError
