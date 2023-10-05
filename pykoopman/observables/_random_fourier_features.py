"""module for random fourier features observables"""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class RandomFourierFeatures(BaseObservables):
    """
    Random Fourier Features for observables.

    This class applies the random Fourier features method for kernel approximation.
    It can include the system state in the kernel function. It uses the
    Gaussian kernel by default.

    Args:
        include_state (bool, optional): If True, includes the system state. Defaults to
            True.
        gamma (float, optional): The scale of the Gaussian kernel. Defaults to 1.0.
        D (int, optional): The number of random samples in Monte Carlo approximation.
            Defaults to 100.
        random_state (int, None, optional): The seed of the random number for repeatable
            experiments. Defaults to None.

    Attributes:
        include_state (bool): If True, includes the system state.
        gamma (float): The scale of the Gaussian kernel.
        D (int): The number of random samples in Monte Carlo approximation.
        random_state (int, None): The seed of the random number for repeatable
            experiments.
        measurement_matrix_ (numpy.ndarray): A row feature vector right multiply with
            `measurement_matrix_` will return the system state.
        n_input_features_ (int): Dimension of input features, e.g., system state.
        n_output_features_ (int): Dimension of transformed/output features, e.g.,
            observables.
        w (numpy.ndarray): The frequencies randomly sampled for random fourier features.
    """

    def __init__(self, include_state=True, gamma=1.0, D=100, random_state=None):
        """
        Initialize the RandomFourierFeatures class with given parameters.

        Args:
            include_state (bool, optional): If True, includes the system state.
                Defaults to True.
            gamma (float, optional): The scale of the Gaussian kernel. Defaults to 1.0.
            D (int, optional): The number of random samples in Monte Carlo
                approximation. Defaults to 100.
            random_state (int or None, optional): The seed of the random number
                for repeatable experiments. Defaults to None.
        """
        super(RandomFourierFeatures, self).__init__()
        self.include_state = include_state
        self.gamma = gamma
        self.D = D
        self.random_state = random_state

    def fit(self, x, y=None):
        """
        Set up observable.

        Args:
            x (numpy.ndarray): Measurement data to be fit. Shape (n_samples,
                n_input_features_).
            y (numpy.ndarray, optional): Time-shifted measurement data to be fit.
                Defaults to None.

        Returns:
            self: Returns a fitted RandomFourierFeatures instance.
        """
        x = validate_input(x)
        np.random.seed(self.random_state)
        self.n_consumed_samples = 0

        self.n_input_features_ = x.shape[1]
        # although we have double the output dim, the convergence
        # rate is described in only self.n_components
        self.n_output_features_ = 2 * self.D

        if self.include_state is True:
            self.n_output_features_ += self.n_input_features_

        # 1. generate (n_feature, n_component) random w
        self.w = np.sqrt(2.0 * self.gamma) * np.random.normal(
            0, 1, [self.n_input_features_, self.D]
        )

        # 3. get the C to map back to state
        if self.include_state:
            self.measurement_matrix_ = np.zeros(
                (self.n_input_features_, self.n_output_features_)
            )
            self.measurement_matrix_[
                : self.n_input_features_, : self.n_input_features_
            ] = np.eye(self.n_input_features_)
        else:
            # we have to transform the data x in order to find a matrix by fitting
            # z = np.zeros((x.shape[0], self.n_output_features_))
            # z[:,:x.shape[1]] = x
            # z[:,x.shape[1]:] = self._rff_lifting(x)
            z = self._rff_lifting(x)
            self.measurement_matrix_ = np.linalg.lstsq(z, x)[0].T

        return self

    def transform(self, x):
        """
        Evaluate observable at `x`.

        Args:
            x (numpy.ndarray): Measurement data to be fit. Shape (n_samples,
                n_input_features_).

        Returns:
            y (numpy.ndarray): Evaluation of observables at `x`. Shape (n_samples,
                n_output_features_).
        """

        check_is_fitted(self, "n_input_features_")
        z = np.zeros((x.shape[0], self.n_output_features_))
        z_rff = self._rff_lifting(x)
        if self.include_state:
            z[:, : x.shape[1]] = x
            z[:, x.shape[1] :] = z_rff
        else:
            z = z_rff

        return z

    def get_feature_names(self, input_features=None):
        """
        Return names of observables.

        Args:
            input_features (list of string of length n_features, optional):
                Default list is "x0", "x1", ..., "xn", where n = n_features.

        Returns:
            output_feature_names (list of string of length n_output_features):
                Returns a list of observable names.
        """

        check_is_fitted(self, "n_input_features_")

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]
        else:
            if len(input_features) != self.n_input_features_:
                raise ValueError(
                    "input_features must have n_input_features_ "
                    f"({self.n_input_features_}) elements"
                )

        if self.include_state:
            # very easy to make mistake... python pass list by reference OMG
            output_features = input_features[:]
        else:
            output_features = []
        output_features += [f"cos(w_{i}'x)/sqrt({self.D})" for i in range(self.D)] + [
            f"sin(w_{i}'x)/sqrt({self.D})" for i in range(self.D)
        ]

        return output_features

    def _rff_lifting(self, x):
        """
        Core algorithm that computes random Fourier features.

        This method uses the `cos` and `sin` transformations to get random Fourier
            features.

        Args:
            x (numpy.ndarray): System state.

        Returns:
            z_rff (numpy.ndarray): Random Fourier features evaluated on `x`. Shape
                (n_samples, n_output_features_).
        """

        # 2. get the feature vector z
        xw = np.dot(x, self.w)
        z_rff = np.hstack([np.cos(xw), np.sin(xw)])
        z_rff *= 1.0 / np.sqrt(self.D)
        return z_rff
