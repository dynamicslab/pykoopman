"""module for Radial basis function observables"""
from __future__ import annotations

import numpy as np
from numpy import empty
from numpy import random
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class RadialBasisFunction(BaseObservables):
    """
    This class represents Radial Basis Functions (RBF) used as observables.
    Observables are formed as RBFs of the state variables, interpreted as new state
    variables.

    For instance, a single state variable :math:`[x(t)]` could be evaluated using
    multiple centers, yielding a new set of observables. This implementation supports
    various types of RBFs including 'gauss', 'thinplate', 'invquad', 'invmultquad',
    and 'polyharmonic'.

    Attributes:
        rbf_type (str): The type of radial basis functions to be used.
        n_centers (int): The number of centers to compute RBF with.
        centers (numpy array): The centers to compute RBF with.
        kernel_width (float): The kernel width for Gaussian RBFs.
        polyharmonic_coeff (float): The polyharmonic coefficient for polyharmonic RBFs.
        include_state (bool): Whether to include the input coordinates as additional
            coordinates in the observable.
        n_input_features_ (int): Number of input features.
        n_output_features_ (int): Number of output features = Number of centers plus
            number of input features.

    Note:
        The implementation is based on the following references:
        - Williams, Matthew O and Kevrekidis, Ioannis G and Rowley, Clarence W
          "A data-driven approximation of the {K}oopman operator: extending dynamic
          mode decomposition."
          Journal of Nonlinear Science 6 (2015): 1307-1346
        - Williams, Matthew O and Rowley, Clarence W and Kevrekidis, Ioannis G
          "A Kernel Approach to Data-Driven {K}oopman Spectral Analysis."
          Journal of Computational Dynamics 2.2 (2015): 247-265
        - Korda, Milan and Mezic, Igor
          "Linear predictors for nonlinear dynamical systems: Koopman operator meets
          model predictive control."
          Automatica 93 (2018): 149-160
    """

    def __init__(
        self,
        rbf_type="gauss",
        n_centers=10,
        centers=None,
        kernel_width=1.0,
        polyharmonic_coeff=1.0,
        include_state=True,
    ):
        super().__init__()
        if type(rbf_type) != str:
            raise TypeError("rbf_type must be a string")
        if type(n_centers) != int:
            raise TypeError("n_centers must be an int")
        if n_centers < 0:
            raise ValueError("n_centers must be a nonnegative int")
        if kernel_width < 0:
            raise ValueError("kernel_width must be a nonnegative float")
        if polyharmonic_coeff < 0:
            raise ValueError("polyharmonic_coeff must be a nonnegative float")
        if rbf_type not in [
            "thinplate",
            "gauss",
            "invquad",
            "invmultquad",
            "polyharmonic",
        ]:
            raise ValueError("rbf_type not of available type")
        if type(include_state) != bool:
            raise TypeError("include_states must be a boolean")
        if centers is not None:
            if int(n_centers) not in centers.shape():
                raise ValueError(
                    "n_centers is not equal to centers.shape[1]. "
                    "centers must be of shape (n_input_features, "
                    "n_centers). "
                )
        self.rbf_type = rbf_type
        self.n_centers = int(n_centers)
        self.centers = centers
        self.kernel_width = kernel_width
        self.polyharmonic_coeff = polyharmonic_coeff
        self.include_state = include_state

    def fit(self, x, y=None):
        """
        Initializes the RadialBasisFunction with specified parameters.

        Args:
            rbf_type (str, optional): The type of radial basis functions to be used.
                Options are: 'gauss', 'thinplate', 'invquad', 'invmultquad',
                'polyharmonic'. Defaults to 'gauss'.
            n_centers (int, optional): The number of centers to compute RBF with.
                Must be a non-negative integer. Defaults to 10.
            centers (numpy array, optional): The centers to compute RBF with.
                If provided, it should have a shape of (n_input_features, n_centers).
                Defaults to None, in which case the centers are uniformly distributed
                over input data.
            kernel_width (float, optional): The kernel width for Gaussian RBFs.
                Must be a non-negative float. Defaults to 1.0.
            polyharmonic_coeff (float, optional): The polyharmonic coefficient for
                polyharmonic RBFs. Must be a non-negative float. Defaults to 1.0.
            include_state (bool, optional): Whether to include the input coordinates
                as additional coordinates in the observable. Defaults to True.

        Raises:
            TypeError: If rbf_type is not a string, n_centers is not an int, or
                include_state is not a bool.
            ValueError: If n_centers, kernel_width or polyharmonic_coeff is negative,
                rbf_type is not of available type, or centers is provided but
                n_centers is not equal to centers.shape[1].
        """
        x = validate_input(x)
        n_samples, n_features = x.shape
        self.n_consumed_samples = 0

        self.n_samples_ = n_samples
        self.n_input_features_ = n_features
        if self.include_state is True:
            self.n_output_features_ = n_features * 1 + self.n_centers
        elif self.include_state is False:
            self.n_output_features_ = self.n_centers

        x = validate_input(x)

        if x.shape[1] != self.n_input_features_:
            raise ValueError(
                "Wrong number of input features. "
                f"Expected x.shape[1] = {self.n_input_features_}; "
                f"instead x.shape[1] = {x.shape[1]}."
            )

        if self.centers is None:
            # Uniformly distributed centers
            self.centers = random.rand(self.n_input_features_, self.n_centers)
            # Change range to range of input features' range
            for feat in range(self.n_input_features_):
                xminmax = self._minmax(x[:, feat])

                # Map to range [0,1]
                self.centers[feat, :] = (
                    self.centers[feat, :] - min(self.centers[feat, :])
                ) / (max(self.centers[feat, :]) - min(self.centers[feat, :]))
                # Scale to input features' range
                self.centers[feat, :] = (
                    self.centers[feat, :] * (xminmax[1] - xminmax[0]) + xminmax[0]
                )

        xlift = self._rbf_lifting(x)
        # self.measurement_matrix_ = x.T @ np.linalg.pinv(xlift.T)
        self.measurement_matrix_ = np.linalg.lstsq(xlift, x)[0].T

        return self

    def transform(self, x):
        """
        Apply radial basis function transformation to the data.

        Args:
            x (array-like): Measurement data to be transformed, with shape (n_samples,
                n_input_features). It is assumed that rows correspond to examples,
                which are not required to be equi-spaced in time or in sequential order.

        Returns:
            array-like: Transformed data, with shape (n_samples, n_output_features).

        Raises:
            NotFittedError: If the 'fit' method has not been called before the
                'transform' method.
            ValueError: If the number of features in 'x' does not match the number of
                input features expected by the transformer.
        """
        check_is_fitted(self, ["n_input_features_", "centers"])
        x = validate_input(x)

        if x.shape[1] != self.n_input_features_:
            raise ValueError(
                "Wrong number of input features. "
                f"Expected x.shape[1] = {self.n_input_features_}; "
                f"instead x.shape[1] = {x.shape[1]}."
            )

        y = self._rbf_lifting(x)
        return y

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        Args:
            input_features (list of str, optional): String names for input features,
                if available. By default, the names "x0", "x1", ... ,
                "xn_input_features" are used.

        Returns:
            list of str: Output feature names.

        Raises:
            NotFittedError: If the 'fit' method has not been called before the
                'get_feature_names' method.
            ValueError: If the length of 'input_features' does not match the number of
                input features expected by the transformer.
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

        output_features = []
        if self.include_state is True:
            output_features.extend([f"{xi}(t)" for xi in input_features])
        output_features.extend([f"phi(x(t)-c{i})" for i in range(self.n_centers)])
        return output_features

    def _rbf_lifting(self, x):
        """
        Internal method that performs Radial Basis Function (RBF) transformation.

        Args:
            x (numpy.ndarray): Input data of shape (n_samples, n_input_features)

        Returns:
            y (numpy.ndarray): Transformed data of shape (n_samples, n_output_features)

        Raises:
            ValueError: If 'rbf_type' is not one of the available types.

        Notes:
            This method should not be called directly. It is used internally by the
            'transform' method.
        """
        n_samples = x.shape[0]
        y = empty(
            (n_samples, self.n_output_features_),
            dtype=x.dtype,
        )

        y_index = 0
        if self.include_state is True:
            y[:, : self.n_input_features_] = x
            y_index = self.n_input_features_

        for index_of_center in range(self.n_centers):
            C = self.centers[:, index_of_center]
            r_squared = np.sum((x - C[np.newaxis, :]) ** 2, axis=1)

            match self.rbf_type:
                case "thinplate":
                    y_ = r_squared * np.log(np.sqrt(r_squared))
                    y_[np.isnan(y_)] = 0
                case "gauss":
                    y_ = np.exp(-self.kernel_width**2 * r_squared)
                case "invquad":
                    y_ = np.reciprocal(1 + self.kernel_width**2 * r_squared)
                case "invmultquad":
                    y_ = np.reciprocal(np.sqrt(1 + self.kernel_width**2 * r_squared))
                case "polyharmonic":
                    y_ = r_squared ** (self.polyharmonic_coeff / 2) * np.log(
                        np.sqrt(r_squared)
                    )
                case _:
                    # if none of the above cases match:
                    raise ValueError("provided rbf_type not available")

            y[:, y_index + index_of_center] = y_

        return y

    def _minmax(self, x):
        min_val = min(x)
        max_val = max(x)
        return (min_val, max_val)
