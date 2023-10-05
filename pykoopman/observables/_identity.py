"""module for Linear observables"""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class Identity(BaseObservables):
    """
    A dummy observables class that simply returns its input.
    """

    def __init__(self):
        """
        Initialize the Identity class.

        This constructor initializes the Identity class which simply returns its input
        when transformed.
        """
        super().__init__()
        self.include_state = True

    def fit(self, x, y=None):
        """
        Fit the model to the provided measurement data.

        Args:
            x (array-like): The measurement data to be fit. It must have a shape of
                (n_samples, n_input_features).
            y (None): This parameter is retained for sklearn compatibility.

        Returns:
            self: Returns a fit instance of the class `pykoopman.observables.Identity`.
        """
        x = validate_input(x)
        self.n_input_features_ = self.n_output_features_ = x.shape[1]
        self.n_consumed_samples = 0

        self.measurement_matrix_ = np.eye(x.shape[1]).T
        return self

    def transform(self, x):
        """
        Apply Identity transformation to the provided data.

        Args:
            x (array-like): The measurement data to be transformed. It must have a
                shape of (n_samples, n_input_features).

        Returns:
            array-like: Returns the transformed data which is the same as the input
                data in this case.
        """
        check_is_fitted(self, "n_input_features_")
        return x

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        Args:
            input_features (list of string, optional): The string names for input
                features, if available. By default, the names "x0", "x1", ... ,
                "xn_input_features" are used.

        Returns:
            list of string: Returns the output feature names.
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
        return input_features
