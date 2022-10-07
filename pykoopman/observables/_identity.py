"""
Linear observables
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseObservables


class Identity(BaseObservables):
    """
    A dummy observables class that simply returns its input.
    """

    def __init__(self):
        super().__init__()
        self.include_state = True

    def fit(self, x, y=None):
        """
        Fit to measurement data.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be fit.

        y: None
            Dummy parameter retained for sklearn compatibility.

        Returns
        -------
        self: a fit :class:`pykoopman.observables.Identity` instance
        """
        # TODO: validate input
        self.n_input_features_ = self.n_output_features_ = x.shape[1]
        self.n_consumed_samples = 0

        self.measurement_matrix_ = np.eye(x.shape[1]).T
        return self

    def transform(self, x):
        """
        Apply Identity transformation to data.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be transformed.

        Returns
        -------
        y: array-like, shape (n_samples, n_input_features)
            Transformed data (same as x in this case).
        """
        # TODO validate input
        check_is_fitted(self, "n_input_features_")
        return x

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        Parameters
        ----------
        input_features: list of string, length n_input_features,\
         optional (default None)
            String names for input features, if available. By default,
            the names "x0", "x1", ... ,"xn_input_features" are used.

        Returns
        -------
        output_feature_names: list of string, length n_ouput_features
            Output feature names.
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
