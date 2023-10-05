"""Module for customized observables"""
from __future__ import annotations

from itertools import combinations
from itertools import combinations_with_replacement

import numpy as np
from numpy import empty
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class CustomObservables(BaseObservables):
    """
    A class to map state variables using custom observables.

    This class allows the user to specify a list of functions that map state variables
    to observables. The identity map is automatically included. It can be configured to
    include or exclude self-interaction terms.

    Attributes:
        observables (list of callable): List of functions mapping state variables to
            observables. Univariate functions are applied to each state variable,
            and multivariable functions are applied to combinations of state
            variables. The identity map is automatically included in this list.
        observable_names (list of callable, optional): List of functions mapping from
            names of state variables to names of observables. For example,
            the observable name lambda x: f"{x}^2" would correspond to the function
            x^2. If None, the names "f0(...)", "f1(...)", ... will be used. Default
            is None.
        interaction_only (bool, optional): If True, omits self-interaction terms.
            Function evaluations of the form f(x,x) and f(x,y,x) will be omitted,
            but those of the form f(x,y) and f(x,y,z) will be included. If False,
            all combinations will be included. Default is True.
        n_input_features_ (int): Number of input features.
        n_output_features_ (int): Number of output features.
    """

    def __init__(self, observables, observable_names=None, interaction_only=True):
        """
        Initialize a CustomObservables instance.

        Args:
            observables (list of callable): List of functions mapping state variables
                to observables. Univariate functions are applied to each state
                variable, and multivariable functions are applied to combinations of
                state variables. The identity map is automatically included in this
                list.
            observable_names (list of callable, optional): List of functions mapping
                from names of state variables to names of observables. For example,
                the observable name lambda x: f"{x}^2" would correspond to the
                function x^2. If None, the names "f0(...)", "f1(...)", ... will
                be used. Default is None.
            interaction_only (bool, optional): If True, omits self-interaction terms.
                Function evaluations of the form f(x,x) and f(x,y,x) will be omitted,
                but those of the form f(x,y) and f(x,y,z) will be included. If False,
                all combinations will be included. Default is True.
        """
        super(CustomObservables, self).__init__()
        self.observables = [identity, *observables]
        if observable_names and (len(observables) != len(observable_names)):
            raise ValueError(
                "observables and observable_names must have the same length"
            )
        self.observable_names = observable_names
        self.interaction_only = interaction_only
        self.include_state = True

    def fit(self, x, y=None):
        """
        Fit the model to the measurement data.

        This method calculates the number of input and output features and generates
        default values for 'observable_names' if necessary. It also prepares the
        measurement matrix for data transformation.

        Args:
            x (array-like, shape (n_samples, n_input_features)): Measurement data to be
                fitted.
            y (None): This is a dummy parameter added for compatibility with sklearn's
                API. Default is None.

        Returns:
            self (CustomObservables): This method returns the fitted instance.
        """
        x = validate_input(x)
        n_samples, n_features = x.shape

        n_output_features = 0
        for f in self.observables:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )

        self.n_input_features_ = n_features
        self.n_output_features_ = n_output_features
        self.n_consumed_samples = 0

        if self.observable_names is None:
            self.observable_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(len(self.observables)),
                )
            )

        # First map is the identity
        self.observable_names.insert(0, identity_name)

        # since the first map is identity
        self.measurement_matrix_ = np.zeros(
            (self.n_input_features_, self.n_output_features_)
        )
        self.measurement_matrix_[
            : self.n_input_features_, : self.n_input_features_
        ] = np.eye(self.n_input_features_)

        return self

    def transform(self, x):
        """
        Apply custom transformations to data, computing observables.

        This method applies the user-defined observables functions to the input data,
        effectively transforming the state variables into observable ones.

        Args:
            x (array-like, shape (n_samples, n_input_features)): The measurement data
                to be transformed.

        Returns:
            x_transformed (array-like, shape (n_samples, n_output_features)): The
                transformed data, i.e., the computed observables.
        """
        check_is_fitted(self, "n_input_features_")
        check_is_fitted(self, "n_output_features_")
        x = validate_input(x)

        n_samples, n_features = x.shape

        if n_features != self.n_input_features_:
            raise ValueError("x.shape[1] does not match n_input_features_")

        x_transformed = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        observables_idx = 0
        for f in self.observables:
            for c in self._combinations(
                self.n_input_features_, f.__code__.co_argcount, self.interaction_only
            ):
                x_transformed[:, observables_idx] = f(*[x[:, j] for j in c])
                observables_idx += 1

        return x_transformed

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        This method returns the names of the output features as defined by the
        observable functions. If names for the input features are provided, they are
        used in the output feature names. Otherwise, default names ("x0", "x1", ...,
        "xn_input_features") are used.

        Args:
            input_features (list of string, length n_input_features, optional):
                String names for input features, if available. By default, the names
                "x0", "x1", ... ,"xn_input_features" are used.

        Returns:
            output_feature_names (list of string, length n_output_features):
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

        feature_names = []
        for i, f in enumerate(self.observables):
            feature_names.extend(
                [
                    self.observable_names[i](*[input_features[j] for j in c])
                    for c in self._combinations(
                        self.n_input_features_,
                        f.__code__.co_argcount,
                        self.interaction_only,
                    )
                ]
            )

        return feature_names

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """
        Get the combinations of features to be passed to observable functions.

        This static method generates all possible combinations or combinations with
        replacement (depending on the `interaction_only` flag) of features that are to
        be passed to the observable functions. The combinations are represented as
        tuples of indices.

        Args:
            n_features (int): The total number of features.
            n_args (int): The number of arguments that the observable function accepts.
            interaction_only (bool): If True, combinations of the same feature
                (self-interactions) are omitted. If False, all combinations including
                self-interactions are included.

        Returns:
            iterable of tuples: An iterable over all combinations of feature indices
            to be passed to the observable functions.
        """
        comb = combinations if interaction_only else combinations_with_replacement
        return comb(range(n_features), n_args)


def identity(x):
    """Identity map."""
    return x


def identity_name(x):
    """Name for identity map."""
    return str(x)
