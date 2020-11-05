"""
Linear observables
"""
from itertools import combinations
from itertools import combinations_with_replacement

from numpy import empty
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class CustomObservables(BaseObservables):
    """
    Custom observable functions.

    To ensure invertibility of the set of observables, the identity map is
    automatically included along with user-specified observables.

    Parameters
    ----------
    observables: list of callable
        A list of functions mapping from state variables to observables.
        Univariate functions are applied to each state variable in turn and
        multivariable functions are applied to each combination of state
        variables.
        Note that the identity map is automatically prepended to this list.

    observable_names: list of callable, optional (default None)
        A list of functions mapping from names of state variables to names
        of observables. For example, the observable name
        :code:`lambda x: f"{x}^2"` would correspond to the function :math:`x^2`.
        If None, the names "f0(...)", "f1(...)", ... will be used.

    interaction_only: bool, optional (default True)
        Whether to omit self-interaction terms.
        If True, function evaluations of the form :math:`f(x,x)` and :math:`f(x,y,x)`
        will be omitted, but those of the form :math:`f(x,y)` and :math:`f(x,y,z)`
        will be included.
        If False, all combinations will be included.

    Attributes
    ----------
    n_input_features_ : int
        Number of input features.

    n_output_features_ : int
        Number of output features.
    """

    def __init__(self, observables, observable_names=None, interaction_only=True):
        super(CustomObservables, self).__init__()
        self.observables = [identity, *observables]
        if observable_names and (len(observables) != len(observable_names)):
            raise ValueError(
                "observables and observable_names must have the same length"
            )
        self.observable_names = observable_names
        self.interaction_only = interaction_only

    def fit(self, x, y=None):
        """
        Fit to measurement data.

        Determines the number of input and output features and creates
        default values for :code:`observable_names` if necessary.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be fit.

        y: None
            Dummy parameter retained for sklearn compatibility.

        Returns
        -------
        self: a fit :class:`pykoopman.observables.CustomObservables` instance
        """
        n_samples, n_features = validate_input(x).shape

        n_output_features = 0
        for f in self.observables:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )

        self.n_input_features_ = n_features
        self.n_output_features_ = n_output_features

        if self.observable_names is None:
            self.observable_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(len(self.observables)),
                )
            )
        # First map is the identity
        self.observable_names.insert(0, identity_name)

        return self

    def transform(self, x):
        """
        Apply custom transformations to data, computing observables.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be transformed.

        Returns
        -------
        y: array-like, shape (n_samples, n_output_features)
            Transformed data (observables).
        """
        check_is_fitted(self, "n_input_features_")
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

    def inverse(self, y):
        """
        Invert the transformation.

        This function satisfies
        :code:`self.inverse(self.transform(x)) == x`

        Parameters
        ----------
        y: array-like, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: array-like, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
            In this case, x is identical to y.
        """
        y = validate_input(y)
        check_is_fitted(self, "n_input_features_")
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "Wrong number of input features."
                f"Expected y.shape[1] = {self.n_out_features_}; "
                f"instead y.shape[1] = {y.shape[1]}."
            )
        # self.observables[0] is the identity, so original features are
        # just the first self.n_input_features_ columns
        return y[:, : self.n_input_features_]

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        Parameters
        ----------
        input_features: list of string, length n_input_features, \
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
        """Get the combinations of features to be passed to observable functions."""
        comb = combinations if interaction_only else combinations_with_replacement
        return comb(range(n_features), n_args)


def identity(x):
    """Identity map."""
    return x


def identity_name(x):
    """Name for identity map."""
    return str(x)
