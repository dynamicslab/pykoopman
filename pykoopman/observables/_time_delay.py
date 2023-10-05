"""moduel for time-delay observables"""
from __future__ import annotations

import numpy as np
from numpy import arange
from numpy import empty
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class TimeDelay(BaseObservables):
    """
    A class for creating time-delay observables. These observables are formed by
    taking time-lagged measurements of state variables and interpreting them as new
    state variables.

    The two state variables :math:`[x(t), y(t)]` could be supplemented with two
    time-delays each, yielding a new set of observables:

    .. math::
        [x(t), y(t), x(t-\\Delta$ t), y(t-\\Delta t),
        x(t-2\\Delta t), y(t - 2\\Delta t)]

    This example corresponds to taking :code:`delay =` :math:`\\Delta t` and
    :code:`n_delays = 2`.

    Note that when transforming data the first :code:`delay * n_delays` rows/samples
    are dropped as there is insufficient time history to form time-delays for them.

    For more information, see the following references:

        Brunton, Steven L., et al.
        "Chaos as an intermittently forced linear system."
        Nature communications 8.1 (2017): 1-9.

        Susuki, Yoshihiko, and Igor Mezić.
        "A prony approximation of Koopman mode decomposition."
        2015 54th IEEE Conference on Decision and Control (CDC). IEEE, 2015.

        Arbabi, Hassan, and Igor Mezic.
        "Ergodic theory, dynamic mode decomposition, and computation
        of spectral properties of the Koopman operator."
        SIAM Journal on Applied Dynamical Systems 16.4 (2017): 2096-2126.

    Args:
        delay (int, optional): The length of each delay. Defaults to 1.
        n_delays (int, optional): The number of delays to compute for each
            variable. Defaults to 2.

    Attributes:
        include_state (bool): If True, includes the system state.
        delay (int): The length of each delay.
        n_delays (int): The number of delays to compute for each variable.
        _n_consumed_samples (int): Number of samples consumed when :code:`transform`
            is called,i.e. :code:`n_delays * delay`.
    """

    def __init__(self, delay=1, n_delays=2):
        """
        Initialize the TimeDelay class with given parameters.

        Args:
            delay (int, optional): The length of each delay. Defaults to 1.
            n_delays (int, optional): The number of delays to compute for each
                variable. Defaults to 2.

        Raises:
            ValueError: If delay or n_delays are negative.
        """
        super(TimeDelay, self).__init__()
        if delay < 0:
            raise ValueError("delay must be a nonnegative int")
        if n_delays < 0:
            raise ValueError("n_delays must be a nonnegative int")

        self.include_state = True
        self.delay = int(delay)
        self.n_delays = int(n_delays)
        self._n_consumed_samples = self.delay * self.n_delays

    def fit(self, x, y=None):
        """
        Fit the model to measurement data.

        Args:
            x (array-like): The input data, shape (n_samples, n_input_features).
            y (None): Dummy parameter for sklearn compatibility.

        Returns:
            TimeDelay: The fitted instance.
        """

        x = validate_input(x)
        n_samples, n_features = x.shape

        self.n_input_features_ = n_features
        self.n_output_features_ = n_features * (1 + self.n_delays)

        self.measurement_matrix_ = np.zeros(
            (self.n_input_features_, self.n_output_features_)
        )
        self.measurement_matrix_[
            : self.n_input_features_, : self.n_input_features_
        ] = np.eye(self.n_input_features_)

        return self

    def transform(self, x):
        """
        Add time-delay features to the data, dropping the first :code:`delay -
        n_delays` samples.

        Args:
            x (array-like): The input data, shape (n_samples, n_input_features).
                It is assumed that rows correspond to examples that are equi-spaced
                in time and are in sequential order.

        Returns:
            y (array-like): The transformed data, shape (n_samples - delay * n_delays,
            n_output_features).
        """

        check_is_fitted(self, "n_input_features_")
        x = validate_input(x)

        if x.shape[1] != self.n_input_features_:
            raise ValueError(
                "Wrong number of input features. "
                f"Expected x.shape[1] = {self.n_input_features_}; "
                f"instead x.shape[1] = {x.shape[1]}."
            )

        self._n_consumed_samples = self.delay * self.n_delays
        if len(x) < self._n_consumed_samples + 1:
            raise ValueError(
                "x has too few rows. To compute time-delay features with "
                f"delay = {self.delay} and n_delays = {self.n_delays} "
                f"x must have at least {self._n_consumed_samples + 1} rows."
            )

        y = empty(
            (x.shape[0] - self._n_consumed_samples, self.n_output_features_),
            dtype=x.dtype,
        )
        y[:, : self.n_input_features_] = x[self._n_consumed_samples :]

        for i in range(self._n_consumed_samples, x.shape[0]):
            y[i - self._n_consumed_samples, self.n_input_features_ :] = x[
                self._delay_inds(i), :
            ].flatten()

        return y

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        Args:
            input_features (list of str, optional): Names for input features.
                If None, defaults to "x0", "x1", ... ,"xn_input_features".

        Returns:
            list of str: Names of the output features.
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

        output_features = [f"{xi}(t)" for xi in input_features]
        output_features.extend(
            [
                f"{xi}(t-{i * self.delay}dt)"
                for i in range(1, self.n_delays + 1)
                for xi in input_features
            ]
        )

        return output_features

    def _delay_inds(self, index):
        """
        Private method to get the indices for the delayed data.

        Args:
            index (int): The index from which to calculate the delay indices.

        Returns:
            array-like: The delay indices.
        """
        return index - self.delay * arange(1, self.n_delays + 1)

    @property
    def n_consumed_samples(self):
        """
        The number of samples that are consumed as "initial conditions" for
        other samples, i.e., the number of samples for which time delays cannot
        be computed.

        Returns:
            int: The number of consumed samples.
        """
        return self._n_consumed_samples
