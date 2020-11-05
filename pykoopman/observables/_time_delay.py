"""
Time-delay observables
"""
from numpy import arange
from numpy import empty
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class TimeDelay(BaseObservables):
    r"""
    Time-delay observables. Observables formed by taking time-lagged
    measurements of state variables and interpreting them as new state
    variables.

    For example, the two state variables :math:`[x(t), y(t)]` could be
    supplemented with two time-delays each, yielding a new set of
    observables:

    .. math::
        [x(t), y(t), x(t-\Delta t), y(t-\Delta t),
        x(t-2\Delta t), y(t - 2\Delta t)]

    This example corresponds to taking :code:`delay =` :math:`\Delta t`
    and :code:`n_delays = 2`.

    Note that when transforming data the first
    :code:`delay * n_delays` rows/samples are dropped as there is
    insufficient time history to form time-delays for them.

    See the following references for more information.

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


    Parameters
    ----------
    delay: nonnegative integer, optional (default 1)
        The length of each delay.

    n_delays: nonnegative integer, optional (default 2)
        The number of delays to compute for each variable.

    Attributes
    ----------
    n_input_features_ : int
        Number of input features.

    n_output_features_ : int
        Number of output features.

    _n_consumed_samples : int
        Number of samples consumed when :code:`transform` is called,
        i.e. :code:`n_delays * delay`.
    """

    def __init__(self, delay=1, n_delays=2):
        super(TimeDelay, self).__init__()
        if delay < 0:
            raise ValueError("delay must be a nonnegative int")
        if n_delays < 0:
            raise ValueError("n_delays must be a nonnegative int")

        self.delay = int(delay)
        self.n_delays = int(n_delays)
        self._n_consumed_samples = self.delay * self.n_delays

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
        self: returns a fit :class:`pykoopman.observables.TimeDelay` instance
        """
        n_samples, n_features = validate_input(x).shape

        self.n_input_features_ = n_features
        self.n_output_features_ = n_features * (1 + self.n_delays)

        return self

    def transform(self, x):
        """
        Add time-delay features to the data, dropping the first
        :code:`delay - n_delays` samples.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be transformed.
            It is assumed that rows correspond to examples that are equi-spaced
            in time and are in sequential order.

        Returns
        -------
        y: array-like, shape (n_samples - delay * n_delays, n_output_features)
            Transformed data. Note that the number of output examples is,
            in general, different from the number input. In particular,
            :code:`n_samples - delay * n_delays`
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

    def inverse(self, y):
        """
        Invert the transformation.

        This function satisfies
        :code:`self.inverse(self.transform(x)) == x[self._n_consumed_samples:]`

        Parameters
        ----------
        y: array-like, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: array-like, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
        """
        check_is_fitted(self, "n_input_features_")
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "Wrong number of input features."
                f"Expected y.shape[1] = {self.n_out_features_}; "
                f"instead y.shape[1] = {y.shape[1]}."
            )

        # The first n_input_features_ columns correspond to the un-delayed
        # measurements
        return y[:, : self.n_input_features_]

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
        return index - self.delay * arange(1, self.n_delays + 1)

    @property
    def n_consumed_samples(self):
        """
        The number of samples that are consumed as "initial conditions" for
        other samples. I.e. the number of samples for which time delays cannot
        be computed.
        """
        return self._n_consumed_samples
