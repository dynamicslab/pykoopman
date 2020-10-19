from warnings import catch_warnings
from warnings import filterwarnings
from warnings import warn

from numpy import empty
from numpy import vstack
from pydmd import DMD
from pydmd import DMDBase
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .common import validate_input
from .observables import Identity
from .observables import TimeDelay
from .regression import BaseRegressor
from .regression import DMDRegressor


class Koopman(BaseEstimator):
    """
    Discrete-Time Koopman class.

    Parameters
    ----------
    observables: observables object, optional \
            (default :class:`pykoopman.observables.Identity`)
        Map(s) to apply to raw measurement data before estimating the
        Koopman operator.
        Must extend :class:`pykoopman.observables.BaseObservables`.
        The default option, :class:`pykoopman.observables.Identity` leaves
        the input untouched.

    regressor: regressor object, optional (default ``DMD``)
        The regressor used to learn the Koopman operator from the observables.
        ``regressor`` can either extend the
        :class:`pykoopman.regression.BaseRegressor`, or ``pydmd.DMDBase``.
        In the latter case, the pydmd object must have both a ``fit``
        and a ``predict`` method.

    quiet: booolean, optional (default False)
        Whether or not warnings should be silenced during fitting.

    Attributes
    ----------
    model: sklearn.pipeline.Pipeline
        Internal representation of the forward model.
        Applies the observables and the regressor.

    n_input_features_: int
        Number of input features before computing observables.

    n_output_features_: int
        Number of output features after computing observables.
    """

    def __init__(self, observables=None, regressor=None, quiet=False):
        if observables is None:
            observables = Identity()
        if regressor is None:
            regressor = DMD(svd_rank=2)
        if isinstance(regressor, DMDBase):
            regressor = DMDRegressor(regressor)
        elif not isinstance(regressor, (BaseRegressor)):
            raise TypeError("Regressor must be from valid class")

        self.observables = observables
        self.regressor = regressor
        self.quiet = quiet

    def fit(self, x):
        """
        Fit the Koopman model by learning an approximate Koopman operator.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_features)
            Measurement data to be fit. Each row should correspond to an example
            and each column a feature. It is assumed that examples are
            equi-spaced in time (i.e. a uniform timestep is assumed).

        Returns
        -------
        self: returns a fit ``Koopman`` instance
        """
        x = validate_input(x)

        steps = [
            ("observables", self.observables),
            ("regressor", self.regressor),
        ]
        self.model = Pipeline(steps)

        action = "ignore" if self.quiet else "default"
        with catch_warnings():
            filterwarnings(action, category=UserWarning)
            self.model.fit(x)

        self.n_input_features_ = self.model.steps[0][1].n_input_features_
        self.n_output_features_ = self.model.steps[0][1].n_output_features_

        return self

    def predict(self, x):
        """
        Predict the state one timestep in the future.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_input_features)
            Current state.

        Returns
        -------
        y: numpy.ndarray, shape (n_samples, n_input_features)
            Predicted state one timestep in the future.
        """
        check_is_fitted(self, "n_output_features_")
        return self.observables.inverse(self._step(x))

    def simulate(self, x0, n_steps=1):
        """
        Simulate an initial state forward in time with the learned Koopman
        model.

        Parameters
        ----------
        x0: numpy.ndarray, shape (n_input_features,) or \
                (n_consumed_samples + 1, n_input_features)
            Initial state from which to simulate.
            If using :code:`TimeDelay` observables, ``x0`` should contain
            enough examples to compute all required time delays,
            i.e. ``n_consumed_samples + 1``.

        n_steps: int, optional (default 1)
            Number of forward steps to be simulated.

        Returns
        -------
        y: numpy.ndarray, shape (n_steps, n_input_features)
            Simulated states.
            Note that ``y[0, :]`` is one timestep ahead of ``x0``.
        """
        check_is_fitted(self, "n_output_features_")
        # Could have an option to only return the end state and not all
        # intermediate states to save memory.
        y = empty((n_steps, self.n_input_features_), dtype=self.koopman_matrix.dtype)
        y[0] = self.predict(x0)

        if isinstance(self.observables, TimeDelay):
            n_consumed_samples = self.observables.n_consumed_samples
            for k in range(n_consumed_samples):
                y[k + 1] = self.predict(vstack((x0[k + 1 :], y[: k + 1])))

            for k in range(n_consumed_samples, n_steps - 1):
                y[k + 1] = self.predict(y[k - n_consumed_samples : k + 1])
        else:
            for k in range(n_steps - 1):
                y[k + 1] = self.predict(y[k])

        return y

    def score(self, x, y=None, cast_as_real=True, metric=r2_score, **metric_kws):
        """
        Score the model prediction for the next timestep.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_input_features)
            State measurements.
            Each row should correspond to the system state at some point
            in time.
            If ``y`` is not passed, then it is assumed that the examples are
            equi-spaced in time and are given in sequential order.
            If ``y`` is passed, then this assumption need not hold.

        y: numpy.ndarray, shape (n_samples, n_input_features), optional \
                (default None)
            State measurements one timestep in the future.
            Each row of this array should give the corresponding row in x advanced
            forward in time by one timestep.
            If None, the rows of ``x`` are used to construct ``y``.

        cast_as_real: bool, optional (default True)
            Whether to take the real part of predictions when computing the score.
            Many Scikit-learn metrics do not support complex numbers.

        metric: callable, optional (default ``r2_score``)
            The metric function used to score the model predictions.

        metric_kws: dict, optional
            Optional parameters to pass to the metric function.

        Returns
        -------
        score: float
            Metric function value for the model predictions at the next timestep.
        """
        check_is_fitted(self, "n_output_features_")
        x = validate_input(x)

        if isinstance(self.observables, TimeDelay):
            n_consumed_samples = self.observables.n_consumed_samples

            # User may pass in too-large
            if y is not None and len(y) == len(x):
                warn(
                    f"The first {n_consumed_samples} entries of y were ignored because "
                    "TimeDelay obesrvables were used."
                )
                y = y[n_consumed_samples:]
        else:
            n_consumed_samples = 0

        if y is None:
            if cast_as_real:
                return metric(
                    x[n_consumed_samples + 1 :].real,
                    self.predict(x[:-1]).real,
                    **metric_kws,
                )
            else:
                return metric(
                    x[n_consumed_samples + 1 :], self.predict(x[:-1]), **metric_kws
                )
        else:
            if cast_as_real:
                return metric(y.real, self.predict(x).real, **metric_kws)
            else:
                return metric(y, self.predict(x), **metric_kws)

    def get_feature_names(self, input_features=None):
        """
        Get the names of the individual features constituting the observables.

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
        return self.observables.get_feature_names(input_features=input_features)

    def _step(self, x):
        """
        Map x one timestep forward in the space of observables.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_input_features)
            State vectors to be stepped forward.

        Returns
        -------
        X': numpy.ndarray, shape (n_samples, self.n_output_features_)
            Observables one timestep after x.
        """
        check_is_fitted(self, "n_output_features_")
        return self.model.predict(x)

    @property
    def koopman_matrix(self):
        """
        The Koopman matrix K satisfying g(X') = g(X) * K
        where g denotes the observables map and X' denotes x advanced one timestep.
        """
        check_is_fitted(self, "n_output_features_")
        return self.model.steps[-1][1].coef_
