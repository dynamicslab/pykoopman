from numpy import empty
from pydmd import DMD
from pydmd import DMDBase
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .common import validate_input
from .observables import Identity
from .regression import BaseRegressor
from .regression import DMDRegressor


class Koopman(BaseEstimator):
    """
    Discrete-Time Koopman class.

    Parameters
    ----------
    observables: observables object, optional (default ``Identity``)
        Map(s) to apply to raw measurement data before estimating the
        Koopman operator.
        Must extend the ``pykoopman.observables.BaseObservables`` class.
        The default option, ``Identity`` leaves the input untouched.

    regressor: regressor object, optional (default ``DMD``)
        The regressor used to learn the Koopman operator from the observables.
        ``regressor`` can either extend the
        ``pykoopman.regression.BaseRegressor`` class, or the ``pydmd.DMDBase``
        class. In the latter case, the pydmd object must have both a ``fit``
        and a ``predict`` method.

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

    def __init__(self, observables=None, regressor=None):
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

    def fit(self, x):
        """
        Fit the Koopman model by learning an approximate Koopman operator.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_examples, n_features)
            Measurement data to be fit. Each row should correspond to an example
            and each column a feature. It is assumed that examples are
            equi-spaced in time (i.e. a uniform timestep is assumed).

        Returns
        -------
        self: returns a ``Koopman`` instance
        """
        x = validate_input(x)

        steps = [
            ("observables", self.observables),
            ("regressor", self.regressor),
        ]
        self.model = Pipeline(steps)

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
        check_is_fitted(self, "model")
        return self.observables.inverse(self._step(x))

    def simulate(self, x0, n_steps=1):
        """
        Simulate an initial state forward in time with the learned Koopman
        model.

        Parameters
        ----------
        x0: numpy.ndarray, shape (n_input_features,)
            Initial state from which to simulate.

        n_steps: int, optional (default 1)
            Number of forward steps to be simulated.

        Returns
        -------
        y: numpy.ndarray, shape (n_steps, n_input_features)
            Simulated states.
            Note that ``y[0, :]`` is one timestep ahead of ``x0``.
        """
        check_is_fitted(self, "model")
        # Could have an option to only return the end state and not all
        # intermediate states to save memory.
        y = empty((n_steps, self.n_input_features_), dtype=self.koopman_matrix.dtype)
        y[0] = self.predict(x0)
        for k in range(n_steps - 1):
            y[k + 1] = self.predict(y[k])

        return y

    def _step(self, x):
        """
        Map x one timestep forward in the space of observables.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_examples, n_input_features)
            State vectors to be stepped forward.

        Returns
        -------
        X': numpy.ndarray, shape (n_examples, self.n_output_features_)
            Observables one timestep after x.
        """
        check_is_fitted(self, "model")
        return self.model.predict(x)

    @property
    def koopman_matrix(self):
        """
        The Koopman matrix K satisfying g(X') = g(X) * K
        where g denotes the observables map and X' denotes x advanced one timestep.
        """
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].coef_
