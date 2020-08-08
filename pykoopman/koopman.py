from numpy import empty
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from pykoopman.observables import Identity
from pykoopman.regression import BaseRegressor
from pykoopman.regression import DMDRegressor

from pykoopman.common import validate_input

class Koopman(BaseEstimator):
    """Primary Discrete-Time Koopman class."""

    def __init__(self, observables=None, regressor=None):
        if observables is None:
            observables = Identity()
        if regressor is None:
            regressor = DMDRegressor()

        if not isinstance(regressor, BaseRegressor):
            raise TypeError("regressor must be from valid class")

        self.observables = observables
        self.regressor = regressor

    def fit(self, x, x_dot=None, dt=None):
        if dt is None:
            dt = self.dt_default

        x = validate_input(x, dt)
        if x_dot is None:
            x_dot = x[1:]
            x = x[:-1]
        else:
            x_dot = validate_input(x_dot, dt)

        steps = [
            ("observables", self.observables),
            ("regressor", self.regressor),
        ]
        self.model = Pipeline(steps)

        # TODO: make this solve the correct problem
        self.model.fit(x)

        self.n_input_features_ = self.model.steps[0][1].n_input_features_
        self.n_output_features_ = self.model.steps[0][1].n_output_features_

        return self

    def predict(self, x):
        check_is_fitted(self, "model")
        return self.observables.inverse(self._step(x))

    def simulate(self, x, n_steps=1):
        check_is_fitted(self, "model")
        # Could have an option to only return the end state and not all
        # intermediate states to save memory.
        output = empty((n_steps, self.n_input_features_))
        output[0] = self.predict(x)
        for k in range(n_steps - 1):
            output[k + 1] = self.predict(output[k])

        return output

    def _step(self, x):
        # TODO: rename this
        check_is_fitted(self, "model")
        return self.model.predict(x)

    @property
    def koopman_matrix(self):
        """
        Get the Koopman matrix K such that
        g(X') = g(X) * K
        """
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].coef_
