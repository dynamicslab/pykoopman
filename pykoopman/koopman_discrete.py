from pydmd import DMD
from pydmd import DMDBase
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .common.base import validate_input
from .observables import Polynomial
from .regression import BaseRegressor
from .regression import DMDRegressor


class Koopman:
    """Primary Koopman class."""

    def __init__(self, observables=None, regressor=None, dt_default=1):
        if observables is None:
            observables = Polynomial(degree=1)
        if regressor is None:
            regressor = DMD()
        if not isinstance(dt_default, float) and not isinstance(dt_default, int):
            raise ValueError("dt_default must be a positive number")
        elif dt_default <= 0:
            raise ValueError("dt_default must be a positive number")
        else:
            self.dt_default = dt_default

        self.observables = observables
        self.regressor = regressor

    def fit(self, x, x_dot=None, dt=None, x_shift=None):
        if dt is None:
            dt = self.dt_default

        x = validate_input(x, dt)
        if x_dot is None:
            x_dot = x[1:]
            x = x[:-1]
        else:
            x_dot = validate_input(x_dot, dt)

        if isinstance(self.regressor, DMDBase):
            regressor = DMDRegressor(self.regressor)
        else:
            regressor = BaseRegressor(self.regressor)

        steps = [
            ("observables", self.observables),
            ("regressor", regressor),
        ]
        self.model = Pipeline(steps)

        # TODO: make this solve the correct problem
        self.model.fit(x, x_dot)

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
        output = [self.predict(x)]
        for k in range(n_steps - 1):
            output.append(self.predict(output[-1]))

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
