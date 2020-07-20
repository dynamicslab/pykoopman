from pydmd import DMD
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from pykoopman.differentiation import FiniteDifference
from pykoopman.observables import Polynomial


class Koopman:
    """Primary Koopman class."""

    def __init__(self, observables=None, differentiator=None, regressor=None):
        if observables is None:
            observables = Polynomial()
        if differentiator is None:
            differentiator = FiniteDifference()
        if regressor is None:
            regressor = DMD()

        self.observables = observables
        self.differentiator = differentiator
        self.regressor = regressor

    def fit(self, x, t=None):
        x_dot = self.differentiator(x, t)

        steps = [
            ("observables", self.observables),
            ("regressor", self.regressor),
        ]
        self.model = Pipeline(steps)

        # TODO: make this solve the correct problem
        self.model.fit(x, x_dot)

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

    def get_koopman_matrix(self,):
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].coef_
