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
    """Primary Discrete-Time Koopman class."""

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
        check_is_fitted(self, "model")
        return self.observables.inverse(self._step(x))

    def simulate(self, x0, n_steps=1):
        check_is_fitted(self, "model")
        # Could have an option to only return the end state and not all
        # intermediate states to save memory.
        output = empty(
            (n_steps, self.n_input_features_), dtype=self.koopman_matrix.dtype
        )
        output[0] = self.predict(x0)
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
