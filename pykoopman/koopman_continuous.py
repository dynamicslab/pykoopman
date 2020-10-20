from numpy import arange
from pydmd import DMD
from pydmd import DMDBase
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .common import drop_nan_rows
from .common import validate_input
from .differentiation import Derivative
from .observables import Identity
from .regression import BaseRegressor
from .regression import DMDRegressor


class KoopmanContinuous(BaseEstimator):
    """
    Continuous-time Koopman class.

    Parameters
    ----------
    observables: observables object, optional \
            (default :class:`pykoopman.observables.Identity`)
        Map(s) to apply to raw measurement data before estimating the
        Koopman operator.
        Must extend :class:`pykoopman.observables.BaseObservables`.
        The default option, :class`pykoopman.observables.Identity` leaves
        the input untouched.

    differentiator: callable, optional (default centered difference)
        Function used to compute numerical derivatives. The function must
        have the call signature :code:`differentiator(x, t)`, where ``x`` is
        a 2D numpy ndarray of shape ``(n_samples, n_features)`` and ``t`` is
        a 1D numpy ndarray of shape ``(n_samples,)``.

    regressor: regressor object, optional (default ``DMD``)
        The regressor used to learn the Koopman operator from the observables.
        ``regressor`` can either extend
        :class:`pykoopman.regression.BaseRegressor`, or the ``pydmd.DMDBase``
        class. In the latter case, the pydmd object must have both a ``fit``
        and a ``predict`` method.

    TODO
    """

    def __init__(
        self, observables=None, differentiator=None, regressor=None, dt_default=1
    ):
        if observables is None:
            observables = Identity()
        if differentiator is None:
            differentiator = Derivative(kind="finite_difference", k=1)
        if regressor is None:
            regressor = DMD()
        if not isinstance(dt_default, float) and not isinstance(dt_default, int):
            raise ValueError("dt_default must be a positive number")
        elif dt_default <= 0:
            raise ValueError("dt_default must be a positive number")
        else:
            self.dt_default = dt_default

        self.observables = observables
        self.differentiator = differentiator
        self.regressor = regressor

    def fit(self, x, x_dot=None, dt=None, t=None, x_shift=None):
        if dt is None:
            dt = self.dt_default

        if t is None:
            t = dt * arange(x.shape[0])

        # TODO: validate data
        x = validate_input(x)

        # TODO: this will probably need to change as we need to compute derivatives
        # after computing observables
        if x_dot is None:
            x_dot = self.differentiator(x, t)

        # Some differentiation methods generate NaN entries at endpoints
        x_dot, x = drop_nan_rows(x_dot, x)

        if isinstance(self.regressor, DMDBase):
            regressor = DMDRegressor(self.regressor)
        else:
            regressor = BaseRegressor(self.regressor)

        steps = [
            ("observables", self.observables),
            ("regressor", regressor),
        ]
        self.model = Pipeline(steps)

        # TODO: make this solves the correct problem
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
