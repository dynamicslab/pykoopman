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
from .koopman import Koopman
from .differentiation import Derivative


class KoopmanContinuous(Koopman):
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

    def __init__(self,
                 observables=None,
                 differentiator=Derivative(kind='finite_difference', k=1),
                 regressor=None
                 ):

        super().__init__(observables, regressor)
        self.differentiator = differentiator

    def predict(self, x):
        check_is_fitted(self, "model")
        return self.observables.inverse(self._step(x))

    def simulate(self, x, n_steps=1):
        check_is_fitted(self, "model")
        # TODO
        # Could have an option to only return the end state and not all
        # intermediate states to save memory.
        output = [self.predict(x)]
        for k in range(n_steps - 1):
            output.append(self.predict(output[-1]))

    def _step(self, x):
        # TODO:
        check_is_fitted(self, "model")
        return self.model.predict(X=x, u=None)
