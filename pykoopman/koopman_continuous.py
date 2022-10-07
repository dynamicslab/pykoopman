import numpy as np
from sklearn.utils.validation import check_is_fitted

from .differentiation import Derivative
from .koopman import Koopman


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

    def __init__(
        self,
        observables=None,
        differentiator=Derivative(kind="finite_difference", k=1),
        regressor=None,
    ):

        super().__init__(observables, regressor)
        self.differentiator = differentiator

    def predict(self, x, t=0, u=None):
        """Predict using continuous-time Koopman model"""
        check_is_fitted(self, "model")

        if u is None:
            ypred = self.model.predict(X=x, t=t)
        else:
            ypred = self.model.predict(X=x, u=u, t=t)

        output = []
        for k in range(ypred.shape[0]):
            output.append(np.squeeze(self.observables.inverse(ypred[k][np.newaxis, :])))

        return output

    def simulate(self, x, t=0, u=None):
        """Simulate continuous-time Koopman model"""
        check_is_fitted(self, "model")

        # Note: the above method predict is doing simulation.

        # output = [self.predict(x)]
        # for k in range(n_steps - 1):
        #     output.append(self.predict(output[-1]))
        pass

    def _step(self, x, u=None):
        """For consistency kept."""
        raise NotImplementedError("ContinuousKoopman does not have a step function.")
