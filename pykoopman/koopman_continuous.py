"""module for continuous time Koopman class"""
from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .differentiation import Derivative
from .koopman import Koopman


class KoopmanContinuous(Koopman):
    """
    Continuous-time Koopman class.

    Args:
        observables: Observables object, optional
            (default: pykoopman.observables.Identity)
            Map(s) to apply to raw measurement data before
            estimating the Koopman operator. Must extend
            pykoopman.observables.BaseObservables. The default
            option, pykoopman.observables.Identity, leaves the
            input untouched.
        differentiator: Callable, optional
            (default: centered difference)
            Function used to compute numerical derivatives.
            The function must have the call signature
            differentiator(x, t), where x is a 2D numpy ndarray
            of shape (n_samples, n_features) and t is a 1D numpy
            ndarray of shape (n_samples,).
        regressor: Regressor object, optional
            (default: DMD)
            The regressor used to learn the Koopman operator from
            the observables. regressor can either extend
            pykoopman.regression.BaseRegressor, or the
            pydmd.DMDBase class. In the latter case, the pydmd
            object must have both a fit and a predict method.
    """

    def __init__(
        self,
        observables=None,
        differentiator=Derivative(kind="finite_difference", k=1),
        regressor=None,
    ):
        """
        Continuous-time Koopman class.

        Args:
            observables: Observables object, optional
                (default: pykoopman.observables.Identity)
                Map(s) to apply to raw measurement data before
                estimating the Koopman operator. Must extend
                pykoopman.observables.BaseObservables. The default
                option, pykoopman.observables.Identity, leaves the
                input untouched.
            differentiator: Callable, optional
                (default: centered difference)
                Function used to compute numerical derivatives.
                The function must have the call signature
                differentiator(x, t), where x is a 2D numpy ndarray
                of shape (n_samples, n_features) and t is a 1D numpy
                ndarray of shape (n_samples,).
            regressor: Regressor object, optional
                (default: DMD)
                The regressor used to learn the Koopman operator from
                the observables. regressor can either extend
                pykoopman.regression.BaseRegressor, or the
                pydmd.DMDBase class. In the latter case, the pydmd
                object must have both a fit and a predict method.
        """
        super().__init__(observables, regressor)
        self.differentiator = differentiator

    def predict(self, x, dt=0, u=None):
        """
        Predict using continuous-time Koopman model.

        Args:
            x: numpy.ndarray
                State measurements. Each row should correspond to
                the system state at some point in time.
            dt: float, optional (default: 0)
                Time step between measurements. If specified, the
                prediction is made for the given time step in the
                future.
            u: numpy.ndarray, optional (default: None)
                Control input/actuation data. Each row should
                correspond to one sample and each column a control
                variable or feature.

        Returns:
            output: numpy.ndarray
                Predicted state using the continuous-time Koopman
                model. Each row corresponds to the predicted state
                for the corresponding row in x.
        """
        check_is_fitted(self, "_pipeline")

        if u is None:
            ypred = self._pipeline.predict(X=x, t=dt)
        else:
            ypred = self._pipeline.predict(X=x, u=u, t=dt)

        output = self.observables.inverse(ypred)

        return output

    def simulate(self, x, t=0, u=None):
        """
        Simulate continuous-time Koopman model.

        Args:
            x: numpy.ndarray
                Initial state from which to simulate. Each row
                corresponds to the system state at some point in time.
            t: float, optional (default: 0)
                Time at which to simulate the system. If specified,
                the simulation is performed for the given time.
            u: numpy.ndarray, optional (default: None)
                Control input/actuation data. Each row should
                correspond to one sample and each column a control
                variable or feature.

        Returns:
            output: numpy.ndarray
                Simulated states of the system. Each row corresponds
                to the simulated state at a specific time point.
        """
        check_is_fitted(self, "_pipeline")

        if u is None:
            ypred = self._pipeline.predict(X=x, t=t)
        else:
            ypred = self._pipeline.predict(X=x, u=u, t=t)

        output = []
        for k in range(ypred.shape[0]):
            output.append(np.squeeze(self.observables.inverse(ypred[k][np.newaxis, :])))

        return np.array(output)

    def _step(self, x, u=None):
        """
        Placeholder method for step function.

        This method is not implemented in the ContinuousKoopman class
        as there is no explicit step function for continuous-time
        Koopman models.

        Raises:
            NotImplementedError: This method is not implemented
                in the ContinuousKoopman class.
        """
        raise NotImplementedError("ContinuousKoopman does not have a step function.")
