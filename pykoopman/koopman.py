from warnings import warn

from numpy import empty
import numpy as np
from pydmd import DMD
from pydmd import DMDBase
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .common import validate_input
from .observables import Identity
from .regression import BaseRegressor
from .regression import DMDc
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

    n_control_features_: int
        Number of control features used as input to the system.
    time: dictionary
        Time vector properties.
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

    def fit(self, x, u=None, dt=1):
        """
        Fit the Koopman model by learning an approximate Koopman operator.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_features)
            Measurement data to be fit. Each row should correspond to an example
            and each column a feature. It is assumed that examples are
            equi-spaced in time (i.e. a uniform timestep is assumed).

        u: numpy.ndarray, shape (n_samples, n_control_features)
            Control/actuation/external parameter data. Each row should correspond
            to one sample and each column a control variable or feature.
            The control variable may be amplitude of an actuator or an external,
            time-varying parameter. It is assumed that samples are equi-spaced
            in time (i.e. a uniform timestep is assumed) and correspond to the
            samples in x.
        dt: float, (default=1)
            Time step between samples

        Returns
        -------
        self: returns a fit ``Koopman`` instance
        """
        x = validate_input(x)

        if u is None:
            self.n_control_features_ = 0
        elif not isinstance(self.regressor, DMDc):
            raise ValueError(
                "Control input u was passed, but self.regressor is not DMDc"
            )

        steps = [
            ("observables", self.observables),
            ("regressor", self.regressor),
        ]
        self.model = Pipeline(steps)

        if u is None:
            self.model.fit(x)
        else:
            self.model.fit(x, u)

        self.n_input_features_ = self.model.steps[0][1].n_input_features_
        self.n_output_features_ = self.model.steps[0][1].n_output_features_
        if hasattr(self.model.steps[1][1], "n_control_features_"):
            self.n_control_features_ = self.model.steps[1][1].n_control_features_

        self.time = dict([('tstart', 0),
                           ('tend', dt * (self.model.steps[1][1].n_samples_ - 1)),
                           ('dt', dt)])
        return self

    def predict(self, x, u=None):
        """
        Predict the state one timestep in the future.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_samples, n_input_features)
            Current state.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        Returns
        -------
        y: numpy.ndarray, shape (n_samples, n_input_features)
            Predicted state one timestep in the future.
        """
        check_is_fitted(self, "model")
        return self.observables.inverse(self._step(x, u))

    def simulate(self, x0, u=None, n_steps=1):
        """
        Simulate an initial state forward in time with the learned Koopman
        model.

        Parameters
        ----------
        x0: numpy.ndarray, shape (n_input_features,)
            Initial state from which to simulate.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        n_steps: int, optional (default 1)
            Number of forward steps to be simulated.

        Returns
        -------
        xhat: numpy.ndarray, shape (n_steps, n_input_features)
            Simulated states.
            Note that ``xhat[0, :]`` is one timestep ahead of ``x0``.
        """
        check_is_fitted(self, "model")
        # Could have an option to only return the end state and not all
        # intermediate states to save memory.
        xhat = empty((n_steps, self.n_input_features_),
                     dtype=self.koopman_matrix.dtype)

        if u is None:
            xhat[0] = self.predict(x0)
            for k in range(n_steps - 1):
                xhat[k + 1] = self.predict(xhat[k])
        else:
            xhat[0] = self.predict(x0, u[0])
            for k in range(n_steps - 1):
                xhat[k + 1] = self.predict(xhat[k], u[k + 1])

        return xhat

    def _step(self, x, u=None):
        """
        Map x one timestep forward in the space of observables.

        Parameters
        ----------
        x: numpy.ndarray, shape (n_examples, n_input_features)
            State vectors to be stepped forward.

        u: numpy.ndarray, shape (n_samples, n_control_features), \
                optional (default None)
            Time series of external actuation/control.

        Returns
        -------
        X': numpy.ndarray, shape (n_examples, self.n_output_features_)
            Observables one timestep after x.
        """

        check_is_fitted(self, "model")

        if u is None or self.n_control_features_ == 0:
            if self.n_control_features_ > 0:
                # TODO: replace with u = 0 as default
                raise TypeError(
                    "Model was fit using control variables, so u is required"
                )
            elif u is not None:
                warn(
                    "Control variables u were ignored because control variables"
                    "were not used when the model was fit"
                )
            return self.model.predict(X=x)
        else:
            if not isinstance(self.regressor, DMDc):
                raise ValueError(
                    "Control input u was passed, but self.regressor is not DMDc"
                )
            return self.model.predict(X=x, u=u)

    @property
    def koopman_matrix(self):
        """
        The Koopman matrix K satisfying g(X') = g(X) * K
        where g denotes the observables map and X' denotes x advanced
        one timestep.
        """
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].coef_

    @property
    def state_transition_matrix(self):
        """
        The state transition matrix A satisfies x' = Ax + Bu.
        # TODO: consider whether we want to match sklearn and have A and B satisfy
        # x' = xA + uB instead
        """
        check_is_fitted(self, "model")
        if not isinstance(self.regressor, DMDc):
            raise ValueError(
                "self.regressor is not DMDc, so object has no "
                "state_transition_matrix"
            )
        return self.model.steps[-1][1].state_matrix_

    @property
    def control_matrix(self):
        """
        The control matrix (or vector) B satisfies x' = Ax + Bu.
        """
        check_is_fitted(self, "model")
        if not isinstance(self.regressor, DMDc):
            raise ValueError(
                "self.regressor is not DMDc, so object has no control_matrix"
            )
        return self.model.steps[-1][1].control_matrix_

    @property
    def projection_matrix(self):
        """
        The control matrix (or vector) B satisfies x' = Ax + Bu.
        """
        check_is_fitted(self, "model")
        if not isinstance(self.regressor, DMDc):
            raise ValueError(
                "self.regressor is not DMDc, so object has no projection_matrix"
            )
        return self.model.steps[-1][1].projection_matrix_

    @property
    def projection_matrix_output(self):
        """
        The control matrix (or vector) B satisfies x' = Ax + Bu.
        """
        check_is_fitted(self, "model")
        if not isinstance(self.regressor, DMDc):
            raise ValueError(
                "self.regressor is not DMDc, so object has no "
                "projection_matrix_output"
            )
        return self.model.steps[-1][1].projection_matrix_output_

    @property
    def modes(self):
        """
        Koopman modes
        """
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].modes_

    @property
    def eigenvalues(self):
        """
        Discrete-time Koopman eigenvalues obtained from spectral decomposition
        of the Koopman matrix
        """
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].eigenvalues_

    @property
    def frequencies(self):
        """
        Oscillation frequencies of Koopman modes/eigenvectors
        """
        check_is_fitted(self, "model")
        dt = self.time['dt']
        return np.imag(np.log(self.eigenvalues) / dt) / (2 * np.pi)
        # return self.model.steps[-1][1].frequencies_

    @property
    def eigenvalues_continuous(self):
        """
        Continuous-time Koopman eigenvalues obtained from spectral decomposition
        of the Koopman matrix
        """
        check_is_fitted(self, "model")
        dt = self.time['dt']
        return np.log(self.eigenvalues) / dt
        # return self.model.steps[-1][1].eigenvalues_continuous_