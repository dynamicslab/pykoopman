"""module for discrete time Koopman class"""
from __future__ import annotations

from warnings import catch_warnings
from warnings import filterwarnings
from warnings import warn

import numpy as np
from numpy import empty
from numpy import vstack
from pydmd import DMD
from pydmd import DMDBase
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .common import validate_input
from .observables import Identity
from .observables import TimeDelay
from .regression import BaseRegressor
from .regression import DMDc
from .regression import EDMDc
from .regression import EnsembleBaseRegressor
from .regression import PyDMDRegressor


class Koopman(BaseEstimator):
    """Discrete-Time Koopman class.

    The input-output data is all row-wise if stated elsewhere.
    All of the matrix, are based on column-wise linear system.
    This class is inherited from `pykoopman.regression.BaseEstimator`.

    Args:
        observables: observables object, optional
            (default: `pykoopman.observables.Identity`)
            Map(s) to apply to raw measurement data before estimating the
            Koopman operator.
            Must extend `pykoopman.observables.BaseObservables`.
            The default option, `pykoopman.observables.Identity`, leaves
            the input untouched.

        regressor: regressor object, optional (default: `DMD`)
            The regressor used to learn the Koopman operator from the observables.
            `regressor` can either extend the `pykoopman.regression.BaseRegressor`,
            or `pydmd.DMDBase`.
            In the latter case, the pydmd object must have both a `fit`
            and a `predict` method.

        quiet: boolean, optional (default: False)
            Whether or not warnings should be silenced during fitting.

    Attributes:
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

    def __init__(self, observables=None, regressor=None, quiet=False):
        """Constructor for the Koopman class.

        Args:
            observables: observables object, optional
                (default: `pykoopman.observables.Identity`)
                Map(s) to apply to raw measurement data before estimating the
                Koopman operator.
                Must extend `pykoopman.observables.BaseObservables`.
                The default option, `pykoopman.observables.Identity`, leaves
                the input untouched.

            regressor: regressor object, optional (default: `DMD`)
                The regressor used to learn the Koopman operator from the observables.
                `regressor` can either extend the `pykoopman.regression.BaseRegressor`,
                or `pydmd.DMDBase`.
                In the latter case, the pydmd object must have both a `fit`
                and a `predict` method.

            quiet: boolean, optional (default: False)
                Whether or not warnings should be silenced during fitting.
        """
        if observables is None:
            observables = Identity()
        if regressor is None:
            regressor = PyDMDRegressor(DMD(svd_rank=2))  # set default svd rank 2
        if isinstance(regressor, DMDBase):
            regressor = PyDMDRegressor(regressor)
        elif not isinstance(regressor, (BaseRegressor)):
            raise TypeError("Regressor must be from valid class")
        self.observables = observables
        self.regressor = regressor
        self.quiet = quiet

    def fit(self, x, y=None, u=None, dt=1):
        """
        Fit the Koopman model by learning an approximate Koopman operator.

        Args:
            x: numpy.ndarray, shape (n_samples, n_features)
                Measurement data to be fit. Each row should correspond to an example
                and each column a feature. If only x is provided, it is assumed that
                examples are equi-spaced in time (i.e., a uniform timestep is assumed).

            y: numpy.ndarray, shape (n_samples, n_features), optional (default: None)
                Target measurement data to be fit, i.e., it is assumed y = fun(x). Each
                row should correspond to an example and each column a feature. The
                samples in x and y are generally not required to be consecutive and
                equi-spaced.

            u: numpy.ndarray, shape (n_samples, n_control_features), optional (default:
                None) Control/actuation/external parameter data. Each row should
                correspond to one sample and each column a control variable or feature.
                The control variable may be the amplitude of an actuator or an external,
                time-varying parameter. It is assumed that samples in u occur at the
                time instances of the corresponding samples in x,
                e.g., x(t+1) = fun(x(t), u(t)).

            dt: float, optional (default: 1)
                Time step between samples

        Returns:
            self: returns a fit `Koopman` instance
        """
        x = validate_input(x)

        if u is None:
            self.n_control_features_ = 0
        elif not isinstance(self.regressor, DMDc) and not isinstance(
            self.regressor, EDMDc
        ):
            raise ValueError(
                "Control input u was passed, " "but self.regressor is not DMDc or EDMDc"
            )

        if y is None:  # or isinstance(self.regressor, PyDMDRegressor):
            # if there is only 1 trajectory OR regressor is PyDMD
            regressor = self.regressor
        else:
            # multiple trajectories
            regressor = EnsembleBaseRegressor(
                regressor=self.regressor,
                func=self.observables.transform,
                inverse_func=self.observables.inverse,
            )

        steps = [
            ("observables", self.observables),
            ("regressor", regressor),
        ]
        self._pipeline = Pipeline(steps)  # create `model` object using Pipeline

        action = "ignore" if self.quiet else "default"
        with catch_warnings():
            filterwarnings(action, category=UserWarning)
            if u is None:
                self._pipeline.fit(x, y, regressor__dt=dt)
            else:
                self._pipeline.fit(x, y, regressor__u=u, regressor__dt=dt)
            # update the second step with just the regressor, not the
            # EnsembleBaseRegressor
            if isinstance(self._pipeline.steps[1][1], EnsembleBaseRegressor):
                self._pipeline.steps[1] = (
                    self._pipeline.steps[1][0],
                    self._pipeline.steps[1][1].regressor_,
                )

        # pykoopman's n_input/output_features are simply
        # observables's input output features
        # observable's input features are just the number
        # of states. but the output features can be really high
        self.n_input_features_ = self._pipeline.steps[0][1].n_input_features_
        self.n_output_features_ = self._pipeline.steps[0][1].n_output_features_
        if hasattr(self._pipeline.steps[1][1], "n_control_features_"):
            self.n_control_features_ = self._pipeline.steps[1][1].n_control_features_

        # compute amplitudes
        if y is None:
            if hasattr(self.observables, "n_consumed_samples"):
                # g0 = self.observables.transform(
                #     x[0 : 1 + self.observables.n_consumed_samples]
                # )
                self._amplitudes = np.abs(
                    self.psi(x[0 : 1 + self.observables.n_consumed_samples].T)
                )
            else:
                # g0 = self.observables.transform(x[0:1])
                self._amplitudes = np.abs(self.psi(x[0:1].T))
        else:
            self._amplitudes = None

        self.time = {
            "tstart": 0,
            "tend": dt * (self._pipeline.steps[1][1].n_samples_ - 1),
            "dt": dt,
        }

        return self

    def predict(self, x, u=None):
        """
        Predict the state one timestep in the future.

        Args:
            x: numpy.ndarray, shape (n_samples, n_input_features)
                Current state.

            u: numpy.ndarray, shape (n_samples, n_control_features),
                optional (default None)
                Time series of external actuation/control.

        Returns:
            y: numpy.ndarray, shape (n_samples, n_input_features)
                Predicted state one timestep in the future.
        """

        check_is_fitted(self, "n_output_features_")
        return self.observables.inverse(self._step(x, u))

    def simulate(self, x0, u=None, n_steps=1):
        """Simulate an initial state forward in time with the learned Koopman model.

        Args:
            x0: numpy.ndarray, shape (n_input_features,) or
                (n_consumed_samples + 1, n_input_features)
                Initial state from which to simulate.
                If using TimeDelay observables, `x0` should contain
                enough examples to compute all required time delays,
                i.e., `n_consumed_samples + 1`.

            u: numpy.ndarray, shape (n_samples, n_control_features),
                optional (default None)
                Time series of external actuation/control.

            n_steps: int, optional (default 1)
                Number of forward steps to be simulated.

        Returns:
            y: numpy.ndarray, shape (n_steps, n_input_features)
                Simulated states.
                Note that `y[0, :]` is one timestep ahead of `x0`.
        """
        check_is_fitted(self, "n_output_features_")
        # Could have an option to only return the end state and not all
        # intermediate states to save memory.

        y = empty((n_steps, self.n_input_features_), dtype=self.A.dtype)
        if u is None:
            y[0] = self.predict(x0)
        else:
            y[0] = self.predict(x0, u[0])

        if isinstance(self.observables, TimeDelay):
            n_consumed_samples = self.observables.n_consumed_samples
            if u is None:
                for k in range(n_consumed_samples):
                    y[k + 1] = self.predict(vstack((x0[k + 1 :], y[: k + 1])))
                for k in range(n_consumed_samples, n_steps - 1):
                    y[k + 1] = self.predict(y[k - n_consumed_samples : k + 1])
            else:
                for k in range(n_consumed_samples):
                    y[k + 1] = self.predict(vstack((x0[k + 1 :], y[: k + 1])), u[k + 1])
                for k in range(n_consumed_samples, n_steps - 1):
                    y[k + 1] = self.predict(y[k - n_consumed_samples : k + 1], u[k + 1])

        else:
            if u is None:
                for k in range(n_steps - 1):
                    y[k + 1] = self.predict(y[k].reshape(1, -1))
            else:
                for k in range(n_steps - 1):
                    y[k + 1] = self.predict(
                        y[k].reshape(1, -1), u[k + 1].reshape(1, -1)
                    )

        return y

    def get_feature_names(self, input_features=None):
        """Get the names of the individual features constituting the observables.

        Args:
            input_features: list of string, length n_input_features,
                optional (default None)
                String names for input features, if available. By default,
                the names "x0", "x1", ..., "xn_input_features" are used.

        Returns:
            output_feature_names: list of string, length n_output_features
                Output feature names.
        """
        check_is_fitted(self, "n_input_features_")
        return self.observables.get_feature_names(input_features=input_features)

    def _step(self, x, u=None):
        """Map x one timestep forward in the space of observables.

        Args:
            x: numpy.ndarray, shape (n_samples, n_input_features)
                State vectors to be stepped forward.

            u: numpy.ndarray, shape (n_samples, n_control_features),
                optional (default None)
                Time series of external actuation/control.

        Returns:
            X': numpy.ndarray, shape (n_samples, self.n_output_features_)
                Observables one timestep after x.
        """
        check_is_fitted(self, "n_output_features_")

        if u is None or self.n_control_features_ == 0:
            if self.n_control_features_ > 0:
                raise TypeError(
                    "Model was fit using control variables, so u is required"
                )
            elif u is not None:
                warn(
                    "Control variables u were ignored because control variables were"
                    " not used when the model was fit"
                )
            return self._pipeline.predict(X=x)
        else:
            if not isinstance(self.regressor, DMDc) and not isinstance(
                self.regressor, EDMDc
            ):
                raise ValueError(
                    "Control input u was passed, but self.regressor is not DMDc "
                    "or EDMDc"
                )
            return self._pipeline.predict(X=x, u=u)

    def phi(self, x_col):
        """Compute the feature matrix phi(x) given `x_col`.

        Args:
            x_col: numpy.ndarray, shape (n_features, n_samples)
                State vectors to be evaluated for phi.

        Returns:
            phi: numpy.ndarray, shape (n_samples, self.n_output_features_)
                Value of phi evaluated at input `x_col`.
        """
        x = x_col.T
        y = self.observables.transform(x)
        phi = self._pipeline.steps[-1][1]._compute_phi(y.T)
        return phi

    def psi(self, x_col):
        """Compute the Koopman psi(x) given `x_col`.

        Args:
            x_col: numpy.ndarray, shape (n_features, n_samples)
                State vectors to be evaluated for psi.

        Returns:
            eigen_phi: numpy.ndarray, shape (n_samples, self.n_output_features_)
                Value of psi evaluated at input `x_col`.
        """
        x = x_col.T
        y = self.observables.transform(x)
        ephi = self._pipeline.steps[-1][1]._compute_psi(y.T)
        return ephi

    @property
    def A(self):
        """Returns the state transition matrix `A`.

        The state transition matrix A satisfies y' = Ay or y' = Ay + Bu,
        respectively, where y = g(x) and y is a low-rank representation.
        """
        check_is_fitted(self, "_pipeline")
        if isinstance(self.regressor, DMDBase):
            raise ValueError("self.regressor " "has no A!")
        if hasattr(self._pipeline.steps[-1][1], "state_matrix_"):
            return self._pipeline.steps[-1][1].state_matrix_
        else:
            raise ValueError("self.regressor" "has no state_matrix")

    @property
    def B(self):
        """Returns the control matrix `B`.

        The control matrix (or vector) B satisfies y' = Ay + Bu.
        y is the reduced system state.
        """
        check_is_fitted(self, "_pipeline")
        if isinstance(self.regressor, DMDBase):
            raise ValueError("this type of self.regressor has no B")
        return self._pipeline.steps[-1][1].control_matrix_

    @property
    def C(self):
        """Returns the measurement matrix (or vector) C.

        The measurement matrix C satisfies x = C * phi_r.
        """
        check_is_fitted(self, "_pipeline")
        # if not isinstance(self.observables, RadialBasisFunction):
        #     raise ValueError("this type of self.observable has no C")
        # return self._pipeline.steps[0][1].measurement_matrix_

        measure_mat = self._pipeline.steps[0][1].measurement_matrix_
        ur = self._pipeline.steps[-1][1].ur
        C = measure_mat @ ur
        return C

    @property
    def W(self):
        """Returns the Koopman modes."""

        check_is_fitted(self, "_pipeline")
        # return self.C @ self._pipeline.steps[-1][1].unnormalized_modes
        return self.C @ self._pipeline.steps[-1][1].eigenvectors_

    @property
    def _regressor_eigenvectors(self):
        """Returns the eigenvectors of the regressor."""
        check_is_fitted(self, "_pipeline")
        return self._pipeline.steps[-1][1].eigenvectors_

    @property
    def lamda(self):
        """Returns the discrete-time Koopman lambda obtained from spectral
        decomposition."""
        check_is_fitted(self, "_pipeline")
        return np.diag(self._pipeline.steps[-1][1].eigenvalues_)

    @property
    def lamda_array(self):
        """Returns the discrete-time Koopman lambda as an array."""
        check_is_fitted(self, "_pipeline")
        return np.diag(self.lamda) + 0j

    @property
    def continuous_lamda_array(self):
        """Returns the continuous-time Koopman lambda as an array."""
        check_is_fitted(self, "_pipeline")
        return np.log(self.lamda_array) / self.time["dt"]

    @property
    def ur(self):
        """Returns the projection matrix Ur."""
        check_is_fitted(self, "_pipeline")
        return self._pipeline.steps[-1][1].ur

    def validity_check(self, t, x):
        """Perform a validity check of eigenfunctions.

        The validity check tests the linearity of eigenfunctions phi(x(t)) == phi(x(0))
        * exp(lambda*t).

        Args:
            t: numpy.ndarray, shape (n_samples,)
                Time vector.
            x: numpy.ndarray, shape (n_samples, n_input_features)
                State vectors to be checked.

        Returns:
            efun_index: list
                Sorted indices of eigenfunctions based on linearity error.
            linearity_error: list
                Linearity error for each eigenfunction.
        """

        psi = self.psi(x.T)
        omega = np.log(np.diag(self.lamda) + 0j) / self.time["dt"]

        # omega = self.eigenvalues_continuous
        linearity_error = []
        for i in range(self.lamda.shape[0]):
            linearity_error.append(
                np.linalg.norm(psi[i, :] - np.exp(omega[i] * t) * psi[i, 0:1])
            )
        sort_idx = np.argsort(linearity_error)
        efun_index = np.arange(len(linearity_error))[sort_idx]
        linearity_error = [linearity_error[i] for i in sort_idx]
        return efun_index, linearity_error

    def score(self, x, y=None, cast_as_real=True, metric=r2_score, **metric_kws):
        """Score the model predictions for the next timestep.

        Parameters:
            x: numpy.ndarray, shape (n_samples, n_input_features)
                State measurements.
            y: numpy.ndarray, shape (n_samples, n_input_features), optional
                (default None). State measurements one timestep in the future.
            cast_as_real: bool, optional (default True)
                Whether to take the real part of predictions when computing the score.
            metric: callable, optional (default r2_score)
                The metric function used to score the model predictions.
            metric_kws: dict, optional
                Optional parameters to pass to the metric function.

        Returns:
            score: float
                Metric function value for the model predictions at the next timestep.
        """
        check_is_fitted(self, "n_output_features_")
        x = validate_input(x)

        if isinstance(self.observables, TimeDelay):
            n_consumed_samples = self.observables.n_consumed_samples

            # User may pass in too-large
            if y is not None and len(y) == len(x):
                warn(
                    f"The first {n_consumed_samples} entries of y were ignored because "
                    "TimeDelay obesrvables were used."
                )
                y = y[n_consumed_samples:]
        else:
            n_consumed_samples = 0

        if y is None:
            if cast_as_real:
                return metric(
                    x[n_consumed_samples + 1 :].real,
                    self.predict(x[:-1]).real,
                    **metric_kws,
                )
            else:
                return metric(
                    x[n_consumed_samples + 1 :], self.predict(x[:-1]), **metric_kws
                )
        else:
            if cast_as_real:
                return metric(y.real, self.predict(x).real, **metric_kws)
            else:
                return metric(y, self.predict(x), **metric_kws)

    def _observable(self):
        """Returns the observable transformation."""
        return self._pipeline.steps[0][1]

    def _regressor(self):
        """Returns the fitted regressor."""
        # this can access the fitted regressor
        # todo: future we need to figure out a way to do time delay multiple
        #  trajectories DMD
        # my idea is to manually call xN observables then concate the data to let
        # the _regressor.fit to update the model coefficients.
        # call this function with _regressor()
        return self._pipeline.steps[1][1]
