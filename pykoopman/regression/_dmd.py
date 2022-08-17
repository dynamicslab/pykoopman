from warnings import warn

from numpy import identity
from pydmd import DMDBase
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor


class DMDRegressor(BaseRegressor):
    """
    Wrapper for PyDMD regressors.

    Parameters
    ----------
    DMDRegressor: DMDBase subclass
        Regressor from PyDMD. Must extend the DMDBase class.
    """

    def __init__(self, regressor):
        if not isinstance(regressor, DMDBase):
            raise ValueError("regressor must be a subclass of DMDBase from pydmd.")
        super(DMDRegressor, self).__init__(regressor)

    def fit(self, x, y=None, dt=1):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_examples, n_features)
            Measurement data to be fit.

        Returns
        -------
        self: returns a fit ``DMDRegressor`` instance
        """
        if y is not None:
            warn("pydmd regressors do not require the y argument when fitting.")
        self.n_samples_, self.n_input_features_ = x.shape
        # We transpose x because PyDMD assumes examples are columns, not rows
        self.regressor.fit(x.T)

        # self.debug_coef_ = self.regressor.operator

        # here we use a trick to get the `A' matrix, not the low rank one
        # note that pydmd only stores the low-rank `A' matrix, which makes
        # sense in high-dimensional system but we focus on low-dimensional,
        # but highly nonlinear system
        self.coef_ = self.regressor.predict(identity(x.shape[1])).T

        # Get Koopman modes, eigenvectors, eigenvalues manually at here
        # self.modes_ = self.regressor.modes
        self._amplitudes_ = self.regressor.amplitudes
        self._eigenvalues_ = self.regressor.eigs
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_examples, n_features)
            Measurement data upon which to base prediction.

        Returns
        -------
        y: numpy ndarray, shape (n_examples, n_features)
            Prediction of x one timestep in the future.

        """
        check_is_fitted(self, "coef_")
        return self.regressor.predict(x.T).T

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "coef_")
        return self._eigenvalues_

    @property
    def amplitudes_(self):
        check_is_fitted(self, "coef_")
        return self._amplitudes_
