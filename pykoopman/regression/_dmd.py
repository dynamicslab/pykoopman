from warnings import warn

from pydmd import DMDBase
from sklearn.utils.validation import check_is_fitted

from ._base import BaseRegressor

# from numpy import identity

# todo: can't we just call DMD in PyKoopman, which is actually calling DMD
# from pydmd? -- No. we want that explicit call so we can use other dmd
# from pydmd without implementing on our own.


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
        # self._coef_ = self.regressor.predict(identity(x.shape[1])).T

        # but in order to follow the guidelines of using pydmd as much as we can
        # we should use the atilde from them. but remember to transpose
        self._coef_ = self.regressor.atilde.T

        self._amplitudes_ = self.regressor.amplitudes
        self._eigenvalues_ = self.regressor.eigs
        self._modes_ = self.regressor.modes
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
    def coef_(self):
        check_is_fitted(self, "_coef_")
        return self._coef_

    @property
    def eigenvalues_(self):
        check_is_fitted(self, "_eigenvalues_")
        return self._eigenvalues_

    @property
    def amplitudes_(self):
        check_is_fitted(self, "_amplitudes_")
        return self._amplitudes_

    @property
    def modes_(self):
        check_is_fitted(self, "_modes_")
        return self._modes_
