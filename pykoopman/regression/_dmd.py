from warnings import warn

from numpy import identity
from pydmd import DMD

from ._base import BaseRegressor


class DMDRegressor(BaseRegressor):
    """Wrapper for PyDMD regressors."""

    def __init__(self):
        super().__init__(DMD())

    # PyDMD uses transposed data
    def fit(self, x, y=None):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_examples, n_features)
            Measurement data to be fit.
        """
        if y is not None:
            warn("pydmd regressors do not require the y argument when fitting.")
        self.regressor.fit(x.T)
        self.coef_ = self.regressor.predict(identity(x.shape[1])).T

    def predict(self, x):
        """
        Parameters
        ----------
        x: numpy ndarray, shape (n_examples, n_features)
            Measurement data to be fit.
        """
        return super(DMDRegressor, self).predict(x.T).T
