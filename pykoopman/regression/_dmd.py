from warnings import warn
import numpy as np
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
        prediction = self.regressor.predict(x.T).T
        if prediction.ndim == 1:
            return prediction[:, np.newaxis]
        else:
            return prediction
