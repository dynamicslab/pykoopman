import numpy as np
from sklearn.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """
    Wrapper class for PyKoopman regressors.
    """

    def __init__(self, regressor):
        if not hasattr(regressor, "fit") or not callable(getattr(regressor, "fit")):
            raise AttributeError("regressor does not have a callable fit method")
        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            raise AttributeError("regressor does not have a callable predict method")

        self.regressor = regressor

    def fit(self, x, y):
        self.regressor.fit(x, y)

    def predict(self, x):
        prediction = self.regressor.predict(x)
        if prediction.ndim == 1:
            return prediction[:, np.newaxis]
        else:
            return prediction
