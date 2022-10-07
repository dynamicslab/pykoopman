from sklearn.base import BaseEstimator


class BaseRegressor(BaseEstimator):
    """
    Wrapper class for PyKoopman regressors.

    Parameters
    ----------
    regressor: regressor object
        A regressor object implementing ``fit`` and ``predict`` methods.
    """

    def __init__(self, regressor):
        # check .fit
        if not hasattr(regressor, "fit") or not callable(getattr(regressor, "fit")):
            raise AttributeError("regressor does not have a callable fit method")
        # check .predict
        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            raise AttributeError("regressor does not have a callable predict method")
        self.regressor = regressor

    def fit(self, x, y=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
