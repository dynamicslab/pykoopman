from sklearn.compose import TransformedTargetRegressor


class EnsembleBaseRegressor(TransformedTargetRegressor):
    """
    Wrapper class for PyKoopman regressors using ensemble or non-consecutive training
    data.

    Parameters
    ----------
    regressor: regressor object
        A regressor object implementing ``fit`` and ``predict`` methods.
    """

    def __init__(self, regressor, func, inverse_func):
        super().__init__(regressor=regressor, func=func, inverse_func=inverse_func)
        if not hasattr(regressor, "fit") or not callable(getattr(regressor, "fit")):
            raise AttributeError("regressor does not have a callable fit method")
        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            raise AttributeError("regressor does not have a callable predict method")
