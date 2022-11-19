from sklearn.compose import TransformedTargetRegressor


class EnsembleBaseRegressor(TransformedTargetRegressor):
    """
    Wrapper class for PyKoopman regressors using ensemble or non-consecutive training
    data.

    Parameters
    ----------
    regressor : sklearn.base.BaseEstimator
        A regressor object implementing ``fit`` and ``predict`` methods.

    func : function
        Function to apply to `y` before passing to :meth:`fit`. Cannot be set
        at the same time as `transformer`. The function needs to return a
        2-dimensional array. If `func is None`, the function used will be the
        identity function.

    inverse_func : function
        Function to apply to the prediction of the regressor. Cannot be set at
        the same time as `transformer`. The function needs to return a
        2-dimensional array. The inverse function is used to return
        predictions to the same space of the original training labels.
    """

    def __init__(self, regressor, func, inverse_func):
        super().__init__(regressor=regressor, func=func, inverse_func=inverse_func)
        if not hasattr(regressor, "fit") or not callable(getattr(regressor, "fit")):
            raise AttributeError("regressor does not have a callable fit method")
        if not hasattr(regressor, "predict") or not callable(
            getattr(regressor, "predict")
        ):
            raise AttributeError("regressor does not have a callable predict method")
