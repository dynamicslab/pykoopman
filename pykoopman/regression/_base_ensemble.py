"""module for handling a ensemble of x-x' pair.

Manual changes are made to add support to complex numeric data
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
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

    def fit(self, X, y, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        **fit_params : dict
            Parameters passed to the `fit` method of the underlying
            regressor.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # y = check_array(
        #     y,
        #     accept_sparse=False,
        #     force_all_finite=True,
        #     ensure_2d=False,
        #     dtype="numeric",
        #     allow_nd=True,
        # )

        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._training_dim = y.ndim

        # transformers are designed to modify X which is 2d dimensional, we
        # need to modify y accordingly.
        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)

        # transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        # FIXME: a FunctionTransformer can return a 1D array even when validate
        # is set to True. Therefore, we need to check the number of dimension
        # first.
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        if self.regressor is None:
            from sklearn.linear_model import LinearRegression

            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        self.regressor_.fit(X, y_trans, **fit_params)

        if hasattr(self.regressor_, "feature_names_in_"):
            self.feature_names_in_ = self.regressor_.feature_names_in_

        return self

    def _fit_transformer(self, y):
        """Check transformer and fit transformer.

        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).

        """
        if self.transformer is not None and (
            self.func is not None or self.inverse_func is not None
        ):
            raise ValueError(
                "'transformer' and functions 'func'/'inverse_func' cannot both be set."
            )
        elif self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:
            if self.func is not None and self.inverse_func is None:
                raise ValueError(
                    "When 'func' is provided, 'inverse_func' must also be provided"
                )
            self.transformer_ = FunctionTransformer(
                func=self.func,
                inverse_func=self.inverse_func,
                validate=True,
                check_inverse=self.check_inverse,
            )
        # XXX: sample_weight is not currently passed to the
        # transformer. However, if transformer starts using sample_weight, the
        # code should be modified accordingly. At the time to consider the
        # sample_prop feature, it is also a good use case to be considered.
        self.transformer_.fit(y)
        # if self.check_inverse:
        #     idx_selected = slice(None, None, max(1, y.shape[0] // 10))
        #     y_sel = _safe_indexing(y, idx_selected)
        #     y_sel_t = self.transformer_.transform(y_sel)
        #     if not np.allclose(y_sel, self.transformer_.inverse_transform(y_sel_t)):
        #         warnings.warn(
        #             "The provided functions or transformer are"
        #             " not strictly inverse of each other. If"
        #             " you are sure you want to proceed regardless"
        #             ", set 'check_inverse=False'",
        #             UserWarning,
        #         )


class FunctionTransformer(TransformerMixin, BaseEstimator):
    """Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    .. versionadded:: 0.17

    Read more in the :ref:`User Guide <function_transformer>`.

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.

    inverse_func : callable, default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.

    validate : bool, default=False
        Indicate that the input X array should be checked before calling
        ``func``. The possibilities are:

        - If False, there is no input validation.
        - If True, then X will be converted to a 2-dimensional NumPy array or
          sparse matrix. If the conversion is not possible an exception is
          raised.

        .. versionchanged:: 0.22
           The default of ``validate`` changed from True to False.

    accept_sparse : bool, default=False
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.

    check_inverse : bool, default=True
       Whether to check that or ``func`` followed by ``inverse_func`` leads to
       the original inputs. It can be used for a sanity check, raising a
       warning when the condition is not fulfilled.

       .. versionadded:: 0.20

    kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to func.

        .. versionadded:: 0.18

    inv_kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to inverse_func.

        .. versionadded:: 0.18

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`. Defined only when
        `validate=True`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `validate=True`
        and `X` has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MaxAbsScaler : Scale each feature by its maximum absolute value.
    StandardScaler : Standardize features by removing the mean and
        scaling to unit variance.
    LabelBinarizer : Binarize labels in a one-vs-all fashion.
    MultiLabelBinarizer : Transform between iterable of iterables
        and a multilabel format.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import FunctionTransformer
    >>> transformer = FunctionTransformer(np.log1p)
    >>> X = np.array([[0, 1], [2, 3]])
    >>> transformer.transform(X)
    array([[0.       , 0.6931...],
           [1.0986..., 1.3862...]])
    """

    def __init__(
        self,
        func=None,
        inverse_func=None,
        *,
        validate=False,
        accept_sparse=False,
        check_inverse=True,
        kw_args=None,
        inv_kw_args=None,
    ):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def _check_input(self, X, *, reset):
        # if self.validate:
        #     return self._validate_data(X, accept_sparse=self.accept_sparse,
        #     reset=reset)
        return X

    def _check_inverse_transform(self, X):
        """Check that func and inverse_func are the inverse."""
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        # X_round_trip = self.inverse_transform(self.transform(X[idx_selected]))
        self.inverse_transform(self.transform(X[idx_selected]))
        # if not _allclose_dense_sparse(X[idx_selected], X_round_trip):
        #     warnings.warn(
        #         "The provided functions are not strictly"
        #         " inverse of each other. If you are sure you"
        #         " want to proceed regardless, set"
        #         " 'check_inverse=False'.",
        #         UserWarning,
        #     )

    def fit(self, X, y=None):
        """Fit transformer by checking X.

        If ``validate`` is ``True``, ``X`` will be checked.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FunctionTransformer class instance.
        """
        X = self._check_input(X, reset=True)
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(X)
        return self

    def transform(self, X):
        """Transform X using the forward function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        X = self._check_input(X, reset=False)
        return self._transform(X, func=self.func, kw_args=self.kw_args)

    def inverse_transform(self, X):
        """Transform X using the inverse function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        # if self.validate:
        #     X = check_array(X, accept_sparse=self.accept_sparse)
        return self._transform(X, func=self.inverse_func, kw_args=self.inv_kw_args)

    def _transform(self, X, func=None, kw_args=None):
        if func is None:
            func = _identity

        return func(X, **(kw_args if kw_args else {}))

    def __sklearn_is_fitted__(self):
        """Return True since FunctionTransfomer is stateless."""
        return True

    def _more_tags(self):
        return {"no_validation": not self.validate, "stateless": True}


def _identity(X):
    """The identity function."""
    return X
