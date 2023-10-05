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
    This class serves as a wrapper for PyKoopman regressors that utilize ensemble or
    non-consecutive training data.

    `EnsembleBaseRegressor` inherits from `TransformedTargetRegressor` and checks
    whether the provided regressor object implements the `fit` and `predict` methods.

    Attributes:
        regressor (sklearn.base.BaseEstimator): A regressor object that implements
            `fit` and `predict` methods.
        func (function): A function to apply to the target `y` before passing it to
            the `fit` method. The function must return a 2-dimensional array.
            If `func` is `None`, the identity function is used.
        inverse_func (function): A function to apply to the prediction of the
            regressor. This function is used to return predictions to the same space
            as the original training labels. It must return a 2-dimensional array.

    Raises:
        AttributeError: If the regressor does not have a callable `fit` or
            `predict` method.
        ValueError: If both `transformer` and functions `func`/`inverse_func`
            are set, or if 'func' is provided while 'inverse_func' is not.

    Note:
        This class does not implement the `fit` method on its own, instead, it checks
        the methods of the provided regressor object and raises an AttributeError if
        the required methods are not present or not callable. It also performs some
        pre-processing on the target values `y` before fitting the regressor, and
        provides additional checks and warnings for the transformer and inverse
        functions.
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
        """
        Fits the model according to the given training data.

        Args:
            X (array-like or sparse matrix of shape (n_samples, n_features)):
                Training vector, where `n_samples` is the number of samples and
                `n_features` is the number of features.
            y (array-like of shape (n_samples,)): Target values.
            **fit_params (dict): Additional parameters passed to the `fit` method of
                the underlying regressor.

        Returns:
            self: The fitted estimator.

        Raises:
            ValueError: If 'transformer' and functions 'func'/'inverse_func' are both
                set, or if 'func' is provided while 'inverse_func' is not.

        Note:
            This method transforms the target `y` before fitting the regressor and
            performs additional checks and warnings for the transformer and inverse
            functions.
        """

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
        """
        Checks the transformer and fits it.

        This method creates the default transformer if necessary, fits it, and
        performs additional inverse checks on a subset (optional).

        Args:
            y (array-like): The target values.

        Raises:
            ValueError: If both 'transformer' and functions 'func'/'inverse_func'
                are set, or if 'func' is provided while 'inverse_func' is not.

        Note:
            The method does not currently pass 'sample_weight' to the transformer.
            However, if the transformer starts using 'sample_weight', the code should
            be modified accordingly. During the consideration of the 'sample_prop'
            feature, this is also a good use case to consider.
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

    This class forwards its X (and optionally y) arguments to a user-defined function
    or function object and returns the result of this function. This is useful for
    stateless transformations such as taking the log of frequencies, doing custom
    scaling, etc.

    Note: If a lambda is used as the function, then the resulting transformer will
        not be pickleable.

    Attributes:
        func (callable): The callable to use for the transformation. This will be
            passed the same arguments as transform, with args and kwargs forwarded.
            If func is None, then func will be the identity function.
        inverse_func (callable): The callable to use for the inverse transformation.
            This will be passed the same arguments as inverse transform, with args
            and kwargs forwarded. If inverse_func is None, then inverse_func will be
            the identity function.
        validate (bool): Indicate that the input X array should be checked before
            calling func. The default is False.
        accept_sparse (bool): Indicate that func accepts a sparse matrix as input.
            The default is False.
        check_inverse (bool): Whether to check that or func followed by inverse_func
            leads to the original inputs. The default is True.
        kw_args (dict): Dictionary of additional keyword arguments to pass to func.
        inv_kw_args (dict): Dictionary of additional keyword arguments to pass to
            inverse_func.
        n_input_features_ (int): Number of features seen during fit. Defined only
            when validate=True.
        feature_names_in_ (ndarray): Names of features seen during fit. Defined only
            when validate=True and X has feature names that are all strings.

    Examples:
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
        """Initialize the FunctionTransformer instance.

        Args:
            func (callable, optional): The callable to use for the transformation.
                This will be passed the same arguments as transform, with args and
                kwargs forwarded. If func is None, then
                func will be the identity function. Defaults to None.
            inverse_func (callable, optional): The callable to use for the inverse
                transformation. This will be passed the same arguments as inverse
                transform, with args and kwargs forwarded. If inverse_func is None, then
                inverse_func will be the identity function. Defaults to None.
            validate (bool, optional): Indicate that the input X array should be
                checked before calling func. Defaults to False.
            accept_sparse (bool, optional): Indicate that func accepts a sparse matrix
                as input. Defaults to False.
            check_inverse (bool, optional): Whether to check that func followed by
                inverse_func leads to the original inputs. Defaults to True.
            kw_args (dict, optional): Dictionary of additional keyword arguments to
                pass to func. Defaults to None.
            inv_kw_args (dict, optional): Dictionary of additional keyword arguments
                to pass to inverse_func. Defaults to None.
        """
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def _check_input(self, X, *, reset):
        """Checks the input X. If validation is enabled, it validates the data.

        Args:
            X (array-like): Input data to be checked/validated.
            reset (bool): Flag indicating whether to reset the validation.

        Returns:
            array-like: The original input data, possibly validated if `validate`
                attribute is set to True.
        """
        # if self.validate:
        #     return self._validate_data(X, accept_sparse=self.accept_sparse,
        #     reset=reset)
        return X

    def _check_inverse_transform(self, X):
        """Checks if the provided functions are the inverse of each other.

        Selects a subset of X and performs a round trip transformation: forward
        transform followed by inverse transform. Raises a warning if the round trip
        does not return the original inputs.

        Args:
            X (array-like): Input data to be checked for inverse transform consistency.
        """
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
        """Fits transformer by checking X.

        If ``validate`` is ``True``, ``X`` will be checked. Also checks if the provided
        functions are the inverse of each other if `check_inverse` is set to True.

        Args:
            X (array-like): The data to fit. Shape should be (n_samples, n_features).
            y (None, optional): Ignored. Not used, present here for API consistency by
                convention.

        Returns:
            FunctionTransformer: The fitted transformer.
        """
        X = self._check_input(X, reset=True)
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(X)
        return self

    def transform(self, X):
        """Transforms X using the forward function.

        Args:
            X (array-like): The data to transform. Shape should be (n_samples,
                n_features).

        Returns:
            array-like: Transformed data with same shape as input.
        """
        X = self._check_input(X, reset=False)
        return self._transform(X, func=self.func, kw_args=self.kw_args)

    def inverse_transform(self, X):
        """Transforms X using the inverse function.

        Args:
            X (array-like): The data to inverse transform. Shape should be
                (n_samples, n_features).

        Returns:
            array-like: Inverse transformed data with the same shape as input.
        """
        # if self.validate:
        #     X = check_array(X, accept_sparse=self.accept_sparse)
        return self._transform(X, func=self.inverse_func, kw_args=self.inv_kw_args)

    def _transform(self, X, func=None, kw_args=None):
        """Applies the given function to the data X.

        Args:
            X (array-like): The data to transform. Shape should be (n_samples,
                n_features).
            func (callable, optional): The function to apply. If None, identity
                function is used.
            kw_args (dict, optional): Additional arguments to pass to the function.

        Returns:
            array-like: Transformed data with the same shape as input.
        """
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
