"""
Base class for specific observable classes
"""
import abc
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

class BaseObservables(TransformerMixin,BaseEstimator):
    """
    Base class for observable classes.

    Forces subclasses to implement 'fit', 'transform', and 'get_feature_names' functions
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Does not do anything for now.

        Parameters
        ----------
        X : np.ndarray with shape [n_samples, n_features]

        Returns
        -------
        self : instance
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, X):
        """
        Transforms data.

        Parameters
        ----------
        X : np.ndarray with shape [n_samples, n_features]
            Data X is transformed row-wise.

        Returns
        -------
        XT : np.ndarray with shape [n_samples, n_output_features]
             Transformed data X into observables XT, where the number of
             observables or features is n_output_features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_names(self, input_features=None):
        """
        Return names of observables.

        Parameters
        ----------
        input_features : list of string of length n_features, optional
            Default list is "x0", "x1", ..., "xn", where n = n_features.

        Returns
        -------
        output_feature_names : list of string of length n_output_features
        """
        raise NotImplementedError

    @property
    def size(self):
        check_is_fitted(self)
        return self_n_output_features_