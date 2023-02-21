"""
Base class for specific observable classes
"""
from __future__ import annotations

import abc

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BaseObservables(TransformerMixin, BaseEstimator):
    """Base class for observable classes.

    Forces subclasses to implement 'fit', 'transform', and 'get_feature_names' functions
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Does not do anything for now.

        Parameters
        ----------
        X : numpy ndarray with shape [n_samples, n_features]

        Returns
        -------
        self : instance
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, X):
        """Transforms data.

        Parameters
        ----------
        X : numpy ndarray with shape [n_samples, n_features]
            Data X is transformed row-wise.

        Returns
        -------
        XT : numpy ndarray with shape [n_samples, n_output_features]
             Transformed data X into observables XT, where the number of
             observables or features is n_output_features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_names(self, input_features=None):
        """Return names of observables.

        Parameters
        ----------
        input_features : list of string of length n_features, optional
            Default list is "x0", "x1", ..., "xn", where n = n_features.

        Returns
        -------
        output_feature_names : list of string of length n_output_features
        """
        raise NotImplementedError

    def inverse(self, y):
        """Invert the transformation.

        This function satisfies
        :code:`self.inverse(self.transform(x)) == x`

        Parameters
        ----------
        y: array-like, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: array-like, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
            In this case, x is identical to y.
        """
        check_is_fitted(self, ["n_input_features_", "measurement_matrix_"])
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "Wrong number of input features."
                f"Expected y.shape[1] = {self.n_output_features_}; "
                f"instead y.shape[1] = {y.shape[1]}."
            )

        return y @ self.measurement_matrix_.T

    def __add__(self, other):
        if isinstance(self, ConcatObservables):
            return ConcatObservables(self.observables_list_ + [other])
        else:
            return ConcatObservables([self, other])

    @property
    def size(self):
        check_is_fitted(self)
        return self.n_output_features_


# learned from https://github.com/dynamicslab/pysindy/blob/
# d0d96f4466b9c16cdd349fdc515abe9081e5b2cf/pysindy/feature_library/base.py#L235


class ConcatObservables(BaseObservables):
    """Concatnate two `BaseObservables` instances  into one instance of
    `ConcatObservables`

    When two BaseObservables are concated, we will only keep the first
    one having the identity mapping, while the identity mapping in rest
    will be removed. Note that if the second to last observables are
    polynomial with `include_bias=True`, we need to remove bias feature
    as well.

    Parameters
    ----------
    observables_list_ : list, optional, default=None
        a list of `BaseObservables`

    Attributes
    ----------
    observables_list_ : list, optional, default=None
        a list of `BaseObservables`

    include_state : bool
        True if linear feature (i.e., system state) is included. This
        indicator can help to identify if redundant linear feature can
        be removed.

    n_input_features_ : int
        Dimension of input features, e.g., system state

    n_output_features_ : int
        Dimension of transformed/output features, e.g., observables

    n_consumed_samples : int
        Number of effective samples, e.g., time-delay will cause loss
        of the number of samples because those samples are stacked

    measurement_matrix_ : numpy.ndarray, shape (n_input_features_,
    n_output_features_)
        A row feature vector right multiply with `measurement_matrix_`
        will return the system state

    """

    def __init__(self, observables_list_=None):
        super(ConcatObservables, self).__init__()
        self.observables_list_ = observables_list_
        assert hasattr(
            self.observables_list_[0], "include_state"
        ), "first observable must have `include_state' attribute"
        self.include_state = self.observables_list_[0].include_state

    def fit(self, X, y=None):
        """Set up observable

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_input_features_)
            Measurement data to be fit.

        y : numpy.ndarray, optional, default=None
            Time-shifted measurement data to be fit

        Returns
        -------
        self: returns a fitted ``ConcatObservables`` instance
        """

        # first one must call fit of every observable in the observer list
        # so that n_input_features_ and n_output_features_ are defined
        for obs in self.observables_list_:
            obs.fit(X, y)

        self.n_input_features_ = X.shape[1]

        # total number of output features takes care of redundant identity features
        # for polynomial feature, we will remove the 1 as well if include_bias is true

        first_obs_output_features = self.observables_list_[0].n_output_features_
        s = 0
        for obs in self.observables_list_[1:]:
            assert hasattr(
                obs, "include_state"
            ), "observable Must have `include_state' attribute"
            if obs.include_state:
                s += obs.n_output_features_ - obs.n_input_features_
            else:
                s += obs.n_output_features_
            if getattr(obs, "include_bias", False) and hasattr(
                self.observables_list_[0], "include_bias"
            ):
                s -= 1

        self.n_output_features_ = first_obs_output_features + s
        #         # self.include_state=True

        # self.n_output_features_ = self.observables_list_[0].n_output_features_ + sum(
        #     [
        #         obs.n_output_features_ - obs.n_input_features_ - 1
        #         if getattr(obs, "include_bias", False)
        #         else obs.n_output_features_ - obs.n_input_features_
        #         for obs in self.observables_list_[1:]
        #     ]
        # )

        # take care of consuming samples in time delay observables: \
        # we will look for the largest delay
        max_n_consumed_samples = 0
        for obs in self.observables_list_:
            if hasattr(obs, "n_consumed_samples"):
                max_n_consumed_samples = max(
                    max_n_consumed_samples, obs.n_consumed_samples
                )
        self.n_consumed_samples = max_n_consumed_samples

        # C comes from the first observable
        self.measurement_matrix_ = np.zeros(
            [self.n_input_features_, self.n_output_features_]
        )
        first_obs_measurement_matrix = self.observables_list_[0].measurement_matrix_
        self.measurement_matrix_[
            : first_obs_measurement_matrix.shape[0],
            : first_obs_measurement_matrix.shape[1],
        ] = first_obs_measurement_matrix

        return self

    def transform(self, X):
        """Evaluate observable at `X`

        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_input_features_)
            Measurement data to be fit.

        Returns
        -------
        y: numpy.ndarray, shape (n_samples, n_output_features_)
            Evaluation of observables at `X`
        """

        # for obs in self.observables_list_:
        #     check_is_fitted(obs, "n_consumed_samples_")
        check_is_fitted(self, "n_input_features_")
        num_samples_updated = X.shape[0] - self.n_consumed_samples
        y_list = [self.observables_list_[0].transform(X)[-num_samples_updated:, :]]

        # only include those features that are not state
        y_rest_list = []
        for obs in self.observables_list_[1:]:
            if obs.include_state:
                y_new = obs.transform(X)[-num_samples_updated:, obs.n_input_features_ :]
            else:
                y_new = obs.transform(X)[-num_samples_updated:, :]
            if getattr(obs, "include_bias", False) and hasattr(
                self.observables_list_[0], "include_bias"
            ):
                y_new = y_new[:, 1:]

            y_rest_list.append(y_new)
        y_list += y_rest_list

        # y_list += [
        #     obs.transform(X)[-num_samples_updated:, obs.n_input_features_ :]
        #     for obs in self.observables_list_[1:]
        # ]
        y = np.hstack(y_list)
        return y

    def get_feature_names(self, input_features=None):
        """Return names of observables

        Parameters
        ----------
        input_features : list of string of length n_features, optional
            Default list is "x0", "x1", ..., "xn", where n = n_features.

        Returns
        -------
        output_feature_names : list of string of length n_output_features
        """

        # for obs in self.observables_list_:
        #     check_is_fitted(obs, "n_input_features_")
        check_is_fitted(self, "n_input_features_")

        concat_feature_names = self.observables_list_[0].get_feature_names()
        for obs in self.observables_list_[1:]:
            if getattr(obs, "include_bias", False):
                concat_feature_names += obs.get_feature_names()[
                    obs.n_input_features_ + 1 :
                ]
            else:
                concat_feature_names += obs.get_feature_names()[obs.n_input_features_ :]
        return concat_feature_names

    def inverse(self, y):
        """Invert the transformation to get system state `x`

        This function approximately (due to some of them use least-square)
        satisfies :code:`self.inverse(self.transform(x)) == x`

        Parameters
        ----------
        y: array-like, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: array-like, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
            In this case, x is identical to y.
        """

        # check_is_fitted(self, "n_consumed_samples")
        #
        # # if first observable has state, we just use it done.
        # obs1 = self.observables_list_[0]
        # if getattr(obs1, "include_bias", False):
        #     return y[:, 1 : self.n_input_features_ + 1]
        # else:
        #     return y[:, : self.n_input_features_]

        check_is_fitted(self, ["n_input_features_", "measurement_matrix_"])
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "Wrong number of input features."
                f"Expected y.shape[1] = {self.n_output_features_}; "
                f"instead y.shape[1] = {y.shape[1]}."
            )

        # dim_output_first_obs = self.observables_list_[0].n_output_features_
        x = y @ self.measurement_matrix_.T
        return x
