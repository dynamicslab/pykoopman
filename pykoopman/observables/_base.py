"""Module for base classes for specific observable classes."""
from __future__ import annotations

import abc

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BaseObservables(TransformerMixin, BaseEstimator):
    """
    Abstract base class for observable classes.

    This class defines the interface for observable classes. It uses
    the transformer interface from scikit-learn.
    """

    def __init__(self):
        """
        Initialize a BaseObservables instance.

        Initializes the parent classes with the super function.
        """
        super(BaseObservables, self).__init__()

    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Abstract method for fitting the observables.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray, optional): The target values.

        Raises:
            NotImplementedError: This method must be overwritten by any child class.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, X):
        """
        Abstract method for transforming the data.

        Args:
            X (np.ndarray): The input data.

        Raises:
            NotImplementedError: This method must be overwritten by any child class.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_names(self, input_features=None):
        """
        Abstract method for getting the names of the features.

        Args:
            input_features (list of str, optional): The names of the input features.

        Raises:
            NotImplementedError: This method must be overwritten by any child class.
        """
        raise NotImplementedError

    def inverse(self, y):
        """
        Inverse the transformation.

        Args:
            y (np.ndarray): The transformed data.

        Returns:
            np.ndarray: The original data.

        Raises:
            ValueError: If the shape of the input does not match the expected shape.
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
    """
    This class concatenates two or more `BaseObservables` instances into a single
    `ConcatObservables` instance.

    The concatenated observables are handled in such a way that only the first
    observable with the identity mapping is kept, while the identity mapping in
    the rest is removed. The same applies to observables that are polynomials with
    `include_bias=True`, in which case the bias feature is also removed.

    Args:
        observables_list_ (list, optional): A list of `BaseObservables` instances
        to concatenate. Defaults to None.

    Attributes:
        observables_list_ (list, optional): The list of `BaseObservables` instances
            that were concatenated. Defaults to None.
        include_state (bool): True if a linear feature (i.e., the system state) is
            included. This indicator can help to identify if a redundant linear feature
            can be removed.
        n_input_features_ (int): The dimensionality of the input features, e.g.,
            the system state.
        n_output_features_ (int): The dimensionality of the transformed/output
            features, e.g., the observables.
        n_consumed_samples (int): The number of effective samples. This can be less
            than the total number of samples due to time-delay stacking.
        measurement_matrix_ (numpy.ndarray): This matrix transforms a row feature
            vector to return the system state. Its shape is (n_input_features_,
                n_output_features_).

    Methods:
        fit(X, y=None): Calculates and stores important information such as the
            dimensions of the input and output features, the number of effective
            samples, and the measurement matrix.
        transform(X): Applies the transformation defined by the observables to
            input data.
        get_feature_names(input_features=None): Returns the names of the features
            after transformation.
        inverse(y): Applies the inverse transformation to the transformed data to
            recover the original system state.
    """

    def __init__(self, observables_list_=None):
        """Initializes a ConcatObservables instance.

        Args:
            observables_list_ (list, optional): A list of `BaseObservables` instances.
                If provided, the first observable must have an `include_state`
                attribute. The default value is None.

        Raises:
            AssertionError: If the first observable in `observables_list_` does not have
                an `include_state` attribute.
        """
        super(ConcatObservables, self).__init__()
        self.observables_list_ = observables_list_
        assert hasattr(
            self.observables_list_[0], "include_state"
        ), "first observable must have `include_state' attribute"
        self.include_state = self.observables_list_[0].include_state

    def fit(self, X, y=None):
        """Sets up observable by fitting the model to the data.

        This method fits each observable in the list to the data, determines the
        total number of output features, and sets up the measurement matrix.

        Args:
            X (numpy.ndarray): Measurement data to be fit, with shape (n_samples,
                n_input_features_).
            y (numpy.ndarray, optional): Time-shifted measurement data to be fit.
                Default is None.

        Returns:
            ConcatObservables: A fitted instance of the class.

        Raises:
            AssertionError: If any observable in the list does not have an
                `include_state` attribute, or if the shape of the temporary least
                squares solution does not match the shape of the measurement matrix.
        """

        # first, one must call fit of every observable in the observer list
        # so that n_input_features_ and n_output_features_ are defined
        for obs in self.observables_list_:
            obs.fit(X, y)

        self.n_input_features_ = X.shape[1]

        # total number of output features takes care of redundant identity features
        # for polynomial feature, we will remove the 1 as well if include_bias is true

        first_obs = self.observables_list_[0]
        s = 0
        obs_list_contain_state_counter = 1 if first_obs.include_state else 0
        obs_list_contain_bias_counter = (
            1 if getattr(first_obs, "include_bias", False) else 0
        )
        for obs in self.observables_list_[1:]:
            assert hasattr(obs, "include_state"), (
                "observable Must have `include_state' " "attribute"
            )
            if obs_list_contain_state_counter > 1 and obs.include_state:
                s += obs.n_output_features_ - obs.n_input_features_
            else:
                s += obs.n_output_features_
            if obs_list_contain_bias_counter > 1 and getattr(
                obs, "include_bias", False
            ):
                s -= 1
            obs_list_contain_state_counter += 1 if obs.include_state is True else 0
            obs_list_contain_bias_counter += (
                1 if getattr(obs, "include_bias", False) else 0
            )

        self.n_output_features_ = first_obs.n_output_features_ + s

        # take care of consuming samples in time delay observables: \
        # we will look for the largest delay
        max_n_consumed_samples = 0
        for obs in self.observables_list_:
            if hasattr(obs, "n_consumed_samples"):
                max_n_consumed_samples = max(
                    max_n_consumed_samples, obs.n_consumed_samples
                )
        self.n_consumed_samples = max_n_consumed_samples

        # choosing measurement_matrix
        self.measurement_matrix_ = np.zeros(
            [self.n_input_features_, self.n_output_features_]
        )
        # 1. if any observable has `include_state` == True
        if any([obs.include_state for obs in self.observables_list_]) is True:
            jj = 0
            for i in range(len(self.observables_list_)):
                jcol = self.observables_list_[i].measurement_matrix_.shape[1]
                if self.observables_list_[i].include_state is True:
                    break
                jj += jcol
            self.measurement_matrix_[:, jj : jj + jcol] = self.observables_list_[
                i
            ].measurement_matrix_
        else:
            g = self.transform(X)
            tmp = np.linalg.lstsq(g, X)[0].T
            assert tmp.shape == self.measurement_matrix_.shape
            self.measurement_matrix_ = tmp

        # 1. if first observable does not contain include state but others do
        # then we will use the nearest one's measurement matrix

        # otherwise,

        # C comes from the first observable

        # first_obs_measurement_matrix = self.observables_list_[0].measurement_matrix_
        # self.measurement_matrix_[:first_obs_measurement_matrix.shape[0],
        # :first_obs_measurement_matrix.shape[1],] = first_obs_measurement_matrix

        return self

    def transform(self, X):
        """Evaluate observable at `X`.

        This method checks if the model is fitted and then evaluates the observables
        at the provided data, excluding features that are state or bias based on
        certain conditions.

        Args:
            X (numpy.ndarray): Measurement data to be fit, with shape (n_samples,
                n_input_features_).

        Returns:
            y (numpy.ndarray): Evaluation of observables at `X`, with shape (n_samples,
                n_output_features_).

        Raises:
            NotFittedError: If the model is not fitted yet.
        """

        # for obs in self.observables_list_:
        #     check_is_fitted(obs, "n_consumed_samples_")
        check_is_fitted(self, "n_consumed_samples")
        num_samples_updated = X.shape[0] - self.n_consumed_samples
        first_obs = self.observables_list_[0]
        obs_list_contain_state_counter = 1 if first_obs.include_state else 0
        obs_list_contain_bias_counter = (
            1 if getattr(first_obs, "include_bias", False) else 0
        )
        y_list = [first_obs.transform(X)[-num_samples_updated:, :]]

        # only include those features that are not state
        y_rest_list = []
        for obs in self.observables_list_[1:]:
            if obs_list_contain_state_counter > 1 and obs.include_state:
                y_new = obs.transform(X)[-num_samples_updated:, obs.n_input_features_ :]
            else:
                y_new = obs.transform(X)[-num_samples_updated:, :]
            if obs_list_contain_bias_counter > 1 and getattr(
                obs, "include_bias", False
            ):
                y_new = y_new[:, 1:]
            obs_list_contain_state_counter += 1 if obs.include_state else 0
            obs_list_contain_bias_counter += (
                1 if getattr(obs, "include_bias", False) else 0
            )

            y_rest_list.append(y_new)
        y_list += y_rest_list

        # y_list += [
        #     obs.transform(X)[-num_samples_updated:, obs.n_input_features_ :]
        #     for obs in self.observables_list_[1:]
        # ]
        y = np.hstack(y_list)
        return y

    def get_feature_names(self, input_features=None):
        """Return names of observables.

        This method returns a list of feature names, which are created by
        concatenating the feature names from all observables in the list.

        Args:
            input_features (list of str, optional): Default list is "x0", "x1", ...,
            "xn", where n = n_features. Defaults to None.

        Returns:
            list of str: List of feature names of length n_output_features.

        Raises:
            NotFittedError: If the model is not fitted yet.
        """
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
        """Invert the transformation to get system state `x`.

        This function approximately (due to some of them use least-square)
        satisfies :code:`self.inverse(self.transform(x)) == x`.

        Args:
            y (numpy.ndarray): Data to which to apply the inverse.
                Shape must be (n_samples, n_output_features).
                Must have the same number of features as the transformed data.

        Returns:
            numpy.ndarray: Output of inverse map applied to y.
                Shape will be (n_samples, n_input_features).
                In this case, x is identical to y.

        Raises:
            NotFittedError: If the model is not fitted yet.
            ValueError: If the number of features in `y` does not match
                `n_output_features_`.

        """

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
