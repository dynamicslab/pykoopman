from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseObservables


class RandomFourierFeatures(BaseObservables):
    """Random Fourier Features observables.

    Here we only consider the following kernel:
        :math:`k(x,y) = exp(-gamma*\\|x-y\\|^2)`

    if one includes the system state:
        :math:`k(x,y) = x.T * y + exp(-gamma*\\|x-y\\|^2)`

    See the following reference for more details:
        `Rahimi, A., & Recht, B. (2007). "Random features for large-scale
        kernel machines". Advances in neural information processing systems
        , 20. <https://proceedings.neurips.cc/paper/2007/file/013a006f03dbc
        5392effeb8f18fda755-Paper.pdf>`_

    Parameters
    ----------
    include_state : bool
        `True` if includes state

    gamma : float
        Scale of Gaussian kernel

    D : int
        Number of random samples in Monte Carlo approximation

    random_state : int or NoneType, optional, default=None
        Seed of random number. Useful for repeatable experiments

    Attributes
    ----------
    include_state : bool
        `True` if includes state

    gamma : float
        Scale of Gaussian kernel

    D : int
        Number of random samples in Monte Carlo approximation

    random_state : int or NoneType, optional, default=None
        Seed of random number. Useful for repeatable experiments

    measurement_matrix_ : numpy.ndarray, shape (n_input_features_,
    n_output_features_)
        A row feature vector right multiply with `measurement_matrix_`
        will return the system state

    n_input_features_ : int
        Dimension of input features, e.g., system state

    n_output_features_ : int
        Dimension of transformed/output features, e.g., observables

    w : numpy.ndarray, shape (n_input_features_, D)
        The frequencies randomly sampled for random fourier features
    """

    def __init__(self, include_state=True, gamma=1.0, D=100, random_state=None):
        self.include_state = include_state
        self.gamma = gamma
        self.D = D
        self.random_state = random_state
        super(RandomFourierFeatures, self).__init__()

    def fit(self, x, y=None):
        """Set up observable

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_input_features_)
            Measurement data to be fit.

        y : numpy.ndarray, optional, default=None
            Time-shifted measurement data to be fit

        Returns
        -------
        self: returns a fitted ``RandomFourierFeatures`` instance
        """

        np.random.seed(self.random_state)

        self.n_input_features_ = x.shape[1]
        # although we have double the output dim, the convergence
        # rate is described in only self.n_components
        self.n_output_features_ = 2 * self.D

        if self.include_state:
            self.n_output_features_ += self.n_input_features_

        # 1. generate (n_feature, n_component) random w
        self.w = np.sqrt(2.0 * self.gamma) * np.random.normal(
            0, 1, [self.n_input_features_, self.D]
        )

        # 3. get the measurement_matrix to map back to state
        if self.include_state:
            self.measurement_matrix_ = np.zeros(
                (self.n_input_features_, self.n_output_features_)
            )
            self.measurement_matrix_[
                : self.n_input_features_, : self.n_input_features_
            ] = np.eye(self.n_input_features_)
        else:
            # we have to transform the data x in order to find a matrix by fitting
            # z = np.zeros((x.shape[0], self.n_output_features_))
            # z[:,:x.shape[1]] = x
            # z[:,x.shape[1]:] = self._rff_lifting(x)
            z = self._rff_lifting(x)
            self.measurement_matrix_ = np.linalg.lstsq(z, x)[0].T

        return self

    def transform(self, x):
        """Evaluate observable at `x`

        Parameters
        ----------
        x : numpy.ndarray, shape (n_samples, n_input_features_)
            Measurement data to be fit.

        Returns
        -------
        y: numpy.ndarray, shape (n_samples, n_output_features_)
            Evaluation of observables at `X`
        """

        check_is_fitted(self, "n_input_features_")
        z = np.zeros((x.shape[0], self.n_output_features_))
        z_rff = self._rff_lifting(x)
        if self.include_state:
            z[:, : x.shape[1]] = x
            z[:, x.shape[1] :] = z_rff
        else:
            z = z_rff

        return z

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

        check_is_fitted(self, "n_input_features_")

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_input_features_)]
        else:
            if len(input_features) != self.n_input_features_:
                raise ValueError(
                    "input_features must have n_input_features_ "
                    f"({self.n_input_features_}) elements"
                )

        if self.include_state:
            # very easy to make mistake... python pass list by reference OMG
            output_features = input_features[:]
        else:
            output_features = []
        output_features += [f"cos(w_{i}'x)/sqrt({self.D})" for i in range(self.D)] + [
            f"sin(w_{i}'x)/sqrt({self.D})" for i in range(self.D)
        ]

        return output_features

    def _rff_lifting(self, x):
        """Core algorithm that computes random fourier features

        Here we use the `cos` and `sin` transformations to get
        random fourier features. Other is also possible though.

        Parameters
        ----------
        x : numpy.ndarray
            system state

        Returns
        -------
        z_rff : numpy.ndarray, shape (n_samples, n_output_features_)
            Random fourier features evaluated on `x`
        """

        # 2. get the feature vector z
        xw = np.dot(x, self.w)
        z_rff = np.hstack([np.cos(xw), np.sin(xw)])
        z_rff *= 1.0 / np.sqrt(self.D)
        return z_rff
