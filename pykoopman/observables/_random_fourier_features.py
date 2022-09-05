"""
Random fourier features

"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._base import BaseObservables


class RandomFourierFeatures(BaseObservables):
    """Random Fourier Features observables.

    Here we only consider the following kernel:

        k(x,y) = exp(-gamma*||x-y||^2)

    if one include the state:

        k(x,y) = x^y + exp(-gamma*||x-y||^2)

    Notation in the original paper RR2007 is used.
        D: the number of terms in the monte carlo summation.

    """

    def __init__(self, include_state=True, gamma=1.0, D=100, random_state=None):
        self.include_state = include_state
        self.gamma = gamma
        self.D = D
        self.random_state = random_state
        super(RandomFourierFeatures, self).__init__()

    def fit(self, x, y=None):
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
                (self.n_output_features_, self.n_input_features_)
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
            self.measurement_matrix_ = np.linalg.lstsq(z, x)[0]
        return self

    def transform(self, x):
        check_is_fitted(self, "n_input_features_")

        z = np.zeros((x.shape[0], self.n_output_features_))
        z_rff = self._rff_lifting(x)

        if self.include_state:
            z[:, : x.shape[1]] = x
            z[:, x.shape[1] :] = z_rff
        else:
            z = z_rff
        return z

    def inverse(self, y):
        check_is_fitted(self, "n_output_features_")
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "y has the wrong number of features (columns)."
                f"Expected {self.n_output_features_}, received {y.shape[1]}"
            )

        return y @ self.measurement_matrix_

    def get_feature_names(self, input_features=None):
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
        # 2. get the feature vector z
        xw = np.dot(x, self.w)
        z_rff = np.hstack([np.cos(xw), np.sin(xw)])
        z_rff *= 1.0 / np.sqrt(self.D)
        return z_rff
