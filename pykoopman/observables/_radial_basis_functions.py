"""
Radial basis function observables
"""
import numpy as np
from numpy import empty
from numpy import random
from sklearn.utils.validation import check_is_fitted

from ..common import validate_input
from ._base import BaseObservables


class RadialBasisFunction(BaseObservables):
    r"""
    Radial basis functions as observables. Observables formed as radial basis
    functions of the state variables and interpreting them as new state
    variables.

    For example, the single state variables :math:`[x(t)]` could be
    evaluated using multiple centers, yielding a new set of
    observables:

    .. math::
        [z_1(t), z_2(t), z_3(t)]
    where, e.g., the Gaussian type rbf is used
    .. math::
        z_i = exp(-eps^2*(x(t)-c_i)^2)

    This example corresponds to taking :code:`type =` :math:`gauss`
    and :code:`n_centers = 3`.

    See the following references for more information.

        Williams, Matthew O and Kevrekidis, Ioannis G and Rowley,
        Clarence W
        "A data-driven approximation of the {K}oopman operator:
        extending dynamic mode decomposition."
        Journal of Nonlinear Science 6 (2015): 1307-1346

        Williams, Matthew O and Rowley, Clarence W and Kevrekidis,
        Ioannis G
        "A Kernel Approach to Data-Driven {K}oopman Spectral Analysis."
        Journal of Computational Dynamics 2.2 (2015): 247-265

        Korda, Milan and Mezic, Igor
        "Linear predictors for nonlinear dynamical systems:
        Koopman operator meets model predictive control."
        Automatica 93 (2018): 149-160

    Parameters
    ----------
    rbf_type: string, optional (default 'gauss')
        The type of radial basis functions to be used.
        Options are: 'gauss', 'thinplate', 'invquad',
                     'invmultquad', 'polyharmonic'

    n_centers: nonnegative integer, optional (default 10)
        The number of centers to compute rbf with.

    centers: numpy array, optional (default uniformly distributed over input data)
        The centers to compute rbf with.

    kernel_width: float, optional (default 1.0)
        The kernel width for Gaussian rbfs.

    polyharmonic_coeff: float, optional (default 1.0)
        The polyharmonic coefficient for polyharmonic rbfs.

    include_states: bool, optional (default True)
        Includes the input coordinates as additional coordinates in the observable.

    Attributes
    ----------
    n_input_features_ : int
        Number of input features.

    n_output_features_ : int
        Number of output features = Number of centers plus number of input features.

    """

    def __init__(
        self,
        rbf_type="gauss",
        n_centers=10,
        centers=None,
        kernel_width=1.0,
        polyharmonic_coeff=1.0,
        include_states=True,
    ):
        super().__init__()
        if type(rbf_type) != str:
            raise TypeError("rbf_type must be a string")
        if type(n_centers) != int:
            raise TypeError("n_centers must be an int")
        if n_centers < 0:
            raise ValueError("n_centers must be a nonnegative int")
        if kernel_width < 0:
            raise ValueError("kernel_width must be a nonnegative float")
        if polyharmonic_coeff < 0:
            raise ValueError("polyharmonic_coeff must be a nonnegative float")
        if rbf_type not in [
            "thinplate",
            "gauss",
            "invquad",
            "invmultquad",
            "polyharmonic",
        ]:
            raise ValueError("rbf_type not of available type")
        if type(include_states) != bool:
            raise TypeError("include_states must be a boolean")
        if centers is not None:
            if int(n_centers) not in centers.shape():
                raise ValueError(
                    "n_centers is not equal to centers.shape[1]. "
                    "centers must be of shape (n_input_features, "
                    "n_centers). "
                )
        self.rbf_type = rbf_type
        self.n_centers = int(n_centers)
        self.centers = centers
        self.kernel_width = kernel_width
        self.polyharmonic_coeff = polyharmonic_coeff
        self.include_states = include_states

    def fit(self, x, y=None):
        """
        Fit to measurement data.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be fit.

        y: None
            Dummy parameter retained for sklearn compatibility.

        Returns
        -------
        self: returns a fit :class:`pykoopman.observables.RadialBasisFunction` instance
        """
        n_samples, n_features = validate_input(x).shape

        self.n_samples_ = n_samples
        self.n_input_features_ = n_features
        if self.include_states is True:
            self.n_output_features_ = n_features * 1 + self.n_centers
        elif self.include_states is False:
            self.n_output_features_ = self.n_centers

        x = validate_input(x)

        if x.shape[1] != self.n_input_features_:
            raise ValueError(
                "Wrong number of input features. "
                f"Expected x.shape[1] = {self.n_input_features_}; "
                f"instead x.shape[1] = {x.shape[1]}."
            )

        if self.centers is None:
            # Uniformly distributed centers
            self.centers = random.rand(self.n_input_features_, self.n_centers)
            # Change range to range of input features' range
            for feat in range(self.n_input_features_):
                xminmax = self._minmax(x[:, feat])

                # Map to range [0,1]
                self.centers[feat, :] = (
                    self.centers[feat, :] - min(self.centers[feat, :])
                ) / (max(self.centers[feat, :]) - min(self.centers[feat, :]))
                # Scale to input features' range
                self.centers[feat, :] = (
                    self.centers[feat, :] * (xminmax[1] - xminmax[0]) + xminmax[0]
                )

        xlift = self._rbf_lifting(x)
        self.measurement_matrix_ = x.T @ np.linalg.pinv(xlift.T)

        return self

    def transform(self, x):
        """
        Apply radial basis function transformation to the data.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Measurement data to be transformed.
            It is assumed that rows correspond to examples, which are not required to
            be equi-spaced in time or in sequential order.

        Returns
        -------
        y: array-like, shape (n_samples, n_output_features)
            Transformed data.
        """
        check_is_fitted(self, ["n_input_features_", "centers"])
        x = validate_input(x)

        if x.shape[1] != self.n_input_features_:
            raise ValueError(
                "Wrong number of input features. "
                f"Expected x.shape[1] = {self.n_input_features_}; "
                f"instead x.shape[1] = {x.shape[1]}."
            )

        y = self._rbf_lifting(x)
        return y

    def inverse(self, y):
        """
        Invert the transformation using the fitted measurement matrix.

        This function satisfies
        :code:`self.inverse(self.transform(x)) ~= x[self._n_consumed_samples:]`

        Parameters
        ----------
        y: array-like, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: array-like, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
        """
        check_is_fitted(self, ["n_input_features_", "measurement_matrix_"])
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "Wrong number of input features."
                f"Expected y.shape[1] = {self.n_out_features_}; "
                f"instead y.shape[1] = {y.shape[1]}."
            )

        return y @ self.measurement_matrix_.T

    def get_feature_names(self, input_features=None):
        """
        Get the names of the output features.

        Parameters
        ----------
        input_features: list of string, length n_input_features,\
         optional (default None)
            String names for input features, if available. By default,
            the names "x0", "x1", ... ,"xn_input_features" are used.

        Returns
        -------
        output_feature_names: list of string, length n_ouput_features
            Output feature names.
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

        output_features = []
        if self.include_states is True:
            output_features.extend([f"{xi}(t)" for xi in input_features])
        output_features.extend([f"phi(x(t)-c{i})" for i in range(
                self.n_centers)])
        return output_features

    def _rbf_lifting(self, x):
        n_samples = x.shape[0]
        y = empty(
            (n_samples, self.n_output_features_),
            dtype=x.dtype,
        )

        y_index = 0
        if self.include_states is True:
            y[:, : self.n_input_features_] = x
            y_index = self.n_input_features_

        for index_of_center in range(self.n_centers):
            C = self.centers[:, index_of_center]
            r_squared = np.sum((x - C[np.newaxis, :]) ** 2, axis=1)

            match self.rbf_type:
                case "thinplate":
                    y_ = r_squared * np.log(np.sqrt(r_squared))
                    y_[np.isnan(y_)] = 0
                case "gauss":
                    y_ = np.exp(-self.kernel_width**2 * r_squared)
                case "invquad":
                    y_ = np.reciprocal(1 + self.kernel_width**2 * r_squared)
                case "invmultquad":
                    y_ = np.reciprocal(np.sqrt(1 + self.kernel_width**2 * r_squared))
                case "polyharmonic":
                    y_ = r_squared ** (self.polyharmonic_coeff / 2) * np.log(
                        np.sqrt(r_squared)
                    )
                case _:
                    # if none of the above cases match:
                    raise ValueError("provided rbf_type not available")

            y[:, y_index + index_of_center] = y_

        return y

    def _minmax(self, x):
        min_val = min(x)
        max_val = max(x)
        return (min_val, max_val)
