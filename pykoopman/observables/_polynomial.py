"""
Polynomial observables
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted


class Polynomial(PolynomialFeatures):
    """Polynomial observables

    Parameters
    ----------
    degree : integer
        The degree of the polynomial features. Default = 2.
    interaction_only : boolean, default = False
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).
    include_bias : boolean
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
    order : str in {'C', 'F'}, default 'C'
        Order of output array in the dense case. 'F' order is faster to
        compute, but may slow down subsequent estimators.
    """

    def __init__(self, degree=2, interaction_only=False, include_bias=True, order="C"):
        if degree == 0:
            raise ValueError(
                "degree must be at least 1, otherwise inverse cannot be computed"
            )
        super(Polynomial, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
        )

    def fit(self, x, y=None):
        return super(Polynomial, self).fit(x.real, y)

    def inverse(self, y):
        """
        Invert the transformation.

        This function satisfies
        :code:`self.inverse(self.transform(x)) == x`

        Parameters
        ----------
        y: numpy.ndarray, shape (n_samples, n_output_features)
            Data to which to apply the inverse.
            Must have the same number of features as the transformed data

        Returns
        -------
        x: numpy.ndarray, shape (n_samples, n_input_features)
            Output of inverse map applied to y.
        """
        check_is_fitted(self, "n_output_features_")
        if y.shape[1] != self.n_output_features_:
            raise ValueError(
                "y has the wrong number of features (columns)."
                f"Expected {self.n_output_features_}, received {y.shape[1]}"
            )
        if self.include_bias:
            return y[:, 1 : self.n_input_features_ + 1]
        else:
            return y[:, : self.n_input_features_]
