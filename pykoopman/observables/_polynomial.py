"""
Polynomial observables
"""
from sklearn.preprocessing import PolynomialFeatures


class Polynomial(PolynomialFeatures):
    """Polynomial observables"""

    def __init__(self, degree=2, interaction_only=False, include_bias=True, order="C"):
        super(Polynomial, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
        )
        # Do other setup that we require

    def inverse(y):
        """Invert the transformation.
        I.e. a Polynomial object :code:`p` should satisfy
        :code:`x == p.inverse(p.transform(x))`
        """
        pass
