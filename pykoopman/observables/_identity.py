"""
Linear observables
"""
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .observables import BaseObservables

class Identity(BaseObservables):
    """Linear observables"""

    def __init__(self):
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
