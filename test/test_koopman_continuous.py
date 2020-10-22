import pytest
from sklearn.utils.validation import check_is_fitted

from pykoopman import KoopmanContinuous
from pykoopman.differentiation import Derivative


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_random"), pytest.lazy_fixture("data_random_complex")],
)
def test_derivative_integration(data):
    x = data

    diff = Derivative(kind="finite_difference", k=1)
    model = KoopmanContinuous(differentiator=diff)

    model.fit(x)
    check_is_fitted(model)
