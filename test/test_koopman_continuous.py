import pytest
from pydmd import DMD
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
    dmd = DMD(svd_rank=2)
    model = KoopmanContinuous(differentiator=diff, regressor=dmd)

    model.fit(x)
    check_is_fitted(model)
