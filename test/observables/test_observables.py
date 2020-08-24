# TODO: add unit test checking that model.inverse(model.transform(x)) == x
import pytest
from numpy import linspace
from numpy import stack
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pykoopman.observables import Identity
from pykoopman.observables import Polynomial


@pytest.fixture
def data_small():
    t = linspace(0, 5, 10)
    return stack((t, t ** 2), axis=1)


@pytest.mark.parametrize("observables", [Identity(), Polynomial()])
def test_if_fitted(observables, data_random):
    x = data_random
    with pytest.raises(NotFittedError):
        observables.transform(x)

    with pytest.raises(NotFittedError):
        observables.inverse(x)

    with pytest.raises(NotFittedError):
        observables.get_feature_names()

    observables.fit(x)
    check_is_fitted(observables)


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
    ],
)
def test_inverse(observables, data_random):
    x = data_random
    assert_allclose(observables.inverse(observables.fit_transform(x)), x)


def test_bad_polynomial_inputs():
    with pytest.raises(ValueError):
        Polynomial(degree=0)


def test_identity_feature_names(data_random):
    x = data_random
    model = Identity().fit(x)

    # Default names
    expected_names = [f"x{i}" for i in range(x.shape[1])]
    assert model.get_feature_names() == expected_names

    # Given names
    custome_names = [f"y{i+1}" for i in range(x.shape[1])]
    assert model.get_feature_names(input_features=custome_names) == custome_names


def test_polynomial_feature_names(data_small):
    x = data_small
    model = Polynomial(degree=2).fit(x)

    expected_default_names = ["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"]
    assert model.get_feature_names() == expected_default_names

    custom_names = ["x", "y"]
    expected_custom_names = ["1", "x", "y", "x^2", "x y", "y^2"]
    assert model.get_feature_names(input_features=custom_names) == expected_custom_names
