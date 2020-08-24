# TODO: add unit test checking that model.inverse(model.transform(x)) == x
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pykoopman.observables import Identity
from pykoopman.observables import Polynomial


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
    user_defined_names = [f"y{i+1}" for i in range(x.shape[1])]
    assert (
        model.get_feature_names(input_features=user_defined_names) == user_defined_names
    )
