"""Tests for pykoopman.observables objects."""
from __future__ import annotations

import pytest
from numpy import hstack
from numpy import iscomplexobj
from numpy import linspace
from numpy import stack
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pykoopman.observables import CustomObservables
from pykoopman.observables import Identity
from pykoopman.observables import Polynomial
from pykoopman.observables import RadialBasisFunction
from pykoopman.observables import RandomFourierFeatures
from pykoopman.observables import TimeDelay


@pytest.fixture
def data_small():
    t = linspace(0, 5, 10)
    return stack((t, t**2), axis=1)


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_if_fitted(observables, data_random):
    """
    we iterate over each observable object, first we
    test if it correctly raise NotFittedError when it is not fitted
    but called to .transform, .inverse, and .get_feature_names
    then we fit it, and check if it is fitted at the final step.
    """
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
    "observables_1",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_2",
    [
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
        Identity(),
        Polynomial(),
        TimeDelay(),
        RadialBasisFunction(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_if_fitted_two_obs(observables_1, observables_2, data_random):
    """
    we iterate over each observable object, first we
    test if it correctly raise NotFittedError when it is not fitted
    but called to .transform, .inverse, and .get_feature_names
    then we fit it, and check if it is fitted at the final step.
    """
    observables = observables_1 + observables_2
    test_if_fitted(observables, data_random)


@pytest.mark.parametrize(
    "observables_1",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_2",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_3",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_if_fitted_three_obs(observables_1, observables_2, observables_3, data_random):
    """
    we iterate over each observable object, first we
    test if it correctly raise NotFittedError when it is not fitted
    but called to .transform, .inverse, and .get_feature_names
    then we fit it, and check if it is fitted at the final step.
    """
    observables = observables_1 + observables_2 + observables_3
    test_if_fitted(observables, data_random)


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_inverse(observables, data_random):
    """
    we iterate over all obs to check if the fit_transform works,
    and if the output of fit_transform can be reverse back to x nicely
    with .inverse
    """
    x = data_random
    assert_allclose(observables.inverse(observables.fit_transform(x)), x)


@pytest.mark.parametrize(
    "observables_1",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_2",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
        RadialBasisFunction(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_inverse_two_obs(observables_1, observables_2, data_random):
    """
    we iterate over each observable object, first we
    test if it correctly raise NotFittedError when it is not fitted
    but called to .transform, .inverse, and .get_feature_names
    then we fit it, and check if it is fitted at the final step.
    """
    observables = observables_1 + observables_2
    test_inverse(observables, data_random)


@pytest.mark.parametrize(
    "observables_1",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_2",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        RadialBasisFunction(),
        Polynomial(degree=3, include_bias=False),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_3",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_inverse_three_obs(observables_1, observables_2, observables_3, data_random):
    """
    we iterate over each observable object, first we
    test if it correctly raise NotFittedError when it is not fitted
    but called to .transform, .inverse, and .get_feature_names
    then we fit it, and check if it is fitted at the final step.
    """
    observables = observables_1 + observables_2 + observables_3
    test_inverse(observables, data_random)


def test_time_delay_inverse(data_random):
    x = data_random
    delay = 2
    n_delays = 3
    n_deleted_rows = delay * n_delays

    observables = TimeDelay(delay=delay, n_delays=n_delays)
    y = observables.fit_transform(x)
    # First few rows of x are deleted which don't have enough
    # time history
    assert_array_equal(observables.inverse(y), x[n_deleted_rows:])


@pytest.mark.parametrize(
    "observables_1",
    [
        RadialBasisFunction(include_state=False),
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
        TimeDelay(delay=1, n_delays=2),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_2",
    [
        RadialBasisFunction(kernel_width=1.0, include_state=True),
        RadialBasisFunction(),
        TimeDelay(delay=3, n_delays=4),
        TimeDelay(delay=1, n_delays=6),
        RandomFourierFeatures(include_state=True, gamma=0.01, D=2),
    ],
)
def test_time_delay_inverse_two_obs(observables_1, observables_2, data_random):
    x = data_random
    observables = observables_1 + observables_2
    y = observables.fit_transform(x)
    n_deleted_rows = observables.n_consumed_samples
    # First few rows of x are deleted which don't have enough
    # time history
    assert_allclose(observables.inverse(y), x[n_deleted_rows:], rtol=1e-7)
    # assert_array_equal(observables.inverse(y), x[n_deleted_rows:])


@pytest.mark.parametrize(
    "observables_1",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=1),
        Polynomial(degree=4),
        RadialBasisFunction(),
        Polynomial(include_bias=False),
        Polynomial(degree=3, include_bias=False),
        TimeDelay(delay=1, n_delays=2),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "observables_2",
    [
        TimeDelay(delay=2, n_delays=3),
        TimeDelay(delay=1, n_delays=6),
        RadialBasisFunction(),
        RandomFourierFeatures(include_state=True, gamma=0.3, D=3),
    ],
)
@pytest.mark.parametrize(
    "observables_3",
    [
        TimeDelay(delay=2, n_delays=3),
        RadialBasisFunction(),
        TimeDelay(delay=1, n_delays=6) + TimeDelay(delay=3, n_delays=3),
        RandomFourierFeatures(include_state=False, gamma=0.01, D=2),
    ],
)
def test_time_delay_inverse_three_obs(
    observables_1, observables_2, observables_3, data_random
):
    x = data_random
    observables = observables_1 + observables_2 + observables_3
    y = observables.fit_transform(x)
    n_deleted_rows = observables.n_consumed_samples
    # First few rows of x are deleted which don't have enough
    # time history
    # assert_array_equal(observables.inverse(y), x[n_deleted_rows:])
    assert_allclose(observables.inverse(y), x[n_deleted_rows:], rtol=1e-7)


def test_bad_polynomial_inputs():
    with pytest.raises(ValueError):
        Polynomial(degree=0)


def test_bad_custom_observables_inputs():
    # One too few names
    observables = [lambda x: x, lambda x: x**2, lambda x: 0 * x, lambda x, y: x * y]
    observable_names = [lambda s: f"{s}^2", lambda: str(0), lambda s, t: f"{s} {t}"]

    with pytest.raises(ValueError):
        CustomObservables(observables, observable_names=observable_names)


def test_custom_observables_transform(data_small):
    x = data_small

    observables = [lambda x: x**2]
    y = CustomObservables(observables).fit_transform(x)

    # Identity is automatically prepended to custom observables
    assert_array_equal(y, hstack((x, x**2)))


@pytest.mark.parametrize("delay, n_delays", [(3, 2), (1, 5)])
def test_time_delay_output_shape(data_random, delay, n_delays):
    x = data_random
    y = TimeDelay(delay=delay, n_delays=n_delays).fit_transform(x)

    assert y.shape == (x.shape[0] - delay * n_delays, (n_delays + 1) * x.shape[1])


def test_time_delay_transform_matches_input(data_random):
    x = data_random

    observables = TimeDelay(delay=2, n_delays=4)
    observables.fit(x)

    y = observables.transform(x)
    assert_array_equal(y[1], x[[9, 7, 5, 3, 1]].flatten())


@pytest.mark.parametrize(
    "observables, expected_default_names, expected_custom_names",
    [
        (Identity(), ["x0", "x1"], ["x", "y"]),
        (
            Polynomial(degree=2),
            ["1", "x0", "x1", "x0^2", "x0 x1", "x1^2"],
            ["1", "x", "y", "x^2", "x y", "y^2"],
        ),
        (
            TimeDelay(delay=2, n_delays=2),
            ["x0(t)", "x1(t)", "x0(t-2dt)", "x1(t-2dt)", "x0(t-4dt)", "x1(t-4dt)"],
            ["x(t)", "y(t)", "x(t-2dt)", "y(t-2dt)", "x(t-4dt)", "y(t-4dt)"],
        ),
        (
            pytest.lazy_fixture("data_custom_observables"),
            ["x0", "x1", "x0", "x1", "x0^2", "x1^2", "0", "0", "x0 x1"],
            ["x", "y", "x", "y", "x^2", "y^2", "0", "0", "x y"],
        ),
    ],
)
def test_feature_names(
    observables, expected_default_names, expected_custom_names, data_small
):
    x = data_small

    observables.fit(x)
    assert observables.get_feature_names() == expected_default_names

    custom_names = ["x", "y"]
    assert (
        observables.get_feature_names(input_features=custom_names)
        == expected_custom_names
    )


# so far it does not support complex number for random fourier features.
# shaowu does not think complex.number is necessary at all
@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_complex_data(data_random_complex, observables):
    x = data_random_complex
    y = observables.fit_transform(x)

    assert iscomplexobj(y)
