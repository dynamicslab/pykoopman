"""Tests for (discrete) Koopman objects."""
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pykoopman import Koopman
from pykoopman.observables import Identity
from pykoopman.observables import Polynomial
from pykoopman.observables import TimeDelay


def test_fit(data_random):
    x = data_random
    model = Koopman().fit(x)
    check_is_fitted(model)


def test_predict_shape(data_random):
    x = data_random

    model = Koopman().fit(x)
    assert x.shape == model.predict(x).shape


def test_simulate_accuracy(data_2D_superposition):
    x = data_2D_superposition

    model = Koopman().fit(x)

    n_steps = 10
    x_pred = model.simulate(x[0], n_steps=n_steps)
    assert_allclose(x[1 : n_steps + 1], x_pred)


def test_koopman_matrix_shape(data_random):
    x = data_random
    model = Koopman().fit(x)
    assert model.koopman_matrix.shape[0] == model.n_output_features_


def test_if_fitted(data_random):
    x = data_random
    model = Koopman()

    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.simulate(x)
    with pytest.raises(NotFittedError):
        model.koopman_matrix
    with pytest.raises(NotFittedError):
        model._step(x)
    with pytest.raises(NotFittedError):
        model.score(x)


def test_score_without_target(data_2D_superposition):
    x = data_2D_superposition
    model = Koopman().fit(x)

    # Test without a target
    assert model.score(x) > 0.8


def test_score_with_target(data_2D_superposition):
    x = data_2D_superposition
    model = Koopman().fit(x)

    # Test with a target
    assert model.score(x[::2], y=x[1::2]) > 0.8


def test_score_complex_data(data_random_complex):
    x = data_random_complex
    model = Koopman().fit(x)

    with pytest.raises(ValueError):
        model.score(x, cast_as_real=False)


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_observables_integration(data_random, observables):
    x = data_random
    model = Koopman(observables=observables).fit(x)
    check_is_fitted(model)

    y = model.predict(x)
    assert y.shape[1] == x.shape[1]


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_observables_integration_accuracy(data_1D_cosine, observables):
    x = data_1D_cosine
    model = Koopman(observables=observables, quiet=True).fit(x)

    assert model.score(x) > 0.95


def test_simulate_with_time_delay(data_2D_superposition):
    x = data_2D_superposition

    observables = TimeDelay()
    model = Koopman(observables=observables)
    model.fit(x)

    n_steps = 10
    n_consumed_samples = observables.n_consumed_samples
    x_pred = model.simulate(x[: n_consumed_samples + 1], n_steps=n_steps)
    assert_allclose(
        x[n_consumed_samples + 1 : n_consumed_samples + n_steps + 1], x_pred
    )
