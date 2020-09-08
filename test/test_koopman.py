"""Tests for (discrete) Koopman objects."""
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pykoopman import Koopman


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


def test_score_complex_data(data_2D_superposition):
    x = data_2D_superposition
    model = Koopman().fit(x)

    with pytest.raises(ValueError):
        model.score(x, cast_as_real=False)
