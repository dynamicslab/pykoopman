"""Tests for (discrete) Koopman objects."""
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import numpy as np

from pykoopman import Koopman
from pykoopman import regression
from pykoopman.common import drss

def test_fit(data_random):
    x = data_random
    model = Koopman()
    model.fit(x)
    check_is_fitted(model)


def test_predict_shape(data_random):
    x = data_random

    model = Koopman()
    model.fit(x)
    assert x.shape == model.predict(x).shape


def test_simulate_accuracy(data_2D_superposition):
    x = data_2D_superposition

    model = Koopman()
    model.fit(x)

    n_steps = 10
    x_pred = model.simulate(x[0], n_steps=n_steps)
    assert_allclose(x[1 : n_steps + 1], x_pred)


def test_koopman_matrix_shape(data_random):
    x = data_random
    model = Koopman()
    model.fit(x)
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

def test_if_dmdc_model_is_accurate_with_known_controlmatrix(data_2D_linear_control_system):
    X, C, A, B = data_2D_linear_control_system
    model = Koopman()

    DMDc = regression.DMDc(svd_rank=3, control_matrix=B)
    model = Koopman(regressor=DMDc)
    model.fit(X, C)
    Aest = model.state_transition_matrix
    assert_allclose(Aest, A, 1e-07, 1e-12)

def test_if_dmdc_model_is_accurate_with_unknown_controlmatrix(data_2D_linear_control_system):
    X, C, A, B = data_2D_linear_control_system
    model = Koopman()

    DMDc = regression.DMDc(svd_rank=3)
    model = Koopman(regressor=DMDc)
    model.fit(X, C)
    Aest = model.state_transition_matrix
    Best = model.control_matrix
    assert_allclose(Aest, A, 1e-07, 1e-12)
    assert_allclose(Best, B, 1e-07, 1e-12)

def test_simulate_accuracy_dmdc(data_2D_linear_control_system):
    X, C, _, _ = data_2D_linear_control_system

    DMDc = regression.DMDc(svd_rank=3)
    model = Koopman(regressor=DMDc)
    model.fit(X, C)

    n_steps = len(C)
    x_pred = model.simulate(X[0,:], C, n_steps=n_steps-1)
    assert_allclose(X[1 : n_steps,:], x_pred, 1e-07, 1e-12)

def test_misc_drss_measurement_matrix():
    A,B,C = drss(2,2,0)
    assert_allclose(C,np.identity(2))
