from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pykoopman.common import cqgle


@pytest.fixture
def cqgle_model():
    n = 64
    x = np.linspace(-10, 10, n, endpoint=False)
    dt = 0.01
    return cqgle(n, x, dt)


def test_cqgle_init(cqgle_model):
    assert cqgle_model.n_states == 64
    assert cqgle_model.dt == 0.01
    assert cqgle_model.k.shape == (64,)


def test_cqgle_sys(cqgle_model):
    t = 0
    x = np.zeros(cqgle_model.n_states)
    u = 0
    dx = cqgle_model.sys(t, x, u)
    assert dx.shape == (cqgle_model.n_states,)
    assert np.all(dx == 0)  # Should be zero for zero state


def test_cqgle_simulate(cqgle_model):
    x0 = np.exp(-(cqgle_model.x**2))
    n_int = 10
    n_sample = 2
    X, t = cqgle_model.simulate(x0, n_int, n_sample)

    # Check return shapes
    # X shape: (n_int // n_sample, n_states)
    expected_steps = n_int // n_sample
    assert X.shape == (expected_steps, cqgle_model.n_states)
    assert len(t) == expected_steps


def test_cqgle_collect_data_continuous(cqgle_model):
    n_traj = 3
    x0_single = np.exp(-(cqgle_model.x**2))
    x0 = np.vstack([x0_single] * n_traj)

    X, Y = cqgle_model.collect_data_continuous(x0)

    assert X.shape == (n_traj, cqgle_model.n_states)
    assert Y.shape == (n_traj, cqgle_model.n_states)


def test_cqgle_collect_one_step_data_discrete(cqgle_model):
    n_traj = 3
    x0_single = np.exp(-(cqgle_model.x**2))
    x0 = np.vstack([x0_single] * n_traj)

    X, Y = cqgle_model.collect_one_step_data_discrete(x0)

    assert X.shape == (n_traj, cqgle_model.n_states)
    assert Y.shape == (n_traj, cqgle_model.n_states)


def test_cqgle_collect_one_trajectory_data(cqgle_model):
    x0 = np.exp(-(cqgle_model.x**2))
    n_int = 10
    n_sample = 2
    y = cqgle_model.collect_one_trajectory_data(x0, n_int, n_sample)

    expected_steps = n_int // n_sample
    assert y.shape == (expected_steps, cqgle_model.n_states)


@patch("matplotlib.pyplot.show")
def test_cqgle_visualize_data(mock_show, cqgle_model):
    x0 = np.exp(-(cqgle_model.x**2))
    n_int = 10
    n_sample = 5
    X, t = cqgle_model.simulate(x0, n_int, n_sample)

    # Just ensure it runs without error
    cqgle_model.visualize_data(cqgle_model.x, t, X)
    mock_show.assert_called()


@patch("matplotlib.pyplot.show")
def test_cqgle_visualize_state_space(mock_show, cqgle_model):
    # Create some dummy data (needs enough potential components for SVD)
    X = np.random.rand(10, cqgle_model.n_states)

    # Just ensure it runs without error
    cqgle_model.visualize_state_space(X)
    mock_show.assert_called()
