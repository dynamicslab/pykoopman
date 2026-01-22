from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pykoopman.common import vbe


@pytest.fixture
def vbe_model():
    n = 64
    x = np.linspace(-10, 10, n, endpoint=False)
    dt = 0.01
    return vbe(n, x, dt)


def test_vbe_init(vbe_model):
    assert vbe_model.n_states == 64
    assert vbe_model.dt == 0.01


def test_vbe_sys(vbe_model):
    t = 0
    x = np.zeros(vbe_model.n_states)
    u = 0
    dx = vbe_model.sys(t, x, u)
    assert dx.shape == (vbe_model.n_states,)
    assert np.all(dx == 0)


def test_vbe_simulate(vbe_model):
    x0 = np.exp(-(vbe_model.x**2))
    n_int = 10
    n_sample = 2
    X, t = vbe_model.simulate(x0, n_int, n_sample)

    expected_steps = n_int // n_sample
    assert X.shape == (expected_steps, vbe_model.n_states)
    assert len(t) == expected_steps


def test_vbe_collect_data_continuous(vbe_model):
    n_traj = 3
    x0_single = np.exp(-(vbe_model.x**2))
    x0 = np.vstack([x0_single] * n_traj)

    X, Y = vbe_model.collect_data_continuous(x0)

    assert X.shape == (n_traj, vbe_model.n_states)
    assert Y.shape == (n_traj, vbe_model.n_states)


def test_vbe_collect_one_step_data_discrete(vbe_model):
    n_traj = 3
    x0_single = np.exp(-(vbe_model.x**2))
    x0 = np.vstack([x0_single] * n_traj)

    X, Y = vbe_model.collect_one_step_data_discrete(x0)

    assert X.shape == (n_traj, vbe_model.n_states)
    assert Y.shape == (n_traj, vbe_model.n_states)


def test_vbe_collect_one_trajectory_data(vbe_model):
    x0 = np.exp(-(vbe_model.x**2))
    n_int = 10
    n_sample = 2
    y = vbe_model.collect_one_trajectory_data(x0, n_int, n_sample)

    expected_steps = n_int // n_sample
    assert y.shape == (expected_steps, vbe_model.n_states)


@patch("matplotlib.pyplot.show")
def test_vbe_visualize_data(mock_show, vbe_model):
    x0 = np.exp(-(vbe_model.x**2))
    n_int = 10
    n_sample = 5
    X, t = vbe_model.simulate(x0, n_int, n_sample)

    vbe_model.visualize_data(vbe_model.x, t, X)
    mock_show.assert_called()


@patch("matplotlib.pyplot.show")
def test_vbe_visualize_state_space(mock_show, vbe_model):
    X = np.random.rand(10, vbe_model.n_states)
    vbe_model.visualize_state_space(X)
    mock_show.assert_called()
