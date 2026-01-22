from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pykoopman.common import ks


@pytest.fixture
def ks_model():
    n = 64
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    dt = 0.01
    nu = 0.1
    return ks(n, x, nu, dt)


def test_ks_init(ks_model):
    assert ks_model.n_states == 64
    assert ks_model.dt == 0.01
    assert ks_model.E.shape == (64,)


def test_ks_sys_not_implemented(ks_model):
    with pytest.raises(NotImplementedError):
        ks_model.sys(0, np.zeros(64), 0)


def test_ks_simulate(ks_model):
    x0 = np.sin(ks_model.x)
    n_int = 10
    n_sample = 2
    X, t = ks_model.simulate(x0, n_int, n_sample)

    expected_steps = n_int // n_sample
    assert X.shape == (expected_steps, ks_model.n_states)
    assert len(t) == expected_steps


def test_ks_collect_data_continuous_not_implemented(ks_model):
    with pytest.raises(NotImplementedError):
        ks_model.collect_data_continuous(np.zeros((1, 64)))


def test_ks_collect_one_step_data_discrete(ks_model):
    n_traj = 3
    x0_single = np.sin(ks_model.x)
    x0 = np.vstack([x0_single] * n_traj)

    X, Y = ks_model.collect_one_step_data_discrete(x0)

    assert X.shape == (n_traj, ks_model.n_states)
    assert Y.shape == (n_traj, ks_model.n_states)


def test_ks_collect_one_trajectory_data(ks_model):
    x0 = np.sin(ks_model.x)
    n_int = 10
    n_sample = 2
    y = ks_model.collect_one_trajectory_data(x0, n_int, n_sample)

    expected_steps = n_int // n_sample
    assert y.shape == (expected_steps, ks_model.n_states)


@patch("matplotlib.pyplot.show")
def test_ks_visualize_data(mock_show, ks_model):
    x0 = np.sin(ks_model.x)
    n_int = 10
    n_sample = 5
    X, t = ks_model.simulate(x0, n_int, n_sample)

    ks_model.visualize_data(ks_model.x, t, X)
    mock_show.assert_called()


@patch("matplotlib.pyplot.show")
def test_ks_visualize_state_space(mock_show, ks_model):
    X = np.random.rand(10, ks_model.n_states)
    ks_model.visualize_state_space(X)
    mock_show.assert_called()
