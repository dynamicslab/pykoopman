from __future__ import annotations

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from pykoopman.common.examples import advance_linear_system
from pykoopman.common.examples import drss
from pykoopman.common.examples import forced_duffing
from pykoopman.common.examples import Linear2Ddynamics
from pykoopman.common.examples import lorenz
from pykoopman.common.examples import rev_dvdp
from pykoopman.common.examples import rk4
from pykoopman.common.examples import sine_wave
from pykoopman.common.examples import slow_manifold
from pykoopman.common.examples import square_wave
from pykoopman.common.examples import vdp_osc


def test_drss_shapes():
    n_states = 3
    n_controls = 2
    n_measurements = 4
    A, B, C = drss(n=n_states, p=n_controls, m=n_measurements)
    assert A.shape == (n_states, n_states)
    assert B.shape == (n_states, n_controls)
    assert C.shape == (n_measurements, n_states)


def test_drss_identity_measurement():
    # If m=0, C should be identity
    n_states = 3
    A, B, C = drss(n=n_states, m=0)
    assert C.shape == (n_states, n_states)
    np.testing.assert_array_equal(C, np.eye(n_states))


def test_advance_linear_system():
    n = 2
    A = np.eye(n)
    B = np.eye(n)
    C = np.eye(n)
    x0 = np.array([1.0, 1.0])
    # consistent for 1 step
    # n_steps to simulate
    n_steps = 2
    # Expanding u to match steps if needed, but the function handles 1D u as row vector
    # Let's provide u of shape (p, n_steps-1)
    u_seq = np.ones((n, n_steps - 1))

    x, y = advance_linear_system(x0, u_seq, n_steps, A, B, C)

    # x shape should be (n, n_steps)?? Wait, let's check docstring or implementation.
    # Implementation: x = np.zeros([n, len(x0)]) -> Wait, len(x0) is n.
    # The implementation:
    # x = np.zeros([n, len(x0)]) ??? No, x0 is (n,). len(x0) is n.
    # But usually x should be (n_states, n_time_steps).
    # docstring says: returns x of shape (n, len(x0)).
    # This seems like a potential bug or confusion in docstring vs code
    # if n_steps != n_states.
    # Let's look at code: 'x = np.zeros([n, len(x0)])'
    # where n is passed as arg 'n' (steps).
    # The argument name 'n' shadows dimension 'n'.
    # In function def: advance_linear_system(x0, u, n, ...):
    # n is "Number of steps to simulate"
    # But inside: x = np.zeros([n, len(x0)])
    # So dim 0 is n (steps), dim 1 is len(x0) (states?).
    # Usually states are rows or columns.
    # If n=steps, then x is (steps, states) or (states, steps).
    # Code: x[0, :] = x0. So x is (steps, states).

    # x has shape (n_steps, n_states)
    assert x.shape == (n_steps, len(x0))
    assert y.shape == (n_steps, C.shape[0])


def test_vdp_osc_rk4():
    t = 0
    x = np.array([[1.0], [0.5]])
    u = 0.0
    dt = 0.01

    # Check vdp_osc structure
    dx = vdp_osc(t, x, u)
    assert dx.shape == x.shape

    # Check rk4 integration step
    x_next = rk4(t, x, u, dt, vdp_osc)
    assert x_next.shape == x.shape
    assert not np.array_equal(x, x_next)


def test_square_and_sine_wave():
    # Just smoke tests to ensure they run/return floats
    val_sq = square_wave(10)
    assert (
        isinstance(val_sq, float)
        or isinstance(val_sq, int)
        or isinstance(val_sq, np.float64)
    )

    val_sin = sine_wave(10)
    assert isinstance(val_sin, float) or isinstance(val_sin, np.floating)


def test_lorenz():
    x = [10.0, 10.0, 10.0]
    t = 0.0
    dx = lorenz(x, t)
    assert len(dx) == 3


def test_rev_dvdp():
    x = np.array(
        [[1.0], [0.5]]
    )  # needs to be 2D array (2, 1) based on code usage of x[0,:]?
    # Code: x[0,:] - ...
    # So if we pass (2, 1), x[0,:] is shape (1,).
    t = 0
    x_next = rev_dvdp(t, x)
    assert x_next.shape == x.shape


def test_linear_2d_dynamics():
    sys = Linear2Ddynamics()
    x = np.array([[1.0], [1.0]])

    # Test linear_map
    y = sys.linear_map(x)
    assert y.shape == x.shape

    # Test collect_data
    n_int = 10
    n_traj = 1
    X, Y = sys.collect_data(x, n_int, n_traj)
    # shapes: (n_states, n_int * n_traj)
    assert X.shape == (2, n_int * n_traj)
    assert Y.shape == (2, n_int * n_traj)


def test_slow_manifold():
    model = slow_manifold()
    x = np.array([[0.1], [0.1]])  # (2, 1) to match usage of x[0, :]

    # Test sys
    t = 0
    u = 0
    dx = model.sys(t, x, u)
    assert dx.shape == x.shape

    # Test simulate (requires x0 to be (2, 1))
    x0 = np.array([[0.1], [0.1]])
    n_int = 100
    X = model.simulate(x0, n_int)
    assert X.shape == (2, n_int * 1)  # n_traj is 1


def test_forced_duffing():
    dt = 0.01
    d = 0.1
    alpha = 1.0
    beta = 1.0
    model = forced_duffing(dt, d, alpha, beta)

    assert model.n_states == 2

    # Test sys
    t = 0
    x = np.array([[1.0], [1.0]])
    u = 0.0
    dx = model.sys(t, x, u)
    assert dx.shape == x.shape

    # Test simulate
    x0 = np.array([[1.0], [1.0]])
    n_int = 10
    u_seq = np.zeros((n_int, 1))  # (n_int, n_traj) ?
    # collect_data_discrete uses u[step, :] which implies u is (n_int, n_traj) ??
    # Let's check simulate implementation: u is passed as (n_int, ...?)
    # simulate(x0, n_int, u) -> u[step, :]
    # if x0 has n_traj=1.

    # Wait, in forced_duffing.simulate:
    # u[step, :] is passed to rk4.
    # if u is (n_int, 1), u[step,:] is shape (1,).
    # sys takes u.
    # sys implementation: ... + u
    # if u is scalar or (1,) it broadcasts.

    X = model.simulate(x0, n_int, u_seq)
    assert X.shape == (2, n_int * 1)

    # Test collect_data_continuous
    u_static = 0.0
    X_c, Y_c = model.collect_data_continuous(x0, u_static)
    assert X_c.shape == x0.shape
    assert Y_c.shape == x0.shape

    # Test collect_data_discrete
    X_d, Y_d = model.collect_data_discrete(x0, n_int, u_seq)
    assert X_d.shape == (2, n_int * 1)
    assert Y_d.shape == (2, n_int * 1)


@patch("matplotlib.pyplot.show")
def test_forced_duffing_visualize(mock_show):
    dt = 0.01
    model = forced_duffing(dt, 0.1, 1.0, 1.0)
    t = np.linspace(0, 1, 100)
    X = np.random.rand(2, 100)

    model.visualize_trajectories(t, X, n_traj=1)
    mock_show.assert_not_called()
    # visualize_trajectories doesn't call show() in source??
    # Let's check source.
    # visualize_trajectories: plt.subplots... axs.plot... axs.set... No plt.show()
    # It just makes plots.
    plt.close()  # Close to avoid warning

    model.visualize_state_space(X, X, n_traj=1)
    # visualize_state_space: plt.subplots... axs.plot... No plt.show()
    plt.close()
