"""
Shared pytest fixtures for unit tests.

Put any datasets that are used by multiple unit test files here.
"""
from __future__ import annotations

import os.path

import numpy as np
import pytest
import scipy

from pykoopman.common import advance_linear_system
from pykoopman.common import drss
from pykoopman.common import lorenz
from pykoopman.common import rev_dvdp
from pykoopman.common import torus_dynamics
from pykoopman.observables import CustomObservables

my_path = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture
def data_random():
    x = np.random.randn(50, 10)
    return x


@pytest.fixture
def data_random_complex():
    x = np.random.randn(50, 10) + 1j * np.random.randn(50, 10)
    return x


@pytest.fixture
def data_2D_superposition():
    t = np.linspace(0, 2 * np.pi, 200)
    x = np.linspace(-5, 5, 100)
    [x_grid, t_grid] = np.meshgrid(x, t)

    def sech(x):
        return 1 / np.cosh(x)

    f1 = sech(x_grid + 3) * np.exp(1j * 2.3 * t_grid)
    f2 = 2 * (sech(x_grid) * np.tanh(x_grid)) * np.exp(1j * 2.8 * t_grid)
    return f1 + f2


@pytest.fixture
def data_2D_linear_real_system():
    A = np.array([[1.5, 0], [0, 0.1]])
    B = np.array([[1], [0]])
    x0 = np.array([4, 7])
    u = 0 * np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 3, 5])
    n = len(u) + 1
    x = np.zeros([n, len(x0)])
    x[0, :] = x0
    for i in range(n - 1):
        x[i + 1, :] = A.dot(x[i, :]) + B.dot(u[np.newaxis, i])
    X = x
    return X


@pytest.fixture
def data_1D_cosine():
    t = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(3 * t)
    return x


@pytest.fixture
def data_custom_observables():
    observables = [lambda x: x, lambda x: x**2, lambda x: 0 * x, lambda x, y: x * y]
    observable_names = [
        lambda s: str(s),
        lambda s: f"{s}^2",
        lambda s: str(0),
        lambda s, t: f"{s} {t}",
    ]

    return CustomObservables(observables, observable_names=observable_names)


@pytest.fixture
def data_realistic_custom_observables():
    observables = [lambda x: x**2, lambda x, y: x * y]
    observable_names = [
        lambda s: f"{s}^2",
        lambda s, t: f"{s} {t}",
    ]

    return CustomObservables(observables, observable_names=observable_names)


@pytest.fixture
def data_2D_linear_control_system():
    A = np.array([[1.5, 0], [0, 0.1]])
    B = np.array([[1], [0]])
    x0 = np.array([4, 7])
    u = np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 3, 5])
    n = len(u) + 1
    x = np.zeros([n, len(x0)])
    x[0, :] = x0
    for i in range(n - 1):
        x[i + 1, :] = A.dot(x[i, :]) + B.dot(u[np.newaxis, i])
    X = x
    C = u[:, np.newaxis]

    return X, C, A, B


@pytest.fixture
def data_drss():
    # Seed random generator for reproducibility
    np.random.seed(0)

    n_states = 5
    n_controls = 2
    n_measurements = 50
    A, B, C = drss(n_states, n_controls, n_measurements)

    x0 = np.array([4, 7, 2, 8, 0])
    u = np.array(
        [
            [
                -4,
                -2,
                -1,
                -0.5,
                0,
                0.5,
                1,
                3,
                5,
                9,
                8,
                4,
                3.5,
                1,
                2,
                3,
                1.5,
                0.5,
                0,
                1,
                -1,
                -0.5,
                -2,
                -4,
                -5,
                -7,
                -9,
                -6,
                -5,
                -5.5,
            ],
            [
                4,
                1,
                -1,
                -0.5,
                0,
                1,
                2,
                4,
                3,
                1.5,
                1,
                0,
                -1,
                -1.5,
                -2,
                -1,
                -3,
                -5,
                -9,
                -7,
                -5,
                -6,
                -8,
                -6,
                -4,
                -3,
                -2,
                -0.5,
                0.5,
                3,
            ],
        ]
    )
    n = u.shape[1]
    X, Y = advance_linear_system(x0, u, n, A, B, C)
    U = u.T

    return Y, U, A, B, C


@pytest.fixture
def data_torus_unforced():
    T = 20  # integration time
    dt = 0.05  # time step
    n_samples = int(T / dt)

    np.random.seed(1)  # Seed random generator for reproducibility
    torus = torus_dynamics()
    torus.advance(n_samples, dt)
    xhat_nonzero = torus.Xhat[torus.mask.reshape(torus.n_states**2) == 1, :]

    return xhat_nonzero, torus.frequencies, dt


@pytest.fixture
def data_torus_ct():
    T = 4  # integration time
    dt = 0.01  # time step
    n_samples = int(T / dt)

    np.random.seed(1)  # for reproducibility
    torus = torus_dynamics()
    torus.advance(n_samples, dt)
    xhat = torus.Xhat[torus.mask.reshape(torus.n_states**2) == 1, :]

    return xhat


@pytest.fixture
def data_torus_dt():
    T = 4  # integration time
    dt = 0.01  # time step
    n_samples = int(T / dt)

    np.random.seed(1)  # for reproducibility
    torus = torus_dynamics()
    torus.advance_discrete_time(n_samples, dt)
    xhat = torus.Xhat[torus.mask.reshape(torus.n_states**2) == 1, :]

    return xhat


@pytest.fixture
def data_lorenz():
    x0 = [-8, 8, 27]  # initial condition
    dt = 0.001
    t = np.linspace(dt, 200, 200000)
    x = scipy.integrate.odeint(lorenz, x0, t, atol=1e-12, rtol=1e-12)

    return t, x, dt


@pytest.fixture
def data_rev_dvdp():
    np.random.seed(42)  # for reproducibility
    n_states = 2  # Number of states
    dT = 0.1  # Timestep
    n_traj = 51  # Number of trajectories
    n_int = 1  # Integration length

    # Uniform distribution of initial conditions
    x = X0 = 2 * np.random.random([n_states, n_traj]) - 1

    # training data
    Xtrain = np.zeros((n_states, n_int * n_traj))
    Ytrain = np.zeros((n_states, n_int * n_traj))
    for step in range(n_int):
        y = rev_dvdp(0, x, 0, dT)
        Xtrain[:, (step) * n_traj : (step + 1) * n_traj] = x
        Ytrain[:, (step) * n_traj : (step + 1) * n_traj] = y
        x = y

    x0 = np.array([-0.3, -0.2])
    t = np.arange(0, 10, dT)

    # test data
    Xtest = np.zeros((len(t), n_states))
    Xtest[0, :] = x0
    for step in range(len(t) - 1):
        y = rev_dvdp(0, Xtest[step, :][:, np.newaxis], 0, dT)
        Xtest[step + 1, :] = y.ravel()

    return dT, X0, Xtrain, Ytrain, Xtest


@pytest.fixture
def data_for_validty_check():
    A = np.array([[-0.9, -0.3], [0.2, -0.7]])
    B = np.zeros((2, 1))
    C = np.eye(2)
    x0 = np.array([-2, 2])
    N = 50
    X, Y = advance_linear_system(x0, np.zeros((1, N)), N, A, B, C)
    return X, np.arange(N)


@pytest.fixture
def data_vdp_edmdc():
    # path = os.path.join(my_path, "../data/test.csv")
    xpred = np.loadtxt(os.path.join(my_path, "data_vdp_for_edmdc.txt"), delimiter=",")
    # xpred = np.loadtxt("./test/data_vdp_for_edmdc.txt", delimiter=",")
    return xpred
