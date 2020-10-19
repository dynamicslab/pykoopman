"""
Shared pytest fixtures for unit tests.

Put any datasets that are used by multiple unit test files here.
"""
import numpy as np
import pytest

from pykoopman.observables import CustomObservables


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
def data_1D_cosine():
    t = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(3 * t)
    return x


@pytest.fixture
def data_custom_observables():
    observables = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x, lambda x, y: x * y]
    observable_names = [
        lambda s: str(s),
        lambda s: f"{s}^2",
        lambda s: str(0),
        lambda s, t: f"{s} {t}",
    ]

    return CustomObservables(observables, observable_names=observable_names)


@pytest.fixture
def data_realistic_custom_observables():
    observables = [lambda x: x ** 2, lambda x, y: x * y]
    observable_names = [
        lambda s: f"{s}^2",
        lambda s, t: f"{s} {t}",
    ]

    return CustomObservables(observables, observable_names=observable_names)
