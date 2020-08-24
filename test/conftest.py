"""
Shared pytest fixtures for unit tests.

Put any datasets that are used by multiple unit test files here.
"""
import numpy as np
import pytest


@pytest.fixture
def data_random():
    x = np.random.randn(50, 10)
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
