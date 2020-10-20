"""Tests for differentiation methods."""
import numpy as np
import pytest
from derivative import dxdt

from pykoopman.differentiation import Derivative


@pytest.fixture
def data_1D_quadratic():
    t = np.linspace(0, 5, 100)
    x = t.reshape(-1, 1) ** 2
    x_dot = 2 * t.reshape(-1, 1)

    return x, t, x_dot


@pytest.fixture
def data_1D_bad_shape():
    t = np.linspace(0, 5, 100)
    x = t ** 2
    x_dot = 2 * t

    return x, t, x_dot


@pytest.fixture
def data_2D_quadratic():
    t = np.linspace(0, 5, 100)
    x = np.zeros((len(t), 2))
    x[:, 0] = t ** 2
    x[:, 1] = -(t ** 2)

    x_dot = np.zeros_like(x)
    x_dot[:, 0] = 2 * t
    x_dot[:, 1] = -2 * t

    return x, t, x_dot


@pytest.fixture(params=["data_1D_quadratic", "data_1D_bad_shape", "data_2D_quadratic"])
def data(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "kws",
    [
        dict(kind="spectral"),
        dict(kind="spline", s=1e-2),
        dict(kind="trend_filtered", order=0, alpha=1e-2),
        dict(kind="finite_difference", k=1),
        dict(kind="savitzky_golay", order=3, left=1, right=1),
    ],
)
def test_derivative_package_equivalence(data, kws):
    x, t, _ = data

    x_dot_pykoopman = Derivative(**kws)(x, t)
    x_dot_derivative = dxdt(x, t, axis=0, **kws).reshape(x_dot_pykoopman.shape)

    np.testing.assert_array_equal(x_dot_pykoopman, x_dot_derivative)


def test_bad_t_values(data_1D_quadratic):
    x, t, _ = data_1D_quadratic

    method = Derivative(kind="finite_difference", k=1)

    with pytest.raises(ValueError):
        method(x, t=-1)

    with pytest.raises(ValueError):
        method(x, t[:5])

    with pytest.raises(ValueError):
        inds = np.arange(len(t))
        # Swap two time entries
        inds[[0, 1]] = inds[[1, 0]]
        method(x, t[inds])


def test_accuracy(data):
    x, t, x_dot = data

    method = Derivative(kind="finite_difference", k=1)
    x_dot_method = method(x, t)

    if x_dot.ndim == 1:
        x_dot = x_dot.reshape(-1, 1)

    # Ignore endpoints
    np.testing.assert_allclose(x_dot[1:-1], x_dot_method[1:-1])
