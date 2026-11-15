"""Regression tests for RBF observables that retain the input state."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pykoopman.observables import Identity
from pykoopman.observables import RadialBasisFunction


@pytest.mark.parametrize(
    "rbf_type", ["thinplate", "gauss", "invquad", "invmultquad", "polyharmonic"]
)
@pytest.mark.parametrize("concatenate", [False, True])
def test_rbf_inverse_preserves_unseen_states(rbf_type, concatenate):
    # The training trajectory has correlated coordinates. Its least-squares
    # inverse need not recover states outside that trajectory.
    train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    unseen = np.array([[0.0, 1.0], [2.0, -1.0]])
    obs = RadialBasisFunction(
        rbf_type=rbf_type,
        n_centers=2,
        centers=np.array([[0.5, 1.5], [-0.5, 0.5]]),
        include_state=True,
    )
    if concatenate:
        obs = obs + Identity()

    obs.fit(train)
    assert_array_equal(obs.inverse(obs.transform(unseen)), unseen)


def test_rbf_inverse_preserves_states_after_refit():
    obs = RadialBasisFunction(n_centers=2, centers=np.array([[0.0, 1.0], [1.0, 0.0]]))
    train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    unseen = np.array([[0.0, 1.0], [2.0, -1.0]])

    # Switching include_state must rebuild the measurement matrix with the
    # correct dimensions rather than retain a learned or earlier projection.
    obs.set_params(include_state=False).fit(train)
    obs.set_params(include_state=True).fit(train)
    assert_array_equal(obs.inverse(obs.transform(unseen)), unseen)


def test_rbf_without_state_keeps_least_squares_inverse():
    train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    obs = RadialBasisFunction(
        n_centers=2,
        centers=np.array([[0.0, 1.0], [1.0, 0.0]]),
        include_state=False,
    )
    lifted = obs.fit_transform(train)
    expected = np.linalg.lstsq(lifted, train, rcond=-1)[0].T

    assert_allclose(obs.measurement_matrix_, expected)
    assert_allclose(obs.inverse(lifted), lifted @ expected.T)
