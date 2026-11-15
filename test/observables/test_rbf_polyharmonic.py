"""Tests for the polyharmonic RadialBasisFunction at zero distance from a center."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pykoopman.observables import RadialBasisFunction

# Training states and centers from the example in issue 78. centers has shape
# (n_input_features, n_centers), so center 0 is (0, 1) and center 1 is (1, 0).
X_TRAIN = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
CENTERS = np.array([[0.0, 1.0], [1.0, 0.0]])


@pytest.mark.parametrize("k", [0.5, 1.0, 3.0])
def test_polyharmonic_is_zero_on_a_center(k):
    """phi(r) = r**k * log(r) tends to 0 as r -> 0 whenever k > 0."""
    rbf = RadialBasisFunction(
        rbf_type="polyharmonic",
        n_centers=2,
        centers=CENTERS.copy(),
        polyharmonic_coeff=k,
        include_state=True,
    ).fit(X_TRAIN)

    # [0, 1] lies on center 0 and is sqrt(2) away from center 1.
    # [2, -1] is sqrt(8) away from center 0 and sqrt(2) away from center 1.
    y = rbf.transform(np.array([[0.0, 1.0], [2.0, -1.0]]))

    phi_sqrt2 = 2 ** (k / 2) * np.log(2) / 2
    phi_sqrt8 = 8 ** (k / 2) * 3 * np.log(2) / 2
    expected = np.array(
        [
            [0.0, 1.0, 0.0, phi_sqrt2],
            [2.0, -1.0, phi_sqrt8, phi_sqrt2],
        ]
    )
    assert_allclose(y, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("include_state", [True, False])
def test_polyharmonic_fit_with_default_centers_on_1d_data(include_state):
    """With centers=None, fit rescales the random centers to [min(x), max(x)].

    On 1D data the smallest random center is mapped exactly onto the smallest
    sample, so the training lift always contains r = 0 entries.
    """
    np.random.seed(0)
    x = np.linspace(-1.0, 1.0, 11).reshape(-1, 1)
    rbf = RadialBasisFunction(
        rbf_type="polyharmonic", n_centers=5, include_state=include_state
    )

    rbf.fit(x)
    y = rbf.transform(x)

    on_center = x == rbf.centers  # (n_samples, n_centers) by broadcasting
    assert on_center.any()
    offset = 1 if include_state else 0
    assert np.all(np.isfinite(y))
    assert np.all(y[:, offset:][on_center] == 0)


def test_polyharmonic_coeff_zero_keeps_log_singularity():
    """polyharmonic_coeff=0 gives phi(r) = log(r), with no finite value at r = 0.

    The lift keeps -inf there rather than the limit 0 that applies for k > 0.
    """
    rbf = RadialBasisFunction(
        rbf_type="polyharmonic",
        n_centers=2,
        centers=CENTERS.copy(),
        polyharmonic_coeff=0.0,
        include_state=False,
    ).fit(X_TRAIN)

    with np.errstate(divide="ignore"):
        y = rbf.transform(np.array([[0.0, 1.0]]))

    assert y[0, 0] == -np.inf
    assert_allclose(y[0, 1], np.log(2) / 2, rtol=1e-12)
