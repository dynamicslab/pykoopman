"""Tests for pykoopman.koopman_continuous methods."""
from __future__ import annotations

import pytest
from numpy.testing import assert_allclose
from pydmd import DMD
from sklearn.utils.validation import check_is_fitted

from pykoopman import KoopmanContinuous
from pykoopman import observables
from pykoopman import regression
from pykoopman.differentiation import Derivative


@pytest.mark.parametrize(
    "data",
    [pytest.lazy_fixture("data_random"), pytest.lazy_fixture("data_random_complex")],
)
def test_derivative_integration(data):
    x = data

    diff = Derivative(kind="finite_difference", k=1)
    dmd = DMD(svd_rank=2)
    model = KoopmanContinuous(differentiator=diff, regressor=dmd)

    model.fit(x)
    check_is_fitted(model)


def test_havok_prediction(data_lorenz):
    t, x, dt = data_lorenz

    n_delays = 99
    TDC = observables.TimeDelay(delay=1, n_delays=n_delays)
    HAVOK = regression.HAVOK(svd_rank=15)
    Diff = Derivative(kind="finite_difference", k=2)
    model = KoopmanContinuous(observables=TDC, differentiator=Diff, regressor=HAVOK)
    model.fit(x[:, 0], dt=dt)

    known_external_input = model.regressor.forcing_signal

    # one step prediction
    xpred_one_step = model.predict(x[: n_delays + 1, 0], dt, u=known_external_input[0])
    assert_allclose(x[n_delays + 1, 0], xpred_one_step, atol=1e-3)

    # simulate: mult steps prediction
    xpred = model.simulate(
        x=x[: n_delays + 1, 0], t=t[n_delays:] - t[n_delays], u=known_external_input
    )

    assert_allclose(xpred[50], 3.54512034, atol=1e-3)
