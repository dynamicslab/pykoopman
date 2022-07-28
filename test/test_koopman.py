"""Tests for (discrete) Koopman objects."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydmd import DMD
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pykoopman import Koopman
from pykoopman import regression
from pykoopman.common import drss
from pykoopman.observables import Identity
from pykoopman.observables import Polynomial
from pykoopman.observables import TimeDelay


def test_fit(data_random):
    x = data_random
    model = Koopman().fit(x)
    check_is_fitted(model)


def test_predict_shape(data_random):
    x = data_random
    dmd = DMD(svd_rank=10)
    model = Koopman(regressor=dmd).fit(x)
    assert x.shape == model.predict(x).shape


def test_simulate_accuracy(data_2D_superposition):
    x = data_2D_superposition

    regressor = DMD(svd_rank=10)
    model = Koopman(regressor=regressor).fit(x)

    n_steps = 10
    x_pred = model.simulate(x[0], n_steps=n_steps)
    assert_allclose(x[1 : n_steps + 1], x_pred)


def test_koopman_matrix_shape(data_random):
    x = data_random
    dmd = DMD(svd_rank=10)
    model = Koopman(regressor=dmd).fit(x)
    assert model.koopman_matrix.shape[0] == model.n_output_features_


def test_if_fitted(data_random):
    x = data_random
    model = Koopman()

    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.simulate(x)
    with pytest.raises(NotFittedError):
        model.koopman_matrix
    with pytest.raises(NotFittedError):
        model._step(x)
    with pytest.raises(NotFittedError):
        model.score(x)


def test_score_without_target(data_2D_superposition):
    x = data_2D_superposition
    dmd = DMD(svd_rank=10)
    model = Koopman(regressor=dmd).fit(x)

    # Test without a target
    assert model.score(x) > 0.8


def test_score_with_target(data_2D_superposition):
    x = data_2D_superposition
    model = Koopman().fit(x)

    # Test with a target
    assert model.score(x[::2], y=x[1::2]) > 0.8


def test_score_complex_data(data_random_complex):
    x = data_random_complex
    dmd = DMD(svd_rank=10)
    model = Koopman(regressor=dmd).fit(x)

    with pytest.raises(ValueError):
        model.score(x, cast_as_real=False)


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_observables_integration(data_random, observables):
    x = data_random
    model = Koopman(observables=observables).fit(x)
    check_is_fitted(model)

    y = model.predict(x)
    assert y.shape[1] == x.shape[1]


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
def test_observables_integration_accuracy(data_1D_cosine, observables):
    x = data_1D_cosine
    model = Koopman(observables=observables, quiet=True).fit(x)

    assert model.score(x) > 0.95


def test_simulate_with_time_delay(data_2D_superposition):
    x = data_2D_superposition

    observables = TimeDelay()
    model = Koopman(observables=observables)
    model.fit(x)

    n_steps = 10
    n_consumed_samples = observables.n_consumed_samples
    x_pred = model.simulate(x[: n_consumed_samples + 1], n_steps=n_steps)
    assert_allclose(
        x[n_consumed_samples + 1 : n_consumed_samples + n_steps + 1], x_pred
    )


def test_if_dmdc_model_is_accurate_with_known_controlmatrix(
    data_2D_linear_control_system,
):
    X, C, A, B = data_2D_linear_control_system
    # model = Koopman()

    DMDc = regression.DMDc(svd_rank=3, control_matrix=B)
    model = Koopman(regressor=DMDc).fit(X, C)
    Aest = model.state_transition_matrix
    assert_allclose(Aest, A, 1e-07, 1e-12)


def test_if_dmdc_model_is_accurate_with_unknown_controlmatrix(
    data_2D_linear_control_system,
):
    X, C, A, B = data_2D_linear_control_system
    # model = Koopman()

    DMDc = regression.DMDc(svd_rank=3)
    model = Koopman(regressor=DMDc)
    model.fit(X, C)
    Aest = model.state_transition_matrix
    Best = model.control_matrix
    assert_allclose(Aest, A, 1e-07, 1e-12)
    assert_allclose(Best, B, 1e-07, 1e-12)


def test_simulate_accuracy_dmdc(data_2D_linear_control_system):
    X, C, _, _ = data_2D_linear_control_system

    DMDc = regression.DMDc(svd_rank=3)
    model = Koopman(regressor=DMDc).fit(X, C)

    n_steps = len(C)
    x_pred = model.simulate(X[0, :], C, n_steps=n_steps - 1)
    assert_allclose(X[1:n_steps, :], x_pred, 1e-07, 1e-12)


def test_misc_drss_measurement_matrix():
    A, B, C = drss(2, 2, 0)
    assert_allclose(C, np.identity(2))


def test_dmdc_for_highdim_system(data_drss):
    Y, U, A, B, C = data_drss

    DMDc = regression.DMDc(svd_rank=7, svd_output_rank=5)
    model = Koopman(regressor=DMDc)
    model.fit(Y, U)

    # Check spectrum
    Aest = model.state_transition_matrix
    W, V = np.linalg.eig(A)
    West, Vest = np.linalg.eig(Aest)

    idxTrue = np.argsort(W)
    idxEst = np.argsort(West)
    assert_allclose(W[idxTrue], West[idxEst], 1e-07, 1e-12)

    # Check eigenvector reconstruction
    r = 5
    Uc, sc, Vch = np.linalg.svd(C, full_matrices=False)
    Sc = np.diag(sc[:r])
    Cinv = np.dot(Vch[:, :r].T, np.dot(np.linalg.inv(Sc), Uc[:, :r].T))
    P = model.projection_matrix_output
    Atilde = np.dot(Cinv, np.dot(np.dot(P, np.dot(Aest, P.T)), C))
    Wtilde, Vtilde = np.linalg.eig(Atilde)

    idxTilde = np.argsort(Wtilde)
    assert_allclose(abs(W[idxTrue]), abs(Wtilde[idxTilde]), 1e-07, 1e-12)
    # Evecs may be accurate up to a sign; ensured by seeding random generator
    # when producing the data set
    assert_allclose(abs(V[:, idxTrue]), abs(Vtilde[:, idxTilde]), 1e-07, 1e-12)


def test_torus_unforced(data_torus_unforced):
    xhat, frequencies, dt = data_torus_unforced

    frequencies_est = np.zeros(xhat.shape[0])
    for k in range(xhat.shape[0]):
        spec = np.fft.fft(xhat[k, :])
        freq = np.fft.fftfreq(len(xhat[0, :]), dt)
        frequencies_est[k] = freq[np.argmax(abs(spec))]

    assert_allclose(np.sort(frequencies), np.sort(frequencies_est), 1e-02, 1e-01)


def test_torus_discrete_time(data_torus_ct, data_torus_dt):
    xhat_ct = data_torus_ct
    xhat_dt = data_torus_dt

    assert_allclose(xhat_ct, xhat_dt, 1e-12, 1e-12)


# TODO: test torus mode id with dmdc
