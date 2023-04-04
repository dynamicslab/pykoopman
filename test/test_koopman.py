"""Tests for (discrete) Koopman objects."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydmd import DMD
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process.kernels import RBF
from sklearn.utils.validation import check_is_fitted

from pykoopman import Koopman
from pykoopman import observables
from pykoopman import regression
from pykoopman.common import drss
from pykoopman.common import examples
from pykoopman.common import Linear2Ddynamics
from pykoopman.observables import Identity
from pykoopman.observables import Polynomial
from pykoopman.observables import RadialBasisFunction
from pykoopman.observables import RandomFourierFeatures
from pykoopman.observables import TimeDelay
from pykoopman.regression import EDMD
from pykoopman.regression import KDMD
from pykoopman.regression import NNDMD
from pykoopman.regression import PyDMDRegressor


def test_fit(data_random):
    x = data_random
    model = Koopman().fit(x)
    check_is_fitted(model)


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        KDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
        EDMD(svd_rank=10),
        NNDMD,
    ],
)
def test_predict_shape(data_random, regressor):
    x = data_random
    model = Koopman(regressor=regressor).fit(x)
    assert x.shape == model.predict(x).shape


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=5, tlsq_rank=2),
        EDMD(svd_rank=5, tlsq_rank=2),
        PyDMDRegressor(DMD(svd_rank=5, tlsq_rank=2)),
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
    ],
)
def test_simulate_accuracy(data_2D_superposition, regressor):
    x = data_2D_superposition
    # regressor = PyDMDRegressor(DMD(svd_rank=10))
    model = Koopman(regressor=regressor).fit(x)
    n_steps = 50
    x_pred = model.simulate(x[0], n_steps=n_steps)
    assert_allclose(x[1 : n_steps + 1], x_pred)


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=5, tlsq_rank=2),
        EDMD(svd_rank=5, tlsq_rank=2),
        PyDMDRegressor(DMD(svd_rank=5, tlsq_rank=2)),
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
    ],
)
def test_dmd_on_nonconsecutive_data_accuracy(data_2D_linear_real_system, regressor):
    x = data_2D_linear_real_system
    # regressor = regression.PyDMDRegressor(
    #     DMD(svd_rank=2, tlsq_rank=0, forward_backward=False)
    # )
    x_pair = np.hstack([x[:-1], x[1:]])
    x_pair_random = x_pair[np.random.permutation(x_pair.shape[0])]
    model = Koopman(regressor=regressor).fit(
        x=x_pair_random[:, :2], y=x_pair_random[:, 2:]
    )
    n_steps = 9
    x_pred = model.simulate(x[0], n_steps=n_steps)
    assert_allclose(x[1 : n_steps + 1], x_pred)


# def test_koopman_matrix_shape(data_random):
#     x = data_random
#     dmd = DMD(svd_rank=10)
#     model = Koopman(regressor=dmd).fit(x)
#     assert model.koopman_matrix.shape[0] == model.n_output_features_


def test_if_fitted(data_random):
    x = data_random
    model = Koopman()

    with pytest.raises(NotFittedError):
        model.predict(x)
    with pytest.raises(NotFittedError):
        model.simulate(x)
    with pytest.raises(NotFittedError):
        model.A
    with pytest.raises(NotFittedError):
        model.B
    with pytest.raises(NotFittedError):
        model.C
    with pytest.raises(NotFittedError):
        model._step(x)


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
        # KDMD(svd_rank=10) # kernel dmd cannot be applied to complex data
    ],
)
def test_score_without_target(data_2D_superposition, regressor):
    x = data_2D_superposition
    # dmd = DMD(svd_rank=10)
    model = Koopman(regressor=regressor).fit(x)

    # Test without a target
    assert model.score(x) > 0.8


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
    ],
)
def test_score_with_target(data_2D_superposition, regressor):
    x = data_2D_superposition
    model = Koopman(regressor=regressor).fit(x)
    # Test with a target
    assert model.score(x[::2], y=x[1::2]) > 0.8


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
    ],
)
def test_score_complex_data(data_random_complex, regressor):
    x = data_random_complex
    # dmd = DMD(svd_rank=10)
    model = Koopman(regressor=regressor).fit(x)

    with pytest.raises(ValueError):
        model.score(x, cast_as_real=False)


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        TimeDelay(),
        RandomFourierFeatures(),
        RadialBasisFunction(),
        pytest.lazy_fixture("data_custom_observables"),
    ],
)
@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
        KDMD(svd_rank=10, kernel=RBF(length_scale=50.0)),
    ],
)
def test_observables_integration(data_random, observables, regressor):
    x = data_random
    model = Koopman(observables=observables, regressor=regressor).fit(x)
    check_is_fitted(model)

    y = model.predict(x)
    assert y.shape[1] == x.shape[1]


@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(),
        Polynomial(degree=4),
        TimeDelay(delay=1),
        TimeDelay(delay=2),
        TimeDelay(delay=4),
        RadialBasisFunction(),
        RandomFourierFeatures(),
    ],
)
@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
        KDMD(svd_rank=10, kernel=RBF(length_scale=1)),
    ],
)
def test_observables_integration_accuracy(data_1D_cosine, observables, regressor):
    x = data_1D_cosine
    model = Koopman(observables=observables, regressor=regressor, quiet=True).fit(x)
    assert model.score(x) > 0.95


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
    ],
)
@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(degree=2),
        TimeDelay(delay=2),
        TimeDelay(delay=4),
    ],
)
def test_simulate_with_time_delay(data_2D_superposition, regressor, observables):
    x = data_2D_superposition
    # observables = TimeDelay(delay=3)
    model = Koopman(observables=observables, regressor=regressor)
    model.fit(x)
    n_steps = 10
    n_consumed_samples = observables.n_consumed_samples
    x_pred = model.simulate(x[: n_consumed_samples + 1], n_steps=n_steps)
    assert_allclose(
        x[n_consumed_samples + 1 : n_consumed_samples + n_steps + 1], x_pred
    )


@pytest.mark.parametrize(
    "regressor",
    [
        DMD(svd_rank=10),
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
    ],
)
@pytest.mark.parametrize(
    "observables",
    [
        Identity(),
        Polynomial(degree=2),
        TimeDelay(delay=2),
        TimeDelay(delay=4),
    ],
)
def test_simulate_with_time_delay_ensemble(
    data_2D_superposition, regressor, observables
):
    x = data_2D_superposition[:-1]
    y = data_2D_superposition[1:]
    # observables = TimeDelay(delay=3)
    model = Koopman(observables=observables, regressor=regressor)
    model.fit(x, y)
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

    DMDc = regression.DMDc(svd_rank=3, input_control_matrix=B)
    model = Koopman(regressor=DMDc).fit(X, u=C)
    Akoopman = model.A
    Aest = model.ur @ Akoopman @ model.ur.T
    assert_allclose(Aest, A, 1e-07, 1e-12)


def test_if_dmdc_model_is_accurate_with_unknown_controlmatrix(
    data_2D_linear_control_system,
):
    X, C, A, B = data_2D_linear_control_system
    # model = Koopman()

    DMDc = regression.DMDc(svd_rank=3)
    model = Koopman(regressor=DMDc)
    model.fit(X, u=C)  # C is not the measurement matrix!
    Akoopman = model.A
    Bkoopman = model.B
    Aest = model.ur @ Akoopman @ model.ur.T
    Best = model.ur @ Bkoopman
    assert_allclose(Aest, A, 1e-07, 1e-12)
    assert_allclose(Best, B, 1e-07, 1e-12)


def test_simulate_accuracy_dmdc(data_2D_linear_control_system):
    X, C, _, _ = data_2D_linear_control_system

    DMDc = regression.DMDc(svd_rank=3)
    model = Koopman(regressor=DMDc).fit(X, u=C)

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
    model.fit(Y, u=U)

    # Check spectrum
    Aest = model.A
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
    P = model.ur
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


def test_edmdc_vanderpol():
    np.random.seed(42)  # For reproducibility
    n_states = 2
    n_inputs = 1
    dT = 0.01

    # Create training data
    n_traj = 200  # Number of trajectories
    n_int = 1000  # Integration length

    # Uniform forcing in [-1, 1]
    u = 2 * np.random.random([n_int, n_traj]) - 1
    # Uniform distribution of initial conditions
    x = 2 * np.random.random([n_states, n_traj]) - 1

    # Init
    X = np.zeros((n_states, n_int * n_traj))
    Y = np.zeros((n_states, n_int * n_traj))
    U = np.zeros((n_inputs, n_int * n_traj))

    # Integrate
    for step in range(n_int):
        y = examples.rk4(0, x, u[step, :], dT, examples.vdp_osc)
        X[:, (step) * n_traj : (step + 1) * n_traj] = x
        Y[:, (step) * n_traj : (step + 1) * n_traj] = y
        U[:, (step) * n_traj : (step + 1) * n_traj] = u[step, :]
        x = y

    # Create Koopman model
    EDMDc = regression.EDMDc()
    RBF = observables.RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=10,
        centers=None,
        kernel_width=1.0,
        polyharmonic_coeff=1.0,
    )
    model = Koopman(observables=RBF, regressor=EDMDc)
    model.fit(x=X.T, y=Y.T, u=U.T)

    # Create test data
    n_int = 300  # Integration length
    u = np.array([-examples.square_wave(step + 1) for step in range(n_int)])
    x = np.array([0.5, 0.5])
    # x = np.array([[-0.1], [-0.5]])

    # Prediction using Koopman model
    Xkoop = model.simulate(x[np.newaxis, :], u[:, np.newaxis], n_steps=n_int - 1)

    # Add initial condition to simulated data for comparison below
    Xkoop = np.vstack([x[np.newaxis, :], Xkoop])

    assert_allclose(
        Xkoop[-1, :], [-8.473305929876546738e-01, 6.199389628993866308e-02], 1e-07, 1e-9
    )


def test_accuracy_of_edmd_prediction(data_rev_dvdp):
    np.random.seed(42)  # for reproducibility
    dT, X0, Xtrain, Ytrain, Xtest = data_rev_dvdp

    regressor = regression.EDMD()
    # regressor = PyDMDRegressor(DMD(svd_rank=22))
    RBF = observables.RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=20,
        centers=None,
        kernel_width=1.0,
        polyharmonic_coeff=1.0,
        include_state=True,
    )

    model = Koopman(observables=RBF, regressor=regressor)
    model.fit(Xtrain.T, y=Ytrain.T)
    Xkoop = model.simulate(Xtest[0, :][np.newaxis, :], n_steps=np.shape(Xtest)[0] - 1)
    Xkoop = np.vstack([Xtest[0, :][np.newaxis, :], Xkoop])
    assert_allclose(Xtest, Xkoop, atol=2e-2, rtol=1e-10)


@pytest.mark.parametrize(
    "regressor",
    [DMD(svd_rank=2), PyDMDRegressor(DMD(svd_rank=2)), EDMD()],
)
def test_accuracy_koopman_validity_check(data_for_validty_check, regressor):
    X, t = data_for_validty_check
    model = Koopman(regressor=regressor)
    model.fit(X, dt=1)
    efun_index, linearity_error = model.validity_check(t, X)
    assert_allclose([0, 0], linearity_error, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "regressor",
    [
        NNDMD(
            look_forward=1,
            config_encoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            config_decoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=True,
            normalize_mode="max",
            trainer_kwargs=dict(max_epochs=3),
        ),
        NNDMD(
            look_forward=1,
            config_encoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            config_decoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=True,
            normalize_mode="equal",
            trainer_kwargs=dict(max_epochs=3),
        ),
        NNDMD(
            look_forward=1,
            config_encoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            config_decoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=False,
            normalize_mode="max",
            trainer_kwargs=dict(max_epochs=3),
        ),
        NNDMD(
            mode="Dissipative",
            look_forward=1,
            config_encoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            config_decoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=True,
            normalize_mode="max",
            trainer_kwargs=dict(max_epochs=3),
        ),
        NNDMD(
            mode="Dissipative",
            look_forward=1,
            config_encoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            config_decoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=True,
            normalize_mode="equal",
            trainer_kwargs=dict(max_epochs=3),
        ),
        NNDMD(
            mode="Dissipative",
            look_forward=1,
            config_encoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            config_decoder=dict(
                input_size=2, hidden_sizes=[32] * 2, output_size=2, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=False,
            normalize_mode="max",
            trainer_kwargs=dict(max_epochs=3),
        ),
    ],
)
def test_accuracy_nndmd_linear_system(regressor):
    # prepare training data
    sys = Linear2Ddynamics()
    n_pts = 51
    n_int = 2
    xx, yy = np.meshgrid(np.linspace(-1, 1, n_pts), np.linspace(-1, 1, n_pts))
    x = np.vstack((xx.flatten(), yy.flatten()))
    n_traj = x.shape[1]
    X, Y = sys.collect_data(x, n_int, n_traj)

    # create a model then train
    model = Koopman(regressor=regressor)
    model.fit(X.T, Y.T)

    # check eigenvalues
    eigenvalues = np.sort(np.real(np.diag(model.lamda)))
    assert_allclose([0.7, 0.8], eigenvalues, rtol=1e-4, atol=1e-4)
