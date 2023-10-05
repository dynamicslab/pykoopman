"""Test for pykoopman.analytics"""
from __future__ import annotations

import numpy as np
import pytest

import pykoopman as pk
from pykoopman.analytics import ModesSelectionPAD21
from pykoopman.common import Linear2Ddynamics


@pytest.fixture
def data_linear_dynamics():
    # Create instance of the dynamical system
    sys = Linear2Ddynamics()

    # Collect training data
    n_pts = 51
    n_int = 1
    xx, yy = np.meshgrid(np.linspace(-1, 1, n_pts), np.linspace(-1, 1, n_pts))
    x = np.vstack((xx.flatten(), yy.flatten()))
    n_traj = x.shape[1]

    X, Y = sys.collect_data(x, n_int, n_traj)
    return X, Y, sys


def test_sparse_selection(data_linear_dynamics):
    X, Y, sys = data_linear_dynamics

    # run a vanilla model with polynomial features
    regressor = pk.regression.EDMD()
    obsv = pk.observables.Polynomial(degree=3)
    model = pk.Koopman(observables=obsv, regressor=regressor)
    model.fit(X.T, y=Y.T)

    # generate some validation trajectories
    # first trajectory
    n_int_val = 41
    # n_traj_val = 1
    xval = np.array([[-0.3], [-0.3]])
    xval_list = []
    for i in range(n_int_val):
        xval_list.append(xval)
        xval = sys.linear_map(xval)
    Xval1 = np.hstack(xval_list).T

    # second trajectory
    n_int_val = 17
    # n_traj_val = 1
    xval = np.array([[-0.923], [0.59]])
    xval_list = []
    for i in range(n_int_val):
        xval_list.append(xval)
        xval = sys.linear_map(xval)
    Xval2 = np.hstack(xval_list).T

    n_int_val = 23
    # n_traj_val = 1
    xval = np.array([[-2.5], [1.99]])
    xval_list = []
    for i in range(n_int_val):
        xval_list.append(xval)
        xval = sys.linear_map(xval)
    Xval3 = np.hstack(xval_list).T

    # combine three trajectories together
    Xval = [Xval1, Xval2, Xval3]
    # assemble them as a dictionary
    validate_data_traj = [{"t": np.arange(tmp.shape[0]), "x": tmp} for tmp in Xval]

    # perform analysis -- just to check if everything is running
    analysis = ModesSelectionPAD21(
        model, validate_data_traj, truncation_threshold=1e-3, plot=False
    )
    analysis.sweep_among_best_L_modes(
        L=6, ALPHA_RANGE=np.logspace(-7, 1, 10), save_figure=False, plot=False
    )
    analysis.prune_model(i_alpha=6, x_train=X.T)
