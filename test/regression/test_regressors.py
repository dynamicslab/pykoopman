"""Tests for pykoopman.regression objects and methods."""
from __future__ import annotations

import numpy as np
import pykoopman as pk
import pytest
from pydmd import DMD
from pykoopman.regression import BaseRegressor
from pykoopman.regression import EDMD
from pykoopman.regression import KDMD
from pykoopman.regression import NNDMD
from pykoopman.regression import PyDMDRegressor
from sklearn.gaussian_process.kernels import RBF


class RegressorWithoutFit:
    def __init__(self):
        pass

    def predict(self, x):
        return x


class RegressorWithoutPredict:
    def __init__(self):
        pass

    def fit(self, x):
        return self


@pytest.mark.parametrize(
    "regressor", [RegressorWithoutFit(), RegressorWithoutPredict()]
)
def test_bad_regressor_input(regressor):
    """test if BaseRegressor is going to raise TypeError for wrong input"""
    with pytest.raises(TypeError):
        BaseRegressor(regressor)


@pytest.mark.parametrize(
    "data_xy",
    [
        # case 1,2 only work for pykoopman class
        # case 1: single step single traj, no validation
        (np.random.rand(200, 3), None),
        # case 2: single step multiple traj, no validation
        (np.random.rand(200, 3), np.random.rand(200, 3)),
    ],
)
@pytest.mark.parametrize(
    "regressor",
    [
        EDMD(svd_rank=10),
        PyDMDRegressor(DMD(svd_rank=10)),
        KDMD(svd_rank=10, kernel=RBF(length_scale=1)),
    ],
)
def test_fit_regressors(data_xy, regressor):
    """test if using nndmd regressor alone will run the fit without error

    Note:
        `pydmd.DMD` cannot be used to fit nonconsecutive data
    """
    x, y = data_xy
    regressor.fit(x, y)


@pytest.mark.parametrize(
    "data_xy",
    [
        # case 1,2 only work for pykoopman class
        # case 1: single step single traj, no validation
        (np.random.rand(200, 3), None),
        # case 2: single step multiple traj, no validation
        (
            np.random.rand(200, 3),
            np.random.rand(200, 3)  # because "x" is not a list, so we think this
            # is single step
        ),
        # case 3,4 works for regressor directly
        # case 3: multiple traj, no validation
        (
            [np.random.rand(200, 3), np.random.rand(100, 3)],  # this is training
            None,  # no validation
        ),
        # case 4: multiple traj, with validation
        (
            [np.random.rand(100, 3), np.random.rand(100, 3)],  # this is training
            [np.random.rand(300, 3), np.random.rand(400, 3)],  # this is validation
        ),
    ],
)
@pytest.mark.parametrize(
    "regressor",
    [
        NNDMD(
            mode="Dissipative",
            look_forward=2,
            config_encoder=dict(
                input_size=3, hidden_sizes=[32] * 2, output_size=4, activations="swish"
            ),
            config_decoder=dict(
                input_size=4, hidden_sizes=[32] * 2, output_size=3, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=False,
            normalize_mode="max",
            trainer_kwargs=dict(max_epochs=1,accelerator="cpu"),
        )
    ],
)
def test_fit_nndmd_regressor(data_xy, regressor):
    """test if using nndmd regressor alone will run the fit without error"""
    x, y = data_xy
    regressor.fit(x, y)


@pytest.mark.parametrize(
    "data_xy",
    [
        # # case 1,2 only work for pykoopman class
        # # case 1: single step single traj, no validation
        # (
        #         np.random.rand(200, 3),
        #         None
        # ),
        # # case 2: single step multiple traj, no validation
        # (
        #         np.random.rand(200, 3),
        #         np.random.rand(200, 3) # because "x" is not a list, so we think this
        #                                # is single step
        # ),
        # # case 3,4 works for regressor directly
        # # case 3: multiple traj, no validation
        # (
        #         [np.random.rand(200, 3), np.random.rand(100, 3)],  # this is training
        #         None  # no validation
        # ),
        # case 4: multiple traj, with validation
        (
            [np.random.rand(100, 3), np.random.rand(100, 3)],  # this is training
            [np.random.rand(300, 3), np.random.rand(400, 3)],  # this is validation
        ),
    ],
)
@pytest.mark.parametrize(
    "regressor",
    [
        NNDMD(
            mode="Dissipative",
            look_forward=2,
            config_encoder=dict(
                input_size=3, hidden_sizes=[32] * 2, output_size=4, activations="swish"
            ),
            config_decoder=dict(
                input_size=4, hidden_sizes=[32] * 2, output_size=3, activations="linear"
            ),
            batch_size=512,
            lbfgs=True,
            normalize=False,
            normalize_mode="max",
            trainer_kwargs=dict(max_epochs=1,accelerator="cpu"),
        )
    ],
)
def test_fit_dlkoopman(data_xy, regressor):
    """test if using NNDMD regressor work inside pykoopman"""
    model_d = pk.Koopman(regressor=regressor)
    model_d.fit(data_xy[0], data_xy[1], dt=1)
