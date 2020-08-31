"""Tests for pykoopman.regression objects and methods."""
import pytest

from pykoopman.regression import BaseRegressor


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
    with pytest.raises(AttributeError):
        BaseRegressor(regressor)
