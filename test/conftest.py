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
