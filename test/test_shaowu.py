import pytest


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
