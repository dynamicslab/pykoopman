from __future__ import annotations

import numpy as np
import pytest
from pykoopman.common import check_array
from pykoopman.common import drop_nan_rows
from pykoopman.common import validate_input


def test_validate_input_valid_ndarray():
    x = np.array([[1, 2], [3, 4]])
    result = validate_input(x)
    np.testing.assert_array_equal(x, result)


def test_validate_input_valid_list():
    x = [np.array([1, 2]), np.array([3, 4])]
    result = validate_input(x)
    assert len(result) == 2
    np.testing.assert_array_equal(result[0], x[0])
    np.testing.assert_array_equal(result[1], x[1])


def test_validate_input_1d_array_reshape():
    x = np.array([1, 2, 3])
    result = validate_input(x)
    assert result.ndim == 2
    assert result.shape == (3, 1)


def test_validate_input_invalid_type():
    with pytest.raises(
        ValueError, match="x must be array-like OR a list of array-like"
    ):
        validate_input("invalid_string")


def test_validate_input_time_scalar_positive():
    x = np.array([[1, 2], [3, 4]])
    t = 0.1
    # Should not raise
    validate_input(x, t)


def test_validate_input_time_scalar_non_positive():
    x = np.array([[1, 2], [3, 4]])
    t = 0.0
    with pytest.raises(ValueError, match="t must be positive"):
        validate_input(x, t)

    t = -1.0
    with pytest.raises(ValueError, match="t must be positive"):
        validate_input(x, t)


def test_validate_input_time_array_matching_length():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    t = np.array([0.1, 0.2, 0.3])
    # Should not raise
    validate_input(x, t)


def test_validate_input_time_array_mismatch_length():
    x = np.array([[1, 2], [3, 4]])
    t = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="Length of t should match x.shape"):
        validate_input(x, t)


def test_validate_input_time_array_not_increasing():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    t = np.array([0.1, 0.3, 0.2])
    with pytest.raises(
        ValueError, match="Values in t should be in strictly increasing order"
    ):
        validate_input(x, t)


def test_validate_input_time_invalid_type():
    x = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="t must be a scalar or array-like"):
        validate_input(x, "invalid_time")

    # Example where t is None (which is the default check inside validate_input
    # if T_DEFAULT is not matched, though T_DEFAULT logic is tricky)
    # The code says if t is not T_DEFAULT: if t is None: raise.
    # So we pass None explicitly.
    with pytest.raises(ValueError, match="t must be a scalar or array-like"):
        validate_input(x, None)


def test_drop_nan_rows_single_array():
    arr = np.array([[1, 2], [np.nan, 4], [5, 6]])
    (result_arr,) = drop_nan_rows(arr)
    expected = np.array([[1, 2], [5, 6]])
    np.testing.assert_array_equal(result_arr, expected)


def test_drop_nan_rows_multiple_arrays():
    arr = np.array([[1, 2], [np.nan, 4], [5, 6]])
    other = np.array([[10, 20], [30, 40], [50, 60]])
    result_arr, result_other = drop_nan_rows(arr, other)

    expected_arr = np.array([[1, 2], [5, 6]])
    expected_other = np.array([[10, 20], [50, 60]])

    np.testing.assert_array_equal(result_arr, expected_arr)
    np.testing.assert_array_equal(result_other, expected_other)


def test_check_array_real():
    x = np.array([[1, 2], [3, 4]])
    result = check_array(x)
    np.testing.assert_array_equal(x, result)


def test_check_array_complex():
    x = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    result = check_array(x)
    np.testing.assert_array_equal(x, result)
    # Check if it handles struct/validation logic internally
    # (wrapper around sklearn.utils.check_array)
