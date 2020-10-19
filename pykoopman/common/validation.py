import numpy as np
from sklearn.utils import check_array as skl_check_array

DT_DEFAULT = object()


def validate_input(x, dt=DT_DEFAULT):
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be array-like")
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    return check_array(x)


def check_array(x, **kwargs):
    if np.iscomplexobj(x):
        return skl_check_array(x.real, **kwargs) + 1j * skl_check_array(
            x.imag, **kwargs
        )
    else:
        return skl_check_array(x, **kwargs)
