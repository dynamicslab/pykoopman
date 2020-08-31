import numpy as np

DT_DEFAULT = object()


def validate_input(x, dt=DT_DEFAULT):
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be array-like")
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    return x
