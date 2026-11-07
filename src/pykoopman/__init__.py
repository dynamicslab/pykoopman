from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

from .koopman import Koopman
from .koopman_continuous import KoopmanContinuous


__all__ = [
    "Koopman",
    "KoopmanContinuous",
    "common",
    "differentiation",
    "observables",
    "regression",
    "analytics",
]
