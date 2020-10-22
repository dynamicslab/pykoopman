from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
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
]
