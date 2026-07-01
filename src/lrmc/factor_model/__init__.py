from ._base import MatrixCompletionBase
from .cmc import CMC
from .larsmc import LarsMC
from .lmc import LMC
from .scmc import SCMC
from .tvmc import TVMC
from .wcmc import WCMC, WCMCADMM

try:
    from .gcrnn import GraphConvRNN
    _gcrnn_available = True
except Exception:
    _gcrnn_available = False

__all__ = [
    "LMC",
    "CMC",
    "SCMC",
    "WCMC",
    "WCMCADMM",
    "TVMC",
    "LarsMC",
    "GraphConvRNN",
    "MatrixCompletionBase",
]
