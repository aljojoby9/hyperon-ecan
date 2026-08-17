"""Economic Attention Networks plus a neural spreading space for Hyperon.

Implements Variant 1 of Iklé, Pitt, Goertzel & Sellman (2009), then extends
it with implicit Hebbian links from atom embeddings. Classic OpenCog had ECAN;
Hyperon does not yet ship a working port. Neural spreading along embeddings
was never in Classic either.
"""

from .params import ECANParams
from .network import ECAN
from .inference import AttentionChainer, Triple
from .cycle import CognitiveCycle
from .neural import hashed_embedding, cosine
from .semantic import attach_clusters

try:
    from .ext import ecan_atoms  # MeTTa looks for this on the package
except Exception:
    pass

__all__ = [
    "ECAN",
    "ECANParams",
    "AttentionChainer",
    "Triple",
    "CognitiveCycle",
    "hashed_embedding",
    "cosine",
    "attach_clusters",
]
