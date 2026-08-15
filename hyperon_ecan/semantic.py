"""Inject a tiny Neural Space by giving named clusters a shared direction.

Hashed n-grams only capture morphology (`dog`/`dogs`). Semantic neighbors
such as `dog`/`wolf` need either a trained embedding or an explicit cluster.
This helper is the second option, so experiments do not pretend a hash
knows biology.
"""

from __future__ import annotations

import numpy as np

from .network import ECAN
from .neural import hashed_embedding


def attach_clusters(ecan: ECAN, clusters: list[list[str]], blend: float = 0.78) -> None:
    """Place each cluster around a distinct unit vector, plus a hash residual."""
    dim = ecan.p.embedding_dim
    for i, group in enumerate(clusters):
        shared = np.zeros(dim, dtype=np.float64)
        shared[i % dim] = 1.0
        shared[(i * 5 + 3) % dim] = 0.65
        shared = shared / np.linalg.norm(shared)
        for name in group:
            ecan.add(name)
            residual = hashed_embedding(name, dim)
            ecan.set_embedding(name, blend * shared + (1.0 - blend) * residual)
