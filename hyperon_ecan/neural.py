"""Deterministic hashed n-gram embeddings and cosine similarity.

Default embeddings need no trained model. Tokens that share character n-grams
(`dog`/`dogs`, `wolf`/`wolves`) land near each other, which is enough to
drive implicit Hebbian spreading. Callers can replace a vector with any
external embedding (LLM, word2vec, …).
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


def _stable_int(text: str, salt: bytes) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=salt).digest()
    return int.from_bytes(digest, "little", signed=False)


def hashed_embedding(text: str, dim: int = 64, n: int = 3) -> np.ndarray:
    """Signed hashed n-gram bag, L2-normalized. Stable across processes."""
    vec = np.zeros(dim, dtype=np.float64)
    padded = f"#{text.lower()}#"
    if len(padded) < n:
        padded = padded.ljust(n, "#")
    for i in range(len(padded) - n + 1):
        gram = padded[i : i + n]
        idx = _stable_int(gram, b"idx-ecan") % dim
        sign = 1.0 if _stable_int(gram, b"sgn-ecan") % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_similarity(
    embeddings: dict[str, np.ndarray],
    threshold: float,
) -> dict[tuple[str, str], float]:
    """Sparse upper-triangle of (cos - threshold)+ between all atoms."""
    keys = list(embeddings)
    links: dict[tuple[str, str], float] = {}
    for i, a in enumerate(keys):
        ea = embeddings[a]
        for b in keys[i + 1 :]:
            score = cosine(ea, embeddings[b]) - threshold
            if score > 0:
                links[(a, b)] = score
    return links


def mean_embedding(vectors: Iterable[np.ndarray], dim: int) -> np.ndarray:
    stacked = list(vectors)
    if not stacked:
        return np.zeros(dim, dtype=np.float64)
    mean = np.mean(np.stack(stacked, axis=0), axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    return mean
