"""
Sparse vector utilities for hybrid search (BM25-style, hash-based).
Produces sparse vectors suitable for Milvus SPARSE_FLOAT_VECTOR (if supported).
"""
from __future__ import annotations

import re
import math
# Fixed vocabulary size (hash bucket count) for sparse vectors
SPARSE_DIM = 30000


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, return non-empty tokens."""
    if not text:
        return []
    text = text.lower().strip()
    tokens = re.split(r"[^a-z0-9_]+", text)
    return [t for t in tokens if len(t) > 1]


def text_to_sparse_vector(text: str, *, dim: int = SPARSE_DIM) -> dict[int, float]:
    """
    Build a sparse vector from text (hash-based term weights).
    Returns a dict {index: value} as required by pymilvus for SPARSE_FLOAT_VECTOR insert/search.
    """
    tokens = _tokenize(text)
    if not tokens:
        return {}
    # Term frequencies in this document
    tf: dict[int, float] = {}
    for t in tokens:
        idx = hash(t) % dim
        if idx < 0:
            idx += dim
        tf[idx] = tf.get(idx, 0.0) + 1.0
    # Sublinear tf: 1 + log(tf)
    for k in tf:
        tf[k] = 1.0 + math.log(tf[k])
    # L2 normalize so inner product = cosine similarity
    norm = math.sqrt(sum(v * v for v in tf.values()))
    if norm <= 0:
        return {}
    return {i: (v / norm) for i, v in tf.items()}
