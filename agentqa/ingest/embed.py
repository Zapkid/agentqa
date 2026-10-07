"""Local embeddings (no API quota).

``fastembed`` (ONNX, BAAI/bge-small-en-v1.5) is used when installed and its model can be
loaded; otherwise a deterministic feature-hashing embedder is used. The hashing embedder needs
no download, which keeps CI and air-gapped installs working; retrieval quality is lower but
adequate for a corpus of a few hundred chunks. Choose with ``AGENTQA_EMBEDDER=hashing|fastembed``.
See ADR 0004.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Protocol

TOKEN = re.compile(r"[a-z0-9]+")  # snake_case field names split into words


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Signed feature hashing of unigrams and bigrams, L2-normalised."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        toks = TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self.name = f"fastembed-{model}"
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


_default: Embedder | None = None


def default_embedder() -> Embedder:
    global _default
    if _default is None:
        choice = os.environ.get("AGENTQA_EMBEDDER", "hashing")
        if choice == "fastembed":
            _default = FastEmbedEmbedder()
        else:
            _default = HashingEmbedder()
    return _default


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
