"""BM25 lexical scoring, merged with vector ranks (hybrid retrieval).

Lexical matching rescues exact vocabulary ("Idempotency-Key", "half-up", "created_from") that a
small embedding model can blur; the vector side rescues paraphrases.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "is",
    "in",
    "for",
    "on",
    "with",
    "be",
    "it",
    "that",
    "this",
    "as",
    "are",
    "by",
    "at",
    "we",
    "our",
    "must",
    "not",
    "no",
    "if",
    "from",
    "any",
}


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.4, b: float = 0.75) -> None:
        self.ids = list(docs)
        self.tf = [Counter(tokens(docs[i])) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df: Counter[str] = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[tuple[str, float]]:
        q = set(tokens(query))
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += (
                        self.idf[t]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                    )
            out.append((self.ids[i], s))
        return sorted(out, key=lambda x: -x[1])


def interleave(*rankings: list[str], k: int) -> list[str]:
    """Round-robin merge: rank 1 of every ranking, then rank 2, ... (deduplicated), so the best hit
    of each query/method always makes it in. Simpler than score fusion on a small corpus, where
    a chunk that is mediocre everywhere would otherwise beat one that is first somewhere."""
    out: list[str] = []
    depth = max((len(r) for r in rankings), default=0)
    for i in range(depth):
        for r in rankings:
            if i < len(r) and r[i] not in out:
                out.append(r[i])
                if len(out) == k:
                    return out
    return out
