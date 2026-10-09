"""Chroma-backed chunk store. Upserts are keyed by chunk id and skipped when the content hash is
unchanged, so ingestion is idempotent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings

from agentqa.ingest.embed import Embedder, default_embedder
from agentqa.ingest.lexical import BM25, interleave
from agentqa.models import Chunk


class VectorStore:
    def __init__(
        self,
        path: Path | None = None,
        collection: str = "agentqa-chunks",
        embedder: Embedder | None = None,
    ) -> None:
        settings = Settings(anonymized_telemetry=False)
        self.client = (
            chromadb.PersistentClient(path=str(path), settings=settings)
            if path
            else chromadb.EphemeralClient(settings=settings)
        )
        self.embedder = embedder or default_embedder()
        name = f"{collection}-{self.embedder.name}".replace("/", "-")[:60]
        self._bm25: dict[tuple[str, str], BM25] = {}
        self.col = self.client.get_or_create_collection(
            name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )

    def upsert(self, spec_id: str, chunks: list[Chunk]) -> dict[str, int]:
        """Returns counts of {added, updated, unchanged}."""
        ids = [f"{spec_id}:{c.id}" for c in chunks]
        existing: Any = (
            self.col.get(ids=ids, include=["metadatas"]) if ids else {"ids": [], "metadatas": []}
        )
        known = {
            i: (m or {}).get("sha")
            for i, m in zip(existing["ids"], existing["metadatas"] or [], strict=False)
        }
        todo = [(i, c) for i, c in zip(ids, chunks, strict=True) if known.get(i) != c.sha]
        counts = {"added": 0, "updated": 0, "unchanged": len(chunks) - len(todo)}
        if todo:
            vectors = self.embedder.embed([c.text for _, c in todo])
            self.col.upsert(
                ids=[i for i, _ in todo],
                documents=[c.text for _, c in todo],
                embeddings=cast(Any, vectors),
                metadatas=[self._meta(spec_id, c) for _, c in todo],
            )
            for i, _ in todo:
                counts["updated" if i in known else "added"] += 1
            self._bm25.clear()
        return counts

    def _lexical(self, spec_id: str, kind: str | None) -> BM25:
        key = (spec_id, kind or "*")
        if key not in self._bm25:
            where: dict[str, Any] = {"$and": [{"spec_id": spec_id}, {"quarantined": False}]}
            if kind:
                where["$and"].append({"kind": kind})
            got: Any = self.col.get(where=cast(Any, where), include=["documents", "metadatas"])
            docs = {
                str((m or {}).get("chunk_id")): d or ""
                for m, d in zip(got["metadatas"] or [], got["documents"] or [], strict=True)
            }
            self._bm25[key] = BM25(docs)
        return self._bm25[key]

    def hybrid(
        self, spec_id: str, queries: list[str], k: int = 4, kind: str | None = None
    ) -> list[tuple[str, float]]:
        """Vector and BM25 rankings for each query, merged round-robin. Score = 1/position."""
        rankings: list[list[str]] = []
        lex = self._lexical(spec_id, kind)
        for q in queries:
            rankings.append([cid for cid, _ in self.search(spec_id, q, k=2 * k + 4, kind=kind)])
            rankings.append([cid for cid, s in lex.scores(q)[: 2 * k + 4] if s > 0])
        return [(cid, 1.0 / (i + 1)) for i, cid in enumerate(interleave(*rankings, k=k))]

    def get(self, spec_id: str, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Stored text and metadata for chunk ids, in the order given (unknown ids are skipped)."""
        if not chunk_ids:
            return []
        got: Any = self.col.get(
            ids=[f"{spec_id}:{c}" for c in chunk_ids], include=["documents", "metadatas"]
        )
        by_id = {
            str((m or {}).get("chunk_id")): {**(m or {}), "text": d or ""}
            for m, d in zip(got["metadatas"] or [], got["documents"] or [], strict=True)
        }
        return [by_id[c] for c in chunk_ids if c in by_id]

    def delete_doc(self, spec_id: str, doc: str) -> None:
        """Remove every chunk of one document."""
        self.col.delete(where=cast(Any, {"$and": [{"spec_id": spec_id}, {"doc": doc}]}))
        self._bm25.clear()

    @staticmethod
    def _meta(spec_id: str, c: Chunk) -> dict[str, Any]:
        return {
            "spec_id": spec_id,
            "chunk_id": c.id,
            "kind": c.kind,
            "endpoint_id": c.endpoint_id or "",
            "doc": c.doc or "",
            "section": c.section or "",
            "sha": c.sha,
            "quarantined": c.quarantined,
        }

    def search(
        self, spec_id: str, query: str, k: int = 3, kind: str | None = None
    ) -> list[tuple[str, float]]:
        """Top-k (chunk_id, similarity) among non-quarantined chunks of a spec."""
        where: dict[str, Any] = {"$and": [{"spec_id": spec_id}, {"quarantined": False}]}
        if kind:
            where["$and"].append({"kind": kind})
        res: Any = self.col.query(
            query_embeddings=cast(Any, self.embedder.embed([query])),
            n_results=k,
            where=cast(Any, where),
        )
        ids = res["ids"][0] if res["ids"] else []
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(ids)
        return [
            (str((m or {}).get("chunk_id")), 1.0 - float(d))
            for m, d in zip(metas, dists, strict=True)
        ]
