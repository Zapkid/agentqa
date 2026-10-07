"""F1 Ingestor (no LLM, except the optional injection classifier on borderline chunks).

spec + docs -> SpecBundle (endpoints, chunks with quarantine flags, per-endpoint hashes) and an
idempotent upsert into the vector store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from agentqa.guards import injection
from agentqa.ingest.chunker import doc_chunks, endpoint_chunk
from agentqa.ingest.openapi import endpoint_hash, load_spec, parse_endpoints, spec_id
from agentqa.ingest.vectorstore import VectorStore
from agentqa.models import Chunk, SpecBundle
from agentqa.obs import tracing


def doc_paths(docs: str | Path | list[Path] | None) -> list[Path]:
    if docs is None:
        return []
    if isinstance(docs, list):
        return sorted(docs)
    p = Path(docs)
    return sorted(p.glob("*.md")) if p.is_dir() else [p]


def ingest(
    spec_source: str | Path,
    docs: str | Path | list[Path] | None,
    store: VectorStore,
    classifier: injection.Classifier | None = None,
) -> tuple[SpecBundle, dict[str, int]]:
    with tracing.span("ingest", "stage", **{"agentqa.spec": str(spec_source)}) as sp:
        raw = load_spec(spec_source)
        endpoints = parse_endpoints(raw)
        chunks: list[Chunk] = [endpoint_chunk(ep) for ep in endpoints]
        paths = doc_paths(docs)
        for path in paths:
            chunks.extend(doc_chunks(path))
        chunks = injection.apply(chunks, classifier)
        docs_digest = hashlib.sha256(
            "".join(c.sha for c in chunks if c.kind == "doc").encode()
        ).hexdigest()[:8]
        sid = f"{spec_id(raw)}-{docs_digest}"
        bundle = SpecBundle(
            spec_id=sid,
            title=raw.get("info", {}).get("title", "API"),
            version=str(raw.get("info", {}).get("version", "")),
            endpoints=endpoints,
            chunks=chunks,
            endpoint_hashes={ep.id: endpoint_hash(ep) for ep in endpoints},
        )
        counts = store.upsert(sid, chunks)
        quarantined = [c.id for c in chunks if c.quarantined]
        sp.set_attribute("agentqa.endpoints", len(endpoints))
        sp.set_attribute("agentqa.chunks", len(chunks))
        sp.set_attribute("agentqa.quarantined", quarantined)
        return bundle, {**counts, "quarantined": len(quarantined)}
