"""A project's knowledge base: uploaded documents and linked pages, chunked, scanned for prompt
injection and stored in the project's own vector store (Chroma + BM25 hybrid retrieval)."""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from agentqa.guards import injection
from agentqa.ingest.chunker import doc_chunks
from agentqa.ingest.embed import Embedder
from agentqa.ingest.vectorstore import VectorStore
from agentqa.veroniqa.fetch import FetchedPage, fetch_url, html_to_text
from agentqa.veroniqa.projects import Project, Source

MAX_UPLOAD_BYTES = 5_000_000
MAX_PDF_PAGES = 300
UPLOAD_TYPES = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf"}


class Passage(BaseModel):
    """A retrieved chunk, numbered for citation."""

    n: int
    chunk_id: str
    source: str
    section: str
    text: str
    url: str | None = None


def extract_text(filename: str, data: bytes) -> str:
    """Plain text of an uploaded file (Markdown is kept as is)."""
    ext = Path(filename).suffix.lower()
    if ext not in UPLOAD_TYPES:
        raise ValueError(f"unsupported file type {ext or '(none)'}: use {sorted(UPLOAD_TYPES)}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("the file is larger than 5 MB")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages[:MAX_PDF_PAGES]]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    text = data.decode("utf-8", errors="replace")
    if ext in (".html", ".htm"):
        title, body = html_to_text(text)
        return f"# {title}\n\n{body}" if title else body
    return text


def _safe_name(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name.strip() or "document"
    return re.sub(r"[^\w.\- ]+", "_", name)[:120]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "source"


class KnowledgeBase:
    def __init__(
        self,
        project: Project,
        *,
        classifier: injection.Classifier | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.project = project
        self.classifier = classifier
        self.store = VectorStore(project.chroma_dir, collection="veroniqa", embedder=embedder)
        self.ns = project.slug

    # ------------------------------------------------------------------ add / remove

    def add_file(self, filename: str, data: bytes) -> Source:
        name = _safe_name(filename)
        return self._add("file", name, extract_text(name, data))

    def add_link(self, url: str, fetcher: Callable[[str], FetchedPage] = fetch_url) -> Source:
        """Index a linked page. A link to API docs (an OpenAPI/Swagger file, or a Swagger UI page
        that loads one) is indexed as a description of every endpoint, and the spec becomes the
        project's spec if it has none yet."""
        page = fetcher(url)
        source = self._add("link", page.title[:120] or page.url, page.text, url=page.url)
        if page.spec_url and not self.project.config.spec:
            self.project.update(spec=page.spec_url)
        return source

    def _add(self, kind: str, name: str, text: str, url: str | None = None) -> Source:
        if not text.strip():
            raise ValueError(f"{name}: no text could be extracted")
        sid = hashlib.sha256(f"{kind}:{url or name}".encode()).hexdigest()[:10]
        file = f"{sid}-{_slug(name)}.md"
        body = text if text.lstrip().startswith("#") else f"# {name}\n\n{text}"
        if url:
            body = f"{body.rstrip()}\n\nSource: {url}\n"
        path = self.project.docs_dir / file
        self.project.docs_dir.mkdir(parents=True, exist_ok=True)
        self.store.delete_doc(self.ns, file)  # re-adding a source replaces its chunks
        path.write_text(body, encoding="utf-8")
        chunks = injection.apply(doc_chunks(path), self.classifier)
        self.store.upsert(self.ns, chunks)
        source = Source(
            id=sid,
            kind=kind,  # type: ignore[arg-type]
            name=name,
            file=file,
            url=url,
            added_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            chunks=len(chunks),
            quarantined=sum(c.quarantined for c in chunks),
        )
        sources = [s for s in self.project.sources() if s.id != sid] + [source]
        self.project.save_sources(sources)
        return source

    def remove(self, source_id: str) -> bool:
        sources = self.project.sources()
        match = next((s for s in sources if s.id == source_id), None)
        if match is None:
            return False
        self.store.delete_doc(self.ns, match.file)
        (self.project.docs_dir / match.file).unlink(missing_ok=True)
        self.project.save_sources([s for s in sources if s.id != source_id])
        return True

    # ------------------------------------------------------------------ retrieval

    def retrieve(self, question: str, k: int = 5) -> list[Passage]:
        """Hybrid (vector + BM25) retrieval over non-quarantined chunks."""
        if not self.project.sources():
            return []
        ids = [cid for cid, _ in self.store.hybrid(self.ns, [question], k=k, kind="doc")]
        by_file = {s.file: s for s in self.project.sources()}
        out = []
        for i, row in enumerate(self.store.get(self.ns, ids), start=1):
            src = by_file.get(str(row.get("doc")))
            out.append(
                Passage(
                    n=i,
                    chunk_id=str(row.get("chunk_id")),
                    source=src.name if src else str(row.get("doc")),
                    section=str(row.get("section", "")),
                    text=str(row["text"]),
                    url=src.url if src else None,
                )
            )
        return out
