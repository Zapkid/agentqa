"""VeroniQA projects: one folder per project.

<AGENTQA_HOME>/projects/<slug>/
    project.yaml     name, description, spec, target settings
    sources.json     the knowledge sources (uploaded files and links)
    docs/            extracted text of every source, as Markdown (also the requirement docs
                     for test runs)
    chroma/          the project's vector store
    runs/            test runs started from this project
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agentqa.config import REPO_ROOT, agentqa_home

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class ProjectConfig(BaseModel):
    name: str
    slug: str
    description: str = ""
    created_at: str
    spec: str | None = None  # OpenAPI/Swagger file inside the project, or an http(s) URL
    base_url: str | None = None  # the API under test
    target_config: str | None = None  # target YAML: auth scheme, roles, sandbox flag
    local_demo: bool = False  # run against the bundled Orders API (local builds, seeded bugs)


class Source(BaseModel):
    id: str
    kind: Literal["file", "link"]
    name: str
    file: str  # the extracted Markdown under docs/
    url: str | None = None
    added_at: str
    chunks: int = 0
    quarantined: int = 0


class Project:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = ProjectConfig.model_validate(
            yaml.safe_load((root / "project.yaml").read_text(encoding="utf-8"))
        )

    @property
    def slug(self) -> str:
        return self.config.slug

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma"

    def sources(self) -> list[Source]:
        path = self.root / "sources.json"
        if not path.exists():
            return []
        return [Source.model_validate(s) for s in json.loads(path.read_text(encoding="utf-8"))]

    def save_sources(self, sources: list[Source]) -> None:
        (self.root / "sources.json").write_text(
            json.dumps([s.model_dump() for s in sources], indent=1), encoding="utf-8"
        )

    def save(self) -> None:
        (self.root / "project.yaml").write_text(
            yaml.safe_dump(self.config.model_dump(), sort_keys=False), encoding="utf-8"
        )

    def update(self, **fields: object) -> None:
        self.config = self.config.model_copy(update=fields)
        self.save()


def projects_root() -> Path:
    return agentqa_home() / "projects"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48].strip("-")
    if not slug:
        raise ValueError("a project name needs at least one letter or digit")
    return slug


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_project(
    name: str,
    description: str = "",
    *,
    spec: str | None = None,
    base_url: str | None = None,
    target_config: str | None = None,
    local_demo: bool = False,
    root: Path | None = None,
) -> Project:
    root = root or projects_root()
    base = slugify(name)
    slug, n = base, 2
    while (root / slug).exists():
        slug, n = f"{base}-{n}", n + 1
    folder = root / slug
    for sub in ("docs", "runs", "chroma"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    config = ProjectConfig(
        name=name.strip()[:120],
        slug=slug,
        description=description.strip()[:2000],
        created_at=_now(),
        spec=spec,
        base_url=base_url,
        target_config=target_config,
        local_demo=local_demo,
    )
    (folder / "project.yaml").write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False), encoding="utf-8"
    )
    return Project(folder)


def load_project(slug: str, root: Path | None = None) -> Project:
    if not SLUG_RE.match(slug):
        raise ValueError(f"invalid project id {slug!r}")
    root = (root or projects_root()).resolve()
    folder = (root / slug).resolve()
    if folder.parent != root or not (folder / "project.yaml").exists():
        raise FileNotFoundError(f"no project {slug!r}")
    return Project(folder)


def list_projects(root: Path | None = None) -> list[Project]:
    root = root or projects_root()
    if not root.exists():
        return []
    found = [Project(p) for p in sorted(root.iterdir()) if (p / "project.yaml").exists()]
    return sorted(found, key=lambda p: p.config.created_at, reverse=True)


def create_demo_project(root: Path | None = None) -> Project:
    """A project wired to the bundled Orders API: its spec, and its requirement docs copied in as
    sources (including the poisoned one, which the injection scan quarantines)."""
    from agentqa.veroniqa.knowledge import KnowledgeBase

    project = create_project(
        "Orders API demo",
        "The bundled Orders and Invoicing API with 12 seeded bugs.",
        spec=str(REPO_ROOT / "target_api/openapi.json"),
        target_config=str(REPO_ROOT / "target_api/agentqa_target.yaml"),
        local_demo=True,
        root=root,
    )
    kb = KnowledgeBase(project)
    for doc in sorted((REPO_ROOT / "target_api/docs").glob("*.md")):
        kb.add_file(doc.name, doc.read_bytes())
    return project


def delete_project(project: Project) -> None:
    """Remove a project folder (only ever a folder inside the projects root)."""
    root = projects_root().resolve()
    folder = project.root.resolve()
    if folder.parent != root:
        raise ValueError("refusing to delete a folder outside the projects root")
    shutil.rmtree(folder)


__all__ = [
    "Project",
    "ProjectConfig",
    "Source",
    "create_demo_project",
    "create_project",
    "delete_project",
    "list_projects",
    "load_project",
    "slugify",
]
