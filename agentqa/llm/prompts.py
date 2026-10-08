"""Versioned prompt registry.

Each prompt is ``prompts/<name>.md`` with YAML front matter (``name``, ``version`` as semver,
``role``, ``description``) and two sections, ``## system`` (static, cacheable) and ``## user``
(a Jinja template). The file hash goes into the LLM cache key and into span attributes, so a
quality change can be tied to a prompt change.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from agentqa.config import REPO_ROOT
from agentqa.llm.types import CallMetadata, Message

PROMPTS_DIR = REPO_ROOT / "prompts"
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)  # noqa: S701 - prompt templates are plain text for a model, not HTML


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    role: str
    description: str
    system: str
    user_template: str
    sha256: str

    def render(self, **variables: Any) -> list[Message]:
        user = _env.from_string(self.user_template).render(**variables).strip()
        return [
            Message(role="system", content=self.system, cache_prefix=True),
            Message(role="user", content=user),
        ]

    def metadata(self, agent: str, **extra: Any) -> CallMetadata:
        return CallMetadata(
            agent=agent,
            prompt_name=self.name,
            prompt_version=self.version,
            prompt_hash=self.sha256[:16],
            **extra,
        )


def parse_prompt(text: str, source: str = "<string>") -> Prompt:
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not match:
        raise ValueError(f"{source}: missing YAML front matter")
    meta = yaml.safe_load(match.group(1)) or {}
    body = match.group(2)
    for key in ("name", "version", "role"):
        if key not in meta:
            raise ValueError(f"{source}: front matter missing {key!r}")
    if not SEMVER.match(str(meta["version"])):
        raise ValueError(f"{source}: version {meta['version']!r} is not semver")
    sections = re.split(r"^## (system|user)\s*$", body, flags=re.M)
    parts = {sections[i]: sections[i + 1].strip() for i in range(1, len(sections) - 1, 2)}
    if "system" not in parts or "user" not in parts:
        raise ValueError(f"{source}: needs '## system' and '## user' sections")
    return Prompt(
        name=str(meta["name"]),
        version=str(meta["version"]),
        role=str(meta["role"]),
        description=str(meta.get("description", "")),
        system=parts["system"],
        user_template=parts["user"],
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


@lru_cache
def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> Prompt:
    path = directory / f"{name}.md"
    return parse_prompt(path.read_text(encoding="utf-8"), str(path))


def all_prompts(directory: Path = PROMPTS_DIR) -> list[Prompt]:
    return [load_prompt(p.stem, directory) for p in sorted(directory.glob("*.md"))]
