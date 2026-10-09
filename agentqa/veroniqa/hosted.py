"""Hosted mode: VeroniQA on a public URL (VERONIQA_HOSTED=1).

A public demo differs from a local install in three ways:

- Each visitor gets a private workspace (its own project folders and vector stores), named by a
  random id kept in the page URL (?w=...), so a reconnect finds the same workspace and visitors
  never see each other's uploads. Workspaces are temporary and pruned after WORKSPACE_TTL.
- Models are simulated: the server holds no API keys, so nothing a visitor does costs money.
- Test runs only target the bundled Orders API. Pointing the server at an arbitrary base URL
  would let anyone use it to send generated traffic to third-party APIs.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import time
from pathlib import Path

from agentqa.config import agentqa_home

WORKSPACE_ID = re.compile(r"^[0-9a-f]{16}$")
WORKSPACE_TTL = 12 * 3600  # seconds
MAX_WORKSPACES = 200


def hosted() -> bool:
    return os.environ.get("VERONIQA_HOSTED") == "1"


def workspaces_root() -> Path:
    return agentqa_home() / "workspaces"


def new_workspace_id() -> str:
    return secrets.token_hex(8)


def workspace_root(workspace_id: str) -> Path:
    """The projects folder of one visitor's workspace. Ids come from URLs, so they are checked."""
    if not WORKSPACE_ID.match(workspace_id):
        raise ValueError("invalid workspace id")
    return workspaces_root() / workspace_id / "projects"


def prune_workspaces(now: float | None = None, keep: str | None = None) -> int:
    """Delete workspaces idle for longer than WORKSPACE_TTL, and the oldest beyond
    MAX_WORKSPACES. Returns how many were removed."""
    root = workspaces_root()
    if not root.exists():
        return 0
    now = time.time() if now is None else now
    found = sorted(
        (p for p in root.iterdir() if p.is_dir() and WORKSPACE_ID.match(p.name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for i, ws in enumerate(found):
        if ws.name == keep:
            continue
        if i >= MAX_WORKSPACES or now - ws.stat().st_mtime > WORKSPACE_TTL:
            shutil.rmtree(ws, ignore_errors=True)
            removed += 1
    return removed


def touch(workspace_id: str) -> None:
    """Mark a workspace as in use (pruning goes by the folder's modification time)."""
    folder = workspace_root(workspace_id).parent
    folder.mkdir(parents=True, exist_ok=True)
    os.utime(folder)
