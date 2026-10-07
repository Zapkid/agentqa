"""Disk cache for LLM round-trips.

Key = sha256 over (provider, model, messages, tools, schema, params, prompt version). Modes:
``off`` (never read or write), ``read_write`` (normal), ``replay_only`` (read, and raise
CacheMiss on a miss). Replay mode makes evals deterministic, free and CI-safe.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from agentqa.config import REPO_ROOT
from agentqa.llm.types import CacheMiss, RawResponse

CacheMode = Literal["off", "read_write", "replay_only"]


def cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path | None = None, mode: CacheMode | None = None) -> None:
        self.root = (
            Path(root)
            if root
            else Path(os.environ.get("AGENTQA_CACHE_DIR", str(REPO_ROOT / ".cache" / "llm")))
        )
        self.mode: CacheMode = mode or os.environ.get("AGENTQA_CACHE_MODE", "read_write")  # type: ignore[assignment]
        if self.mode not in ("off", "read_write", "replay_only"):
            raise ValueError(f"invalid cache mode {self.mode!r}")
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> RawResponse | None:
        if self.mode == "off":
            return None
        path = self._path(key)
        if path.exists():
            self.hits += 1
            return RawResponse.model_validate_json(path.read_text(encoding="utf-8"))
        self.misses += 1
        if self.mode == "replay_only":
            raise CacheMiss(f"replay_only: no cached response for key {key[:12]}")
        return None

    def put(self, key: str, response: RawResponse) -> None:
        if self.mode != "read_write":
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_dump_json(), encoding="utf-8")
        tmp.replace(path)
