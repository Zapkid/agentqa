from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("AGENTQA_TRACING", "0")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTQA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTQA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("AGENTQA_KILLSWITCH_FILE", str(tmp_path / "KILLSWITCH"))
    monkeypatch.setenv("AGENTQA_PROFILE", "simulated")
    monkeypatch.delenv("AGENTQA_KILL", raising=False)
