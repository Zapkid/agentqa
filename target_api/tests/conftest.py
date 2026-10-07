from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from target_api.app.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("BUGS", raising=False)
    with TestClient(app) as c:
        c.post("/__reset")
        yield c
