"""Veroniqa's Streamlit interface, driven headlessly with Streamlit's AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "agentqa/veroniqa/app.py")


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTQA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTQA_CACHE_MODE", "off")
    monkeypatch.setenv("AGENTQA_PROFILE", "simulated")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _click(at: AppTest, label: str) -> None:
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception


def test_welcome_screen_without_projects() -> None:
    at = _app()
    assert at.header[0].value == "Welcome to Veroniqa"


def test_demo_project_chat_knowledge_and_runs_tab() -> None:
    at = _app()
    _click(at, "Create the Orders API demo project")
    assert at.header[0].value == "Orders API demo"
    texts = " ".join(m.value for m in at.markdown)
    assert "poisoned.md" in texts and "quarantined" in texts  # Knowledge tab lists sources
    assert any(b.label == "Run the tests" for b in at.button)  # Test runs tab

    at.chat_input[0].set_value("What is the p95 latency target for the order list?").run()
    assert not at.exception, at.exception
    texts = " ".join(m.value for m in at.markdown)
    assert "300 ms" in texts

    at.chat_input[0].set_value("add http://127.0.0.1:9/admin to the knowledge").run()
    assert "non-public" in " ".join(m.value for m in at.markdown)


def test_new_project_form_creates_a_project() -> None:
    at = _app()
    at.text_input[0].set_value("Billing API")
    next(b for b in at.button if b.label == "Create project").click().run()
    assert not at.exception, at.exception
    assert at.header[0].value == "Billing API"
