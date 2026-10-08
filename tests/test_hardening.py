"""Security hardening: behaviour that was tightened, and policy checks that keep it tight."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from agentqa.agents.reporter import _md_file_safe, _md_to_html
from agentqa.executor.runner import suite_config
from agentqa.perf.pipeline import capped_orders
from agentqa.store import Store
from agentqa.target_config import load_target

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- behaviour


def test_save_run_rejects_unknown_columns(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.db")
    store.save_run("r1", status="done", profile="simulated")
    assert store.query("SELECT status FROM runs WHERE run_id='r1'")[0]["status"] == "done"
    with pytest.raises(ValueError, match="unknown run field"):
        store.save_run("r1", **{"status = 'x' --": "boom"})


def test_markdown_file_neutralises_html_outside_code() -> None:
    md = "\n".join(
        [
            "> **Note** <b>bold</b>",
            "- finding <script>alert(1)</script> and `kept <b> in code`",
            "```bash",
            "curl 'http://x/?a=1&b=<2>'",
            "```",
        ]
    )
    out = _md_file_safe(md).split("\n")
    assert out[0] == "> **Note** &lt;b&gt;bold&lt;/b&gt;"
    assert (
        "<script>" not in out[1] and "&lt;script&gt;" in out[1] and "`kept <b> in code`" in out[1]
    )
    assert out[3] == "curl 'http://x/?a=1&b=<2>'"  # code fence untouched


@pytest.mark.parametrize(
    "target", ["javascript:alert(1)", "JaVaScRiPt:alert(1)", "data:text/html,<x>", "vbscript:x"]
)
def test_html_report_does_not_link_dangerous_schemes(target: str) -> None:
    out = _md_to_html(f"see [click me]({target})")
    assert "<a " not in out and "href" not in out


def test_html_report_keeps_safe_links() -> None:
    out = _md_to_html("[docs](https://example.com/a) and [top](#top)")
    assert 'href="https://example.com/a"' in out and 'href="#top"' in out


def test_suite_config_carries_read_only_methods_from_config(tmp_path: Path) -> None:
    cfg = suite_config(load_target(), "http://127.0.0.1:1", tmp_path, allow_mutations=False)
    assert cfg["read_only_methods"] == ["GET", "HEAD", "OPTIONS"]


def test_seeded_orders_are_capped() -> None:
    assert capped_orders(10) == 10
    assert capped_orders(10**9) == 200_000


# --------------------------------------------------------------------------- policy


def _dependency_specs() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    specs = list(data["project"]["dependencies"])
    for group in data["project"].get("optional-dependencies", {}).values():
        specs += group
    for group in data.get("dependency-groups", {}).values():
        specs += group
    return specs


def test_every_dependency_has_an_upper_bound() -> None:
    uncapped = [s for s in _dependency_specs() if "<" not in s]
    assert not uncapped, f"add an upper bound to: {uncapped}"


def test_compose_publishes_ports_on_loopback_only() -> None:
    compose = yaml.safe_load((ROOT / "deploy/docker-compose.yml").read_text())
    for name, svc in compose["services"].items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} on all interfaces"
    grafana = compose["services"]["grafana"]["environment"]
    assert grafana["GF_AUTH_ANONYMOUS_ORG_ROLE"] != "Admin"


def test_target_image_runs_as_non_root() -> None:
    lines = (ROOT / "deploy/Dockerfile.target").read_text().splitlines()
    users = [ln.split()[1] for ln in lines if ln.startswith("USER ")]
    assert users and users[-1] not in {"root", "0"}


@pytest.mark.parametrize("workflow", sorted((ROOT / ".github/workflows").glob("*.yml")))
def test_workflows_are_least_privilege_and_do_not_interpolate_inputs(workflow: Path) -> None:
    text = workflow.read_text()
    doc = yaml.safe_load(text)
    assert doc["permissions"] == {"contents": "read"}
    for job in doc["jobs"].values():
        for step in job["steps"]:
            run = step.get("run", "")
            assert not re.search(r"\$\{\{\s*(inputs|github\.event)\.", run), (
                f"{workflow.name}: pass inputs through env, not into the shell: {run[:60]!r}"
            )
            if "uv sync" in run:
                assert "--locked" in run


def test_chroma_is_used_embedded_only() -> None:
    """The known chromadb advisories concern its HTTP server; we must never run or call one."""
    offenders = []
    for path in (ROOT / "agentqa").rglob("*.py"):
        text = path.read_text()
        if re.search(r"HttpClient|AsyncHttpClient|trust_remote_code", text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders
