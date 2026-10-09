"""Capture real Veroniqa screenshots and run facts for promo video 03.

Uses a throwaway AGENTQA_HOME, the simulated models and the real pipeline, then drives the real
Streamlit app in Chromium. Needs Playwright (not a project dependency):

    uv run --with playwright==1.55.0 python scripts/videos/capture_veroniqa.py

Set PLAYWRIGHT_CHROMIUM to a Chromium binary if Playwright's own browser is not installed.
Writes media/veroniqa/{knowledge,chat,ssrf,runs}.png and media/veroniqa/run.json.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "media/veroniqa"
HOME = Path(tempfile.mkdtemp(prefix="veroniqa-capture-"))
os.environ.update(
    {"AGENTQA_HOME": str(HOME), "AGENTQA_CACHE_MODE": "off", "AGENTQA_PROFILE": "simulated"}
)
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from agentqa.veroniqa import Veroniqa, create_demo_project  # noqa: E402

BUGS = ["B01", "B04"]
QUESTION = "What is the p95 latency target for the order list?"
UNSAFE = "add http://169.254.169.254/latest to the knowledge"
SCALE = 2  # device pixel ratio: crisp screenshots in a 1080p video


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def run_facts() -> None:
    project = create_demo_project()
    reply = Veroniqa(project).chat(f"run the tests with {' and '.join(BUGS)}")
    report = json.loads((project.runs_dir / str(reply.run_id) / "report.json").read_text())
    outcomes: dict[str, int] = {}
    for r in report["results"]:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    facts = {
        "run_id": reply.run_id,
        "seeded_bugs": BUGS,
        "outcomes": outcomes,
        "product_bugs": [
            f["title"] for f in report["findings"] if f["classification"] == "product_bug"
        ],
        "tokens": report["cost"]["total_tokens"],
        "simulated": report["simulated"],
        "sources": [{"name": s.name, "quarantined": s.quarantined} for s in project.sources()],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    print("run:", facts["outcomes"], facts["product_bugs"])


def screenshots() -> None:
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "agentqa/veroniqa/app.py"),
         "--server.port", str(port), "--server.headless", "true"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                if httpx.get(f"{url}/_stcore/health", timeout=1).text == "ok":
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        boxes: dict[str, list[float]] = {}

        def mark(name: str, locator: Any, pad: float = 10) -> None:
            b = locator.bounding_box()
            if b:
                boxes[name] = [
                    b["x"] - pad,
                    b["y"] - pad,
                    b["width"] + 2 * pad,
                    b["height"] + 2 * pad,
                ]

        with sync_playwright() as p:
            exe = os.environ.get("PLAYWRIGHT_CHROMIUM") or (
                "/opt/pw-browsers/chromium" if Path("/opt/pw-browsers/chromium").exists() else None
            )
            browser = p.chromium.launch(executable_path=exe)
            page = browser.new_page(
                viewport={"width": 1440, "height": 900}, device_scale_factor=SCALE
            )
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector('h2:has-text("Orders API demo")', timeout=60_000)
            page.get_by_role("tab", name="Knowledge").click()
            time.sleep(2)
            mark("knowledge", page.locator('p:has-text("poisoned.md")').last)
            page.screenshot(path=str(OUT / "knowledge.png"))
            page.get_by_role("tab", name="Chat").click()
            box = page.get_by_placeholder("Ask about the API")
            box.fill(QUESTION)
            box.press("Enter")
            page.wait_for_selector("text=300 ms", timeout=60_000)
            page.get_by_text("Sources (").first.click()
            time.sleep(1.5)
            mark("chat", page.locator('[data-testid="stChatMessage"]').nth(1))
            page.screenshot(path=str(OUT / "chat.png"))
            box.fill(UNSAFE)
            box.press("Enter")
            page.wait_for_selector("text=non-public", timeout=60_000)
            time.sleep(1)
            page.mouse.wheel(0, 2000)
            time.sleep(1)
            mark("ssrf", page.locator('[data-testid="stChatMessage"]').last)
            page.screenshot(path=str(OUT / "ssrf.png"))
            page.get_by_role("tab", name="Test runs").click()
            time.sleep(3)
            mark("runs", page.locator('[data-testid="stDataFrame"]').first)
            page.screenshot(path=str(OUT / "runs.png"))
            browser.close()
        facts = json.loads((OUT / "run.json").read_text())
        facts["screenshot_scale"] = SCALE
        facts["highlights"] = boxes  # CSS pixels in the 1440x900 viewport
        (OUT / "run.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    finally:
        server.terminate()
        server.wait(timeout=30)
    print("screenshots:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    run_facts()
    screenshots()
