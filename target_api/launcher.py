"""Start a target build (clean / single-bug / all-bugs) as a local uvicorn subprocess.

Used by the executor, the differential eval and the performance runner. No Docker needed.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import TracebackType

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class TargetServer:
    def __init__(
        self,
        bugs: list[str] | tuple[str, ...] = (),
        perf_bugs: list[str] | tuple[str, ...] = (),
        port: int | None = None,
        workdir: Path | None = None,
        extra_env: dict[str, str] | None = None,
        cpu_limit_s: int | None = None,
        cpus: set[int] | None = None,
        code_root: Path | None = None,
        app: str = "target_api.app.main:app",
    ) -> None:
        self.bugs = list(bugs)
        self.perf_bugs = list(perf_bugs)
        self.port = port or free_port()
        self.workdir = workdir or Path(tempfile.mkdtemp(prefix="agentqa_target_"))
        self.log_path = self.workdir / "server.log"
        self.extra_env = extra_env or {}
        self.cpu_limit_s = cpu_limit_s
        # pin to these cores (stand-in for a container CPU limit; repeatable perf runs)
        self.cpus = cpus
        # another checkout (e.g. main) to run the baseline build from, for PR A/B checks
        self.code_root = code_root or REPO_ROOT
        self.app = app  # module:attribute of the ASGI app, e.g. examples.tasks_api.app:app
        self.proc: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def label(self) -> str:
        parts = self.bugs + self.perf_bugs
        return "clean" if not parts else "+".join(parts)

    def start(self, timeout_s: float = 20.0) -> TargetServer:
        env = {
            **os.environ,
            "BUGS": ",".join(self.bugs),
            "PERF_BUGS": ",".join(self.perf_bugs),
            "TARGET_DB": str(self.workdir / "orders.db"),
            "TARGET_LOG": str(self.log_path),
            "PYTHONPATH": str(self.code_root),
            **self.extra_env,
        }
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            self.app,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--log-level",
            "warning",
            "--no-access-log",
        ]
        self._stderr = open(self.workdir / "stderr.log", "wb")  # noqa: SIM115 - closed in stop()
        cpus = self.cpus

        def pin() -> None:
            if cpus:
                os.sched_setaffinity(0, cpus)

        self.proc = subprocess.Popen(  # noqa: S603 - uvicorn with fixed arguments
            cmd,
            env=env,
            cwd=self.code_root,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            preexec_fn=pin,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"target exited early: {(self.workdir / 'stderr.log').read_text()[-2000:]}"
                )
            try:
                if httpx.get(f"{self.base_url}/health", timeout=0.5).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        self.stop()
        raise TimeoutError(f"target did not become healthy on {self.base_url}")

    def reset(self) -> None:
        httpx.post(f"{self.base_url}/__reset", timeout=30)

    def seed(self, orders: int) -> int:
        r = httpx.post(f"{self.base_url}/__seed", params={"orders": orders}, timeout=300)
        r.raise_for_status()
        total: int = r.json()["orders"]
        return total

    def metrics(self) -> dict[str, object]:
        data: dict[str, object] = httpx.get(f"{self.base_url}/__metrics", timeout=10).json()
        return data

    def log_lines(self, since_ts: float = 0.0) -> list[str]:
        import json

        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("ts", 0) >= since_ts:
                    out.append(line)
            except ValueError:
                continue
        return out

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if getattr(self, "_stderr", None) is not None:
            self._stderr.close()

    def __enter__(self) -> TargetServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
