"""Static checks on generated test code (before anything runs).

What it stops: generated code that imports something unexpected, opens files, spawns processes,
opens its own network connections (bypassing the sandboxed ``client`` fixture), uses
introspection tricks, embeds a secret, or simply does not parse / has undefined names.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

from agentqa import config
from agentqa.guards.redaction import find_secrets
from agentqa.models import GuardViolation


def check_code(code: str, test_name: str) -> list[GuardViolation]:
    cfg = config.guardrails().sandbox
    allowed_imports = set(cfg.allowed_imports)
    banned_names = {c for c in cfg.banned_calls if "." not in c}
    banned_modules = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "importlib",
        "ctypes",
        "builtins",
    }
    out: list[GuardViolation] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            GuardViolation(kind="static", detail=f"syntax error: {exc.msg} (line {exc.lineno})")
        ]

    tests = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    if [t.name for t in tests] != [test_name]:
        out.append(
            GuardViolation(
                kind="static", detail=f"expected exactly one test function named {test_name}"
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for m in mods:
                if m.split(".")[0] not in allowed_imports:
                    out.append(GuardViolation(kind="static", detail=f"import not allowed: {m}"))
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in banned_names:
                out.append(GuardViolation(kind="static", detail=f"banned call: {fn.id}()"))
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                base = fn.value.id
                if base in banned_modules:
                    out.append(
                        GuardViolation(kind="static", detail=f"banned call: {base}.{fn.attr}()")
                    )
                if base == "httpx":
                    out.append(
                        GuardViolation(
                            kind="static",
                            detail=f"direct network call httpx.{fn.attr}(); use the client fixture",
                        )
                    )
        elif (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            out.append(
                GuardViolation(kind="static", detail=f"dunder attribute access: {node.attr}")
            )
        elif (
            isinstance(node, ast.Name)
            and node.id in banned_modules
            and isinstance(node.ctx, ast.Load)
        ):
            out.append(GuardViolation(kind="static", detail=f"use of module {node.id}"))
    for secret in find_secrets(code):
        out.append(GuardViolation(kind="static", detail=f"possible secret in code: {secret}"))
    out.extend(_pyflakes(code))
    return out


def _pyflakes(code: str) -> list[GuardViolation]:
    """ruff (pyflakes rules) for undefined names and similar errors."""
    prelude = (
        "import datetime, decimal, hashlib, hmac, json, math, re, time, uuid\nimport pytest\n"
        "from datetime import timedelta, timezone\nfrom decimal import ROUND_HALF_UP, Decimal\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.py"
        path.write_text(prelude + code, encoding="utf-8")
        proc = subprocess.run(  # noqa: S603 - ruff with fixed arguments on a temp file
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--no-cache",
                "--isolated",
                "--select",
                "F821,F822,F823,F701,F702,F704,F706",
                "--output-format",
                "concise",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if proc.returncode not in (0, 1):  # 2 = ruff itself failed: never skip the check silently
        return [
            GuardViolation(kind="static", detail=f"ruff could not run: {proc.stderr.strip()[:200]}")
        ]
    issues = [ln.split(":", 3)[-1].strip() for ln in proc.stdout.splitlines() if str(path) in ln]
    return [GuardViolation(kind="static", detail=f"ruff: {i}") for i in issues]
