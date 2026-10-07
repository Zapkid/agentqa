"""Pre-commit secret scan: fail if a staged/tracked file contains an API key pattern.

Usage: python scripts/secret_scan.py [files...]   (no args = all tracked files)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agentqa.guards.redaction import SECRET_PATTERNS

SKIP_SUFFIXES = {".png", ".gif", ".jpg", ".ico", ".lock", ".sqlite"}
ALLOW_MARKER = "secret-scan: allow"


def scan(paths: list[str]) -> list[str]:
    findings = []
    for p in paths:
        path = Path(p)
        if not path.is_file() or path.suffix in SKIP_SUFFIXES or path.name == ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if ALLOW_MARKER in line:
                continue
            for name, pattern in SECRET_PATTERNS:
                if name in {"kv_secret", "bearer"}:
                    continue  # too noisy for source code; covered by runtime redaction
                if pattern.search(line):
                    findings.append(f"{p}:{lineno}: possible {name}")
    return findings


def main() -> int:
    paths = (
        sys.argv[1:]
        or subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    if (
        Path(".env").exists()
        and subprocess.run(
            ["git", "ls-files", "--error-unmatch", ".env"], capture_output=True
        ).returncode
        == 0
    ):
        print(".env is tracked by git: remove it")
        return 1
    findings = scan(paths)
    for f in findings:
        print(f)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
