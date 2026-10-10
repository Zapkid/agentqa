"""veroniqa: the official command-line client for VeroniQA's public API.

Standard library only. Errors from the API are RFC 9457 problem documents; they are printed with
their `code` and `hint` and the command exits with status 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

__version__ = "0.1.0"
DEFAULT_URL = "https://veroniqa.vercel.app"
USER_AGENT = f"veroniqa-cli/{__version__}"

EXIT_OK, EXIT_API, EXIT_USAGE, EXIT_NETWORK = 0, 1, 2, 3


class ApiError(Exception):
    def __init__(self, status: int, problem: dict[str, Any]) -> None:
        super().__init__(problem.get("detail") or f"HTTP {status}")
        self.status = status
        self.problem = problem


class NetworkError(Exception):
    pass


Transport = Callable[[str, str, bytes | None, dict[str, str], float], tuple[int, bytes]]


def urllib_transport(
    method: str, url: str, body: bytes | None, headers: dict[str, str], timeout: float
) -> tuple[int, bytes]:
    if not url.startswith(("https://", "http://")):
        raise NetworkError(f"unsupported URL: {url}")
    req = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310 (scheme checked above)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (scheme checked above)
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise NetworkError(str(getattr(err, "reason", err))) from err


class Client:
    def __init__(
        self, base_url: str = DEFAULT_URL, timeout: float = 30.0, transport: Transport | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport or urllib_transport

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        status, raw = self.transport(method, self.base_url + path, body, headers, self.timeout)
        try:
            data = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            data = None
        if status >= 400:
            problem = (
                data if isinstance(data, dict) else {"detail": raw[:200].decode(errors="replace")}
            )
            raise ApiError(status, problem)
        if data is None:
            raise ApiError(status, {"detail": "the response was not JSON"})
        return data

    def status(self) -> Any:
        return self.request("GET", "/api/v1/status")

    def results(self) -> Any:
        return self.request("GET", "/api/v1/results")

    def ask(self, question: str) -> Any:
        return self.request("POST", "/api/v1/ask", {"question": question})

    def openapi(self) -> Any:
        return self.request("GET", "/openapi.json")

    def demo_spec(self) -> Any:
        return self.request("GET", "/api/v1/demo/openapi.json")


def _print_human(command: str, data: Any, out: Any) -> None:
    if command == "status":
        print(f"{data['service']} {data['version']}: {data['status']}", file=out)
        print(f"models: {data['models']} · authentication: {data['authentication']}", file=out)
        print(f"docs: {data['docs']}", file=out)
    elif command == "results":
        print(f"From the {data['benchmark']}:", file=out)
        for tile in data["tiles"]:
            print(f"  {tile['value']:>6}  {tile['caption']}", file=out)
    elif command == "ask":
        print(data["answer"], file=out)
        for c in data["citations"]:
            print(f"  [{c['n']}] {c['source']} · {c['section']}", file=out)
        if data.get("note"):
            print(f"note: {data['note']}", file=out)
    else:
        print(json.dumps(data, indent=2), file=out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="veroniqa",
        description="Command-line client for VeroniQA's public API (no key needed).",
        epilog="API guide: https://veroniqa.vercel.app/developers",
    )
    parser.add_argument("--version", action="version", version=f"veroniqa {__version__}")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VERONIQA_URL", DEFAULT_URL),
        help=f"deployment to call (default: $VERONIQA_URL or {DEFAULT_URL})",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds (default 30)")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    # --json also works after the command (`veroniqa results --json`); SUPPRESS keeps the
    # subcommand from resetting a value given before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="print raw JSON"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    sub.add_parser("status", parents=[common], help="service status and version")
    sub.add_parser("results", parents=[common], help="the measured results shown on the home page")
    ask = sub.add_parser(
        "ask", parents=[common], help="ask about the demo API; answers cite their sources"
    )
    ask.add_argument("question", nargs="+", help="the question")
    sub.add_parser("openapi", parents=[common], help="print the public API's OpenAPI description")
    sub.add_parser(
        "demo-spec", parents=[common], help="print the OpenAPI spec of the bundled demo API"
    )
    return parser


def main(argv: Sequence[str] | None = None, transport: Transport | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on usage errors and 0 for --help/--version
        return int(exc.code or 0)
    client = Client(args.base_url, args.timeout, transport)
    calls: dict[str, Callable[[], Any]] = {
        "status": client.status,
        "results": client.results,
        "ask": lambda: client.ask(" ".join(args.question)),
        "openapi": client.openapi,
        "demo-spec": client.demo_spec,
    }
    try:
        data = calls[args.command]()
    except ApiError as err:
        p = err.problem
        code = f" [{p['code']}]" if p.get("code") else ""
        print(f"error {err.status}{code}: {p.get('detail', err)}", file=sys.stderr)
        if p.get("hint"):
            print(f"hint: {p['hint']}", file=sys.stderr)
        for field in p.get("errors") or []:
            print(f"  {field.get('field')}: {field.get('message')}", file=sys.stderr)
        return EXIT_API
    except NetworkError as err:
        print(f"network error: {err} ({args.base_url})", file=sys.stderr)
        return EXIT_NETWORK
    if args.json or args.command in ("openapi", "demo-spec"):
        print(json.dumps(data, indent=2))
    else:
        _print_human(args.command, data, sys.stdout)
    return EXIT_OK
