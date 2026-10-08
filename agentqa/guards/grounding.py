"""F4 Grounding guard: the hallucination detector.

Before a generated test runs, every request it *declares* (``requests_made``) and every request
its code *actually makes* (found in the AST) is checked against the spec: the path exists, the
method is allowed on it, the body/query fields exist, and the expected status codes are
documented. Violations go back to the generator once for repair; a test that is still invalid
is quarantined. ``hallucination_rate`` = tests with violations before repair / tests checked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from agentqa.ingest.openapi import match_path
from agentqa.models import Endpoint, GeneratedTest, GuardViolation, RequestDecl
from agentqa.obs import metrics, tracing

HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}


@dataclass
class ActualCall:
    method: str
    path: str  # template with {placeholders} for f-string parts
    body_keys: list[str]
    query_keys: list[str]
    line: int


def _path_from(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                parts.append("{param}")
        return "".join(parts)
    return None


def _dict_keys(node: ast.expr | None) -> list[str]:
    if isinstance(node, ast.Dict):
        return [
            k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
        ]
    return []


def extract_calls(code: str) -> list[ActualCall]:
    """Requests made through the ``client`` fixture: client.get(path, ...), client.request(M, path, ...)."""
    calls: list[ActualCall] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return calls
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "client"
        ):
            continue
        attr = node.func.attr.lower()
        args = list(node.args)
        if attr == "request" and len(args) >= 2 and isinstance(args[0], ast.Constant):
            method, path_node = str(args[0].value).upper(), args[1]
        elif attr in HTTP_VERBS and args:
            method, path_node = attr.upper(), args[0]
        else:
            continue
        path = _path_from(path_node)
        if path is None:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        calls.append(
            ActualCall(
                method,
                path.split("?", 1)[0],
                _dict_keys(kw.get("json")),
                _dict_keys(kw.get("params")),
                node.lineno,
            )
        )
    return calls


def _field_names(ep: Endpoint) -> set[str]:
    return set(ep.body_fields()) | {p.name for p in ep.params}


def check_decl(decl: RequestDecl, endpoints: list[Endpoint]) -> list[GuardViolation]:
    matches = match_path(decl.path, endpoints)
    if not matches:
        return [GuardViolation(kind="unknown_path", detail=f"{decl.path} is not in the spec")]
    eps = [e for e in matches if e.method == decl.method.upper()]
    if not eps:
        allowed = sorted({e.method for e in matches})
        return [
            GuardViolation(
                kind="method_not_allowed", detail=f"{decl.method} {decl.path} (allowed: {allowed})"
            )
        ]
    ep = eps[0]
    out = []
    known = _field_names(ep)
    for f in decl.fields:
        if f.split(".")[0] not in known:
            out.append(
                GuardViolation(kind="unknown_field", detail=f"{ep.id}: field {f!r} not in spec")
            )
    for code in decl.expected_status:
        if str(code) not in ep.status_codes:
            out.append(
                GuardViolation(
                    kind="undocumented_status",
                    detail=f"{ep.id}: status {code} not documented ({sorted(ep.status_codes)})",
                )
            )
    return out


def check_test(test: GeneratedTest, endpoints: list[Endpoint]) -> list[GuardViolation]:
    out: list[GuardViolation] = []
    declared: set[tuple[str, str]] = set()
    for decl in test.requests_made:
        out.extend(check_decl(decl, endpoints))
        for ep in match_path(decl.path, endpoints):
            if ep.method == decl.method.upper():
                declared.add((ep.method, ep.path))
    for call in extract_calls(test.code):
        matches = match_path(call.path, endpoints)
        if not matches:
            out.append(
                GuardViolation(
                    kind="unknown_path",
                    detail=f"line {call.line}: {call.method} {call.path} not in spec",
                )
            )
            continue
        eps = [e for e in matches if e.method == call.method]
        if not eps:
            out.append(
                GuardViolation(
                    kind="method_not_allowed", detail=f"line {call.line}: {call.method} {call.path}"
                )
            )
            continue
        ep = eps[0]
        known = _field_names(ep)
        for key in call.body_keys + call.query_keys:
            if key not in known:
                out.append(
                    GuardViolation(
                        kind="unknown_field",
                        detail=f"line {call.line}: {ep.id} has no field {key!r}",
                    )
                )
        if (ep.method, ep.path) not in declared:
            out.append(
                GuardViolation(
                    kind="undeclared_request",
                    detail=f"line {call.line}: {ep.id} is called but not declared in requests_made",
                )
            )
    # de-duplicate while keeping order
    seen: set[tuple[str, str]] = set()
    unique = []
    for v in out:
        if (v.kind, v.detail) not in seen:
            seen.add((v.kind, v.detail))
            unique.append(v)
    return unique


def record(stage: str, test_name: str, violations: list[GuardViolation]) -> None:
    if violations:
        metrics.inc("agentqa_guardrail_events_total", guardrail="grounding", action=stage)
        tracing.event(
            "guardrail.grounding",
            test=test_name,
            action=stage,
            violations=[f"{v.kind}: {v.detail}" for v in violations[:5]],
        )
