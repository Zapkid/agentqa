"""Tier 0: spec-derived test synthesizer (deterministic, 0 tokens).

Everything that follows mechanically from the OpenAPI document is generated here instead of by
a model, in the spirit of Schemathesis/Hypothesis-style property checks:

- missing-auth: a protected operation called without credentials returns 401
- contract-conformance: a GET response matches its documented schema, with no undocumented fields
- validation: a body missing a required field, or with a wrongly-typed field, is a 4xx
- boundary: enum and min/max violations on query parameters are a 4xx; pagination is consistent
- error-handling: a malformed ``format: uuid`` path parameter is a 4xx, never a 5xx

Generated code uses only the suite fixtures (see executor/conftest_template.py).
"""

from __future__ import annotations

import re
from typing import Any

from agentqa.agents.risk import rule_risk
from agentqa.models import Endpoint, GeneratedTest, RequestDecl, TestIntent

PLACEHOLDER_ID = "00000000-0000-4000-8000-000000000000"
CLIENT_ERRORS = (400, 404, 422)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def _path_params(ep: Endpoint) -> list[str]:
    return re.findall(r"{([^}]+)}", ep.path)


def _fill(path: str, value: str) -> str:
    return re.sub(r"{[^}]+}", value, path)


def _fstring_path(path: str) -> str:
    return 'f"' + path + '"'


def _documented_client_errors(ep: Endpoint) -> list[int]:
    return [c for c in CLIENT_ERRORS if str(c) in ep.status_codes] or [422]


def _list_path_for(ep: Endpoint, endpoints: dict[str, Endpoint]) -> str | None:
    """For /orders/{order_id}/..., the collection GET /orders that yields order ids."""
    params = _path_params(ep)
    if len(params) != 1:
        return None
    prefix = ep.path.split("/{", 1)[0]
    return prefix if f"GET {prefix}" in endpoints else None


def _intent(
    ep: Endpoint, kind: str, category: str, title: str, expected: str, rationale: str
) -> TestIntent:
    return TestIntent(
        id=f"t0-{kind}-{_slug(ep.id)}"[:90],
        endpoint=ep.id,
        category=category,  # type: ignore[arg-type]
        title=title,
        risk=rule_risk(ep, category),
        rationale=rationale,
        expected_behavior=expected,
        source_refs=[f"spec:{ep.id}"],
        origin="t0",
    )


def _test(intent: TestIntent, name: str, code: str, decls: list[RequestDecl]) -> GeneratedTest:
    return GeneratedTest(
        intent_id=intent.id,
        test_name=f"test_t0_{name}"[:84],
        code=code.strip() + "\n",
        requests_made=decls,
        confidence=1.0,
    )


def _body_arg(ep: Endpoint) -> str:
    return ", json={}" if ep.request_schema is not None else ""


class Synthesizer:
    def __init__(self, endpoints: list[Endpoint], privileged_role: str = "admin") -> None:
        self.endpoints = {e.id: e for e in endpoints}
        self.role = privileged_role

    def synthesize(self) -> list[tuple[TestIntent, GeneratedTest]]:
        out: list[tuple[TestIntent, GeneratedTest]] = []
        for ep in self.endpoints.values():
            for maker in (
                self.missing_auth,
                self.contract,
                self.body_validation,
                self.query_boundaries,
                self.malformed_uuid,
                self.pagination,
            ):
                out.extend(maker(ep))
        return out

    # -------------------------------------------------------------- generators

    def missing_auth(self, ep: Endpoint) -> list[tuple[TestIntent, GeneratedTest]]:
        if not ep.requires_auth or "401" not in ep.status_codes:
            return []
        path = _fill(ep.path, PLACEHOLDER_ID)
        intent = _intent(
            ep,
            "auth",
            "auth/permission",
            f"{ep.id} requires credentials",
            "401 when called without credentials",
            "Spec declares a security requirement and documents 401.",
        )
        name = f"auth_required_{_slug(ep.id)}"
        code = f"""
def test_t0_{name}(client):
    r = client.request({ep.method!r}, {path!r}{_body_arg(ep)})
    assert r.status_code == 401, f"expected 401 without credentials, got {{r.status_code}}: {{r.text[:200]}}"
"""
        return [
            (
                intent,
                _test(
                    intent,
                    name,
                    code,
                    [RequestDecl(method=ep.method, path=ep.path, expected_status=[401])],
                ),
            )
        ]

    def contract(self, ep: Endpoint) -> list[tuple[TestIntent, GeneratedTest]]:
        schema = ep.responses.get("200")
        if ep.method != "GET" or not schema:
            return []
        params = _path_params(ep)
        headers = f'auth["{self.role}"]' if ep.requires_auth else "{}"
        intent = _intent(
            ep,
            "contract",
            "contract-conformance",
            f"{ep.id} response matches schema",
            "200 body validates against the documented schema with no undocumented fields",
            "Contract drift and undocumented fields break clients and can leak data.",
        )
        name = f"contract_{_slug(ep.id)}"
        if params:
            list_path = _list_path_for(ep, self.endpoints)
            if list_path is None:
                return []
            var = params[0]
            code = f"""
def test_t0_{name}(client, auth, find_id, schema_check):
    {var} = None
    for candidate in find_id({list_path!r}, {headers}, limit=10):
        r = client.get({_fstring_path(ep.path.replace("{" + var + "}", "{candidate}"))}, headers={headers})
        if r.status_code == 200:
            {var} = candidate
            break
    if {var} is None:
        pytest.skip("no instance with a 200 response")
    schema_check(r.json(), SCHEMA_{name.upper()})


SCHEMA_{name.upper()} = {schema!r}
"""
            decls = [
                RequestDecl(method="GET", path=list_path, expected_status=[200]),
                RequestDecl(method="GET", path=ep.path, expected_status=[200]),
            ]
        else:
            code = f"""
def test_t0_{name}(client, auth, schema_check):
    r = client.get({ep.path!r}, headers={headers})
    assert r.status_code == 200, r.text[:300]
    schema_check(r.json(), SCHEMA_{name.upper()})


SCHEMA_{name.upper()} = {schema!r}
"""
            decls = [RequestDecl(method="GET", path=ep.path, expected_status=[200])]
        return [(intent, _test(intent, name, code, decls))]

    def _target_path(self, ep: Endpoint) -> tuple[str, str, list[RequestDecl]] | None:
        """(setup code, path expression, extra decls) to reach an existing instance."""
        params = _path_params(ep)
        if not params:
            return "", repr(ep.path), []
        list_path = _list_path_for(ep, self.endpoints)
        if list_path is None:
            return None
        var = params[0]
        setup = (
            f'    {var} = next(iter(find_id({list_path!r}, auth["{self.role}"])), None)\n'
            f'    if {var} is None:\n        pytest.skip("no instance available")\n'
        )
        return (
            setup,
            _fstring_path(ep.path),
            [RequestDecl(method="GET", path=list_path, expected_status=[200])],
        )

    def body_validation(self, ep: Endpoint) -> list[tuple[TestIntent, GeneratedTest]]:
        schema = ep.request_schema
        if not schema or ep.method not in ("POST", "PUT", "PATCH") or "properties" not in schema:
            return []
        if not ep.requires_auth and "401" in ep.status_codes:
            return []  # authenticated by a mechanism the spec does not describe (e.g. a signature)
        reach = self._target_path(ep)
        if reach is None:
            return []
        setup, path_expr, pre = reach
        headers = f'auth["{self.role}"]' if ep.requires_auth else "{}"
        statuses = _documented_client_errors(ep)
        out = []
        required = schema.get("required", [])
        if required:
            intent = _intent(
                ep,
                "required",
                "validation",
                f"{ep.id} rejects a body missing required fields",
                f"4xx ({'/'.join(map(str, statuses))}) when required field(s) {required} are missing",
                "Required fields are declared in the request schema.",
            )
            name = f"missing_required_{_slug(ep.id)}"
            code = f"""
def test_t0_{name}(client, auth, find_id):
{setup}    r = client.request({ep.method!r}, {path_expr}, json={{}}, headers={headers})
    assert r.status_code in {tuple(statuses)!r}, f"expected client error, got {{r.status_code}}: {{r.text[:200]}}"
"""
            out.append(
                (
                    intent,
                    _test(
                        intent,
                        name,
                        code,
                        [
                            *pre,
                            RequestDecl(
                                method=ep.method, path=ep.path, fields=[], expected_status=statuses
                            ),
                        ],
                    ),
                )
            )
        wrong = _wrong_type_field(schema)
        if wrong:
            field, bad_value = wrong
            intent = _intent(
                ep,
                "type",
                "validation",
                f"{ep.id} rejects a wrongly typed '{field}'",
                f"4xx when '{field}' has the wrong type",
                "Field types are declared in the request schema.",
            )
            name = f"wrong_type_{_slug(ep.id)}_{_slug(field)}"
            body = {field: bad_value}
            code = f"""
def test_t0_{name}(client, auth, find_id):
{setup}    r = client.request({ep.method!r}, {path_expr}, json={body!r}, headers={headers})
    assert r.status_code in {tuple(statuses)!r}, f"expected client error, got {{r.status_code}}: {{r.text[:200]}}"
"""
            out.append(
                (
                    intent,
                    _test(
                        intent,
                        name,
                        code,
                        [
                            *pre,
                            RequestDecl(
                                method=ep.method,
                                path=ep.path,
                                fields=[field],
                                expected_status=statuses,
                            ),
                        ],
                    ),
                )
            )
        return out

    def query_boundaries(self, ep: Endpoint) -> list[tuple[TestIntent, GeneratedTest]]:
        out = []
        reach = self._target_path(ep) if ep.method == "GET" else None
        if reach is None:
            return []
        setup, path_expr, pre = reach
        headers = f'auth["{self.role}"]' if ep.requires_auth else "{}"
        statuses = _documented_client_errors(ep)
        for p in ep.params:
            if p.location != "query":
                continue
            s = _flatten(p.schema_)
            cases: list[tuple[str, Any, str]] = []
            if "enum" in s:
                cases.append(("enum", "zz_not_a_valid_value", f"value outside enum {s['enum']}"))
            if s.get("type") == "integer" and "minimum" in s:
                cases.append(("min", int(s["minimum"]) - 1, f"value below minimum {s['minimum']}"))
            if s.get("type") == "integer" and "maximum" in s:
                cases.append(("max", int(s["maximum"]) + 1, f"value above maximum {s['maximum']}"))
            for kind, value, why in cases:
                category = "contract-conformance" if kind == "enum" else "boundary"
                intent = _intent(
                    ep,
                    f"q{kind}-{_slug(p.name)}",
                    category,
                    f"{ep.id} rejects {p.name} {why}",
                    f"4xx when query parameter {p.name} has a {why}",
                    "Parameter constraints are declared in the spec.",
                )
                name = f"query_{kind}_{_slug(ep.id)}_{_slug(p.name)}"
                code = f"""
def test_t0_{name}(client, auth, find_id):
{setup}    r = client.get({path_expr}, params={{{p.name!r}: {value!r}}}, headers={headers})
    assert r.status_code in {tuple(statuses)!r}, f"expected client error, got {{r.status_code}}: {{r.text[:200]}}"
"""
                out.append(
                    (
                        intent,
                        _test(
                            intent,
                            name,
                            code,
                            [
                                *pre,
                                RequestDecl(
                                    method="GET",
                                    path=ep.path,
                                    fields=[p.name],
                                    expected_status=statuses,
                                ),
                            ],
                        ),
                    )
                )
        return out

    def malformed_uuid(self, ep: Endpoint) -> list[tuple[TestIntent, GeneratedTest]]:
        uuid_params = [
            p
            for p in ep.params
            if p.location == "path" and _flatten(p.schema_).get("format") == "uuid"
        ]
        if len(uuid_params) != 1:
            return []
        statuses = [c for c in (400, 404, 422) if str(c) in ep.status_codes] or [404, 422]
        headers = f'auth["{self.role}"]' if ep.requires_auth else "{}"
        path = _fill(ep.path, "not-a-uuid")
        intent = _intent(
            ep,
            "baduuid",
            "error-handling",
            f"{ep.id} handles a malformed id",
            f"{'/'.join(map(str, statuses))} (never 5xx) for a malformed UUID path parameter",
            "The path parameter is declared with format uuid.",
        )
        name = f"malformed_uuid_{_slug(ep.id)}"
        code = f"""
def test_t0_{name}(client, auth):
    r = client.request({ep.method!r}, {path!r}{_body_arg(ep)}, headers={headers})
    assert r.status_code < 500, f"server error for malformed id: {{r.status_code}}"
    assert r.status_code in {tuple(statuses)!r}, f"unexpected status {{r.status_code}}"
"""
        return [
            (
                intent,
                _test(
                    intent,
                    name,
                    code,
                    [RequestDecl(method=ep.method, path=ep.path, expected_status=statuses)],
                ),
            )
        ]

    def pagination(self, ep: Endpoint) -> list[tuple[TestIntent, GeneratedTest]]:
        if ep.method != "GET" or _path_params(ep):
            return []
        names = {p.name: _flatten(p.schema_) for p in ep.params if p.location == "query"}
        size_param = next(
            (n for n in ("page_size", "per_page", "limit", "size") if n in names), None
        )
        if "page" not in names or size_param is None:
            return []
        maximum = int(names[size_param].get("maximum", 100))
        big = min(30, maximum)
        headers = f'auth["{self.role}"]' if ep.requires_auth else "{}"
        intent = _intent(
            ep,
            "pagination",
            "boundary",
            f"{ep.id} pages are contiguous",
            "page 2 returns exactly the items that follow page 1, nothing skipped or repeated",
            "The operation is paginated with page and page size parameters.",
        )
        name = f"pagination_{_slug(ep.id)}"
        code = f"""
def test_t0_{name}(client, auth):
    def items(body):
        return body["items"] if isinstance(body, dict) else body

    full = items(client.get({ep.path!r}, params={{"page": 1, {size_param!r}: {big}}}, headers={headers}).json())
    if len(full) < 20:
        pytest.skip("not enough data to compare pages")
    page2 = items(client.get({ep.path!r}, params={{"page": 2, {size_param!r}: 10}}, headers={headers}).json())
    assert [i.get("id") for i in page2] == [i.get("id") for i in full[10:20]], "page 2 is not items 11-20"
"""
        return [
            (
                intent,
                _test(
                    intent,
                    name,
                    code,
                    [
                        RequestDecl(
                            method="GET",
                            path=ep.path,
                            fields=["page", size_param],
                            expected_status=[200],
                        )
                    ],
                ),
            )
        ]


def _flatten(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse ``anyOf: [X, {type: null}]`` (OpenAPI 3.1 optional) to X."""
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            return {**non_null[0], **{k: v for k, v in schema.items() if k != "anyOf"}}
    return schema


def _wrong_type_field(schema: dict[str, Any]) -> tuple[str, Any] | None:
    for name, prop in schema.get("properties", {}).items():
        t = _flatten(prop).get("type")
        bad = {
            "string": 12345,
            "integer": "not-a-number",
            "number": "not-a-number",
            "boolean": "maybe",
            "array": "not-a-list",
            "object": "not-an-object",
        }.get(str(t))
        if bad is not None:
            return name, bad
    return None
