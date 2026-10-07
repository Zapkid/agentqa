from __future__ import annotations

from pathlib import Path

import pytest

from agentqa import config
from agentqa.agents.generator import Generator, example_from_schema
from agentqa.agents.planner import Planner
from agentqa.agents.synthesizer import Synthesizer
from agentqa.guards.grounding import check_test, extract_calls
from agentqa.guards.static_checks import check_code
from agentqa.ingest.ingestor import ingest
from agentqa.ingest.openapi import load_spec, parse_endpoints
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.cache import DiskCache
from agentqa.llm.router import ModelRouter
from agentqa.models import GeneratedTest, RequestDecl, SpecBundle

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "target_api" / "openapi.json"
EPS = parse_endpoints(load_spec(SPEC))


@pytest.fixture(scope="module")
def ingested() -> tuple[SpecBundle, VectorStore]:
    store = VectorStore()
    bundle, _ = ingest(SPEC, ROOT / "target_api" / "docs", store)
    return bundle, store


def _gt(code: str, decls: list[RequestDecl], name: str = "test_x_case") -> GeneratedTest:
    return GeneratedTest(intent_id="i", test_name=name, code=code, requests_made=decls)


def test_t0_synthesizer_covers_spec_and_is_grounded() -> None:
    pairs = Synthesizer(EPS).synthesize()
    kinds = {i.id.split("-")[1] for i, _ in pairs}
    assert {
        "auth",
        "contract",
        "required",
        "type",
        "qenum",
        "qmin",
        "baduuid",
        "pagination",
    } <= kinds
    for intent, test in pairs:
        assert intent.origin == "t0" and intent.source_refs == [f"spec:{intent.endpoint}"]
        assert check_code(test.code, test.test_name) == [], test.test_name
        assert check_test(test, EPS) == [], test.test_name
    # deterministic
    assert [t.code for _, t in Synthesizer(EPS).synthesize()] == [t.code for _, t in pairs]


def test_grounding_catches_invented_endpoint_field_and_status() -> None:
    code = """
def test_x_case(client, auth):
    r = client.post("/v1/orders", json={"line_items": []}, headers=auth["customer"])
    client.get(f"/orders/{1}/shipment")
    assert r.status_code == 201
"""
    v = check_test(
        _gt(
            code,
            [
                RequestDecl(method="POST", path="/orders", fields=["qty"], expected_status=[418]),
                RequestDecl(method="DELETE", path="/orders/{order_id}"),
            ],
        ),
        EPS,
    )
    kinds = {x.kind for x in v}
    assert {"unknown_path", "unknown_field", "undocumented_status", "method_not_allowed"} <= kinds


def test_grounding_flags_undeclared_request_and_accepts_valid() -> None:
    code = """
def test_x_case(client, auth):
    client.get("/products", headers=auth["admin"])
    r = client.post("/orders", json={"items": []}, headers=auth["customer"])
    assert r.status_code == 422
"""
    v = check_test(
        _gt(
            code,
            [RequestDecl(method="POST", path="/orders", fields=["items"], expected_status=[422])],
        ),
        EPS,
    )
    assert [x.kind for x in v] == ["undeclared_request"]
    ok = _gt(
        code,
        [
            RequestDecl(method="GET", path="/products", expected_status=[200]),
            RequestDecl(method="POST", path="/orders", fields=["items"], expected_status=[422]),
        ],
    )
    assert check_test(ok, EPS) == []


def test_extract_calls_fstring_and_request() -> None:
    calls = extract_calls(
        'def t(client):\n    client.request("PATCH", f"/products/{x}", json={"price": "1"})\n'
    )
    assert (
        calls[0].method == "PATCH"
        and calls[0].path == "/products/{param}"
        and calls[0].body_keys == ["price"]
    )


@pytest.mark.parametrize(
    "snippet,needle",
    [
        ("import os\n", "import not allowed"),
        ("open('/etc/passwd')\n", "banned call: open"),
        ("httpx.get('http://evil.example')\n", "direct network call"),
        ("x = ().__class__\n", "dunder"),
        ("eval('1')\n", "banned call: eval"),
        ("key = 'sk-ant-" + "a" * 20 + "'\n", "possible secret"),  # secret-scan: allow
        ("y = undefined_name + 1\n", "ruff"),
    ],
)
def test_static_checks_block(snippet: str, needle: str) -> None:
    code = "def test_x_case(client):\n    " + snippet.replace("\n", "\n    ").rstrip() + "\n"
    if snippet.startswith("import"):
        code = snippet + "def test_x_case(client):\n    pass\n"
    details = " | ".join(v.detail for v in check_code(code, "test_x_case"))
    assert needle in details


def test_static_checks_pass_clean_code() -> None:
    code = "def test_x_case(client, auth):\n    r = client.get('/orders', headers=auth['admin'])\n    assert r.status_code == 200\n"
    assert check_code(code, "test_x_case") == []


def test_planner_simulated_cites_retrieved_chunks(ingested: tuple[SpecBundle, VectorStore]) -> None:
    bundle, store = ingested
    router = ModelRouter("simulated", cache=DiskCache(mode="off"))
    planner = Planner(bundle, store, top_k=5)
    intents, stats = planner.plan(
        bundle.endpoint("POST /orders"), router.for_tier("T2"), config.dispatch().llm_categories
    )
    assert stats.retrieved and all(not r.startswith("doc:poisoned") for r in stats.retrieved)
    assert intents, "strong simulated planner should find the POST /orders rules"
    for it in intents:
        assert set(it.source_refs) <= {"spec:POST /orders", *stats.retrieved}
        assert 1 <= it.risk <= 5 and it.origin == "llm"


def test_generator_tool_loop_and_refusal(ingested: tuple[SpecBundle, VectorStore]) -> None:
    bundle, store = ingested
    gen = Generator(bundle, store)
    assert "not available" in str(gen.run_tool("delete_everything", {}))
    assert "request_schema" in gen.run_tool("get_endpoint_schema", {"endpoint_id": "POST /orders"})
    assert "client" in gen.run_tool("list_fixtures", {})
    ex = gen.run_tool("get_example_payload", {"endpoint_id": "POST /orders"})
    assert isinstance(ex, dict) and "items" in ex
    router = ModelRouter("simulated", cache=DiskCache(mode="off"))
    planner = Planner(bundle, store, top_k=5)
    intents, _ = planner.plan(
        bundle.endpoint("PATCH /products/{product_id}"),
        router.for_tier("T2"),
        config.dispatch().llm_categories,
    )
    test = gen.generate(intents[0], router.for_tier("T2"))
    assert test.intent_id == intents[0].id and test.code.startswith("def test_")
    agents = [e.agent for e in router.ledger.entries]
    assert agents.count("generator") == 2  # tool round + final answer


def test_example_from_schema() -> None:
    assert example_from_schema(
        {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 3}, "s": {"enum": ["a", "b"]}},
        }
    ) == {"n": 3, "s": "a"}
