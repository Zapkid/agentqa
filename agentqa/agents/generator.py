"""F3 Generator agent: TestIntent -> GeneratedTest (a self-contained pytest function).

The model may call four read-only tools. The tool set is fixed here in code; a request for any
other tool is refused and logged (prompt-injection guard: content can never grant permissions).
"""

from __future__ import annotations

import json
from typing import Any, cast

from agentqa.agents.synthesizer import PLACEHOLDER_ID
from agentqa.guards.injection import delimit, refuse_tool
from agentqa.ingest.vectorstore import VectorStore
from agentqa.llm.prompts import load_prompt
from agentqa.llm.router import LLMClient, tool_result_message
from agentqa.llm.types import Message, ToolSpec
from agentqa.models import (
    Endpoint,
    GeneratedTest,
    GeneratorOutput,
    SpecBundle,
    TestIntent,
)
from agentqa.obs import tracing

FIXTURES_DOC = {
    "client": "httpx.Client bound to the API base URL; use relative paths, e.g. client.get('/orders', headers=...)",
    "auth": "dict role -> headers; roles: admin, staff, customer, other_customer",
    "customer_id": "id of the 'customer' role's customer record",
    "other_customer_id": "id of the 'other_customer' role's customer record",
    "sign_webhook": "sign_webhook(body: bytes) -> str, value for the webhook signature header",
    "webhook_header": "name of the webhook signature header (str)",
    "find_id": "find_id(list_path, headers, limit=1, predicate=None) -> list[str] of ids from a collection",
    "schema_check": "schema_check(data, schema): strict JSON-schema assertion",
}

TOOLS = [
    ToolSpec(
        name="get_endpoint_schema",
        description="Spec details (params, request and response schemas) for an endpoint id like 'POST /orders'.",
        parameters={
            "type": "object",
            "properties": {"endpoint_id": {"type": "string"}},
            "required": ["endpoint_id"],
        },
    ),
    ToolSpec(
        name="search_docs",
        description="Search the customer's requirement documents.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="list_fixtures",
        description="List the pytest fixtures available to tests.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="get_example_payload",
        description="A schema-valid example request body for an endpoint id.",
        parameters={
            "type": "object",
            "properties": {"endpoint_id": {"type": "string"}},
            "required": ["endpoint_id"],
        },
    ),
]
TOOL_NAMES = [t.name for t in TOOLS]


def example_from_schema(schema: dict[str, Any] | None, depth: int = 0) -> Any:
    if not schema or depth > 4:
        return None
    if "anyOf" in schema:
        options = [s for s in schema["anyOf"] if s.get("type") != "null"]
        return example_from_schema(options[0] if options else {}, depth)
    if schema.get("examples"):
        return schema["examples"][0]
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        return {
            k: example_from_schema(v, depth + 1) for k, v in schema.get("properties", {}).items()
        }
    if t == "array":
        return [example_from_schema(schema.get("items", {}), depth + 1)]
    if t == "integer":
        return int(schema.get("minimum", 1))
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if schema.get("format") == "uuid":
        return PLACEHOLDER_ID
    if schema.get("format") == "date-time":
        return "2026-01-01T00:00:00Z"
    return "example"


class Generator:
    max_rounds = 4

    def __init__(self, bundle: SpecBundle, store: VectorStore) -> None:
        self.bundle = bundle
        self.store = store
        self.chunks = {c.id: c for c in bundle.chunks}
        self.prompt = load_prompt("generator")

    # ------------------------------------------------------------------ tools

    def _endpoint(self, endpoint_id: str) -> Endpoint | None:
        try:
            return self.bundle.endpoint(endpoint_id)
        except KeyError:
            return None

    def run_tool(self, name: str, args: dict[str, Any]) -> Any:
        with tracing.span(
            f"tool {name}",
            "tool_call",
            **{"agentqa.tool": name, "input.value": json.dumps(args)[:500]},
        ):
            if name not in TOOL_NAMES:
                return refuse_tool("generator", name, TOOL_NAMES)
            if name == "get_endpoint_schema":
                ep = self._endpoint(str(args.get("endpoint_id", "")))
                if ep is None:
                    return f"unknown endpoint {args.get('endpoint_id')!r}; known: {[e.id for e in self.bundle.endpoints]}"
                return {
                    "summary": self.chunks[f"spec:{ep.id}"].text,
                    "request_schema": ep.request_schema,
                    "responses": sorted(ep.status_codes),
                    "params": [p.model_dump(by_alias=True) for p in ep.params],
                }
            if name == "search_docs":
                hits = self.store.search(
                    self.bundle.spec_id, str(args.get("query", "")), k=3, kind="doc"
                )
                return "\n".join(
                    delimit(cid, self.chunks[cid].text) for cid, _ in hits if cid in self.chunks
                )
            if name == "list_fixtures":
                return FIXTURES_DOC
            ep = self._endpoint(str(args.get("endpoint_id", "")))
            if ep is None:
                return f"unknown endpoint {args.get('endpoint_id')!r}"
            return example_from_schema(ep.request_schema)

    # ------------------------------------------------------------------ generation

    def payload(self, intent: TestIntent) -> dict[str, Any]:
        return {
            "intent": intent.model_dump(exclude={"origin"}),
            "endpoint_spec": self.chunks[f"spec:{intent.endpoint}"].text,
            "all_endpoints": [e.id for e in self.bundle.endpoints],
            "test_name_hint": f"test_{intent.id.replace('-', '_')}"[:80],
        }

    def generate(
        self,
        intent: TestIntent,
        client: LLMClient,
        *,
        feedback: str | None = None,
        lessons: str = "",
        task_id: str | None = None,
    ) -> GeneratedTest:
        messages: list[Message] = self.prompt.render(
            payload=json.dumps(self.payload(intent), indent=1),
            feedback=feedback or "",
            lessons=lessons,
        )
        meta = self.prompt.metadata("generator", task_id=task_id)
        with tracing.span(
            f"agent generator {intent.id}",
            "agent",
            **{"agentqa.agent": "generator", "agentqa.intent": intent.id},
        ):
            for _ in range(self.max_rounds):
                result = client.complete(
                    messages,
                    tools=TOOLS,
                    response_schema=GeneratorOutput,
                    max_tokens=2500,
                    metadata=meta,
                )
                if not result.tool_calls:
                    out = cast(GeneratorOutput, result.parsed)
                    return GeneratedTest(intent_id=intent.id, **out.model_dump())
                assert result.assistant_message is not None
                messages = [*messages, result.assistant_message]
                for call in result.tool_calls:
                    messages.append(
                        tool_result_message(
                            call.id, call.name, self.run_tool(call.name, call.arguments)
                        )
                    )
            # Out of tool rounds: ask for the final answer without tools.
            result = client.complete(
                [*messages, Message(role="user", content="Reply now with the final JSON.")],
                response_schema=GeneratorOutput,
                max_tokens=2500,
                metadata=meta,
            )
            out = cast(GeneratorOutput, result.parsed)
            return GeneratedTest(intent_id=intent.id, **out.model_dump())
