"""Swagger 2.0 documents are converted to OpenAPI 3.0 on load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentqa.ingest.openapi import load_spec, parse_endpoints
from agentqa.ingest.swagger2 import to_openapi3

ROOT = Path(__file__).resolve().parent.parent
TASKS_SWAGGER = ROOT / "examples/tasks_api/swagger.json"

DOC: dict[str, Any] = {
    "swagger": "2.0",
    "info": {"title": "Shop", "version": "3"},
    "host": "api.example.com",
    "basePath": "/v2",
    "schemes": ["https"],
    "consumes": ["application/json"],
    "produces": ["application/json"],
    "parameters": {
        "Limit": {"name": "limit", "in": "query", "type": "integer", "maximum": 50},
        "Body": {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {"$ref": "#/definitions/Item"},
        },
    },
    "securityDefinitions": {
        "basic": {"type": "basic"},
        "oauth": {
            "type": "oauth2",
            "flow": "accessCode",
            "authorizationUrl": "https://a.example/auth",
            "tokenUrl": "https://a.example/token",
            "scopes": {"read": "Read"},
        },
    },
    "paths": {
        "/items": {
            "parameters": [{"$ref": "#/parameters/Limit"}],
            "get": {
                "responses": {
                    "200": {
                        "description": "ok",
                        "schema": {"type": "array", "items": {"$ref": "#/definitions/Item"}},
                    }
                },
                "security": [{"basic": []}],
            },
            "post": {
                "parameters": [{"$ref": "#/parameters/Body"}],
                "responses": {
                    "201": {"description": "created", "schema": {"$ref": "#/definitions/Item"}}
                },
            },
        },
        "/upload": {
            "post": {
                "consumes": ["multipart/form-data"],
                "parameters": [
                    {"name": "file", "in": "formData", "type": "file", "required": True},
                    {"name": "note", "in": "formData", "type": "string"},
                ],
                "responses": {"204": {"description": "stored"}},
            }
        },
    },
    "definitions": {
        "Item": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "x-nullable": True},
                "owner": {"$ref": "#/definitions/Item"},
            },
        }
    },
}


def test_converts_structure() -> None:
    out = to_openapi3(DOC)
    assert out["openapi"] == "3.0.3"
    assert out["servers"] == [{"url": "https://api.example.com/v2"}]
    get = out["paths"]["/items"]["get"]
    assert get["parameters"] == [
        {
            "name": "limit",
            "in": "query",
            "required": False,
            "schema": {"type": "integer", "maximum": 50},
        }
    ]
    schema = get["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["items"] == {"$ref": "#/components/schemas/Item"}
    post = out["paths"]["/items"]["post"]
    assert post["requestBody"]["required"] is True
    assert post["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Item"
    }
    item = out["components"]["schemas"]["Item"]["properties"]
    assert item["name"] == {"type": "string", "nullable": True}
    assert item["owner"] == {"$ref": "#/components/schemas/Item"}
    assert out["components"]["securitySchemes"]["basic"] == {"type": "http", "scheme": "basic"}
    flows = out["components"]["securitySchemes"]["oauth"]["flows"]
    assert flows["authorizationCode"]["tokenUrl"] == "https://a.example/token"


def test_form_data_becomes_request_body() -> None:
    body = to_openapi3(DOC)["paths"]["/upload"]["post"]["requestBody"]
    schema = body["content"]["multipart/form-data"]["schema"]
    assert schema["properties"]["file"] == {"type": "string", "format": "binary"}
    assert schema["required"] == ["file"] and body["required"] is True


def test_does_not_mutate_input() -> None:
    before = json.dumps(DOC, sort_keys=True)
    to_openapi3(DOC)
    assert json.dumps(DOC, sort_keys=True) == before


def test_load_spec_accepts_swagger_and_parses_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "swagger.json"
    path.write_text(json.dumps(DOC), encoding="utf-8")
    eps = {e.id: e for e in parse_endpoints(load_spec(path))}
    assert set(eps) == {"GET /items", "POST /items", "POST /upload"}
    assert eps["GET /items"].auth_schemes == ["basic"]
    assert eps["POST /items"].request_schema


def test_tasks_swagger_parses() -> None:
    eps = {e.id: e for e in parse_endpoints(load_spec(TASKS_SWAGGER))}
    assert len(eps) == 7
    assert not eps["GET /health"].requires_auth
    assert all(e.requires_auth for k, e in eps.items() if k != "GET /health")
    status = next(p for p in eps["GET /tasks"].params if p.name == "status")
    assert status.schema_["enum"] == ["open", "done"]
    assert eps["POST /tasks"].request_schema["required"] == ["title"]


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ({"swagger": "1.2"}, "unsupported Swagger version"),
        ({"openapi": "2.5.0"}, "unsupported OpenAPI version"),
        ({"info": {}}, "missing 'openapi' or 'swagger'"),
    ],
)
def test_rejects_unsupported_documents(tmp_path: Path, doc: dict[str, Any], message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_spec(path)
