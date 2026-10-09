"""Links to API documentation: OpenAPI/Swagger files and the Swagger UI pages that display them."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml

from agentqa.veroniqa import VeroniQA, apidocs, create_project
from agentqa.veroniqa.fetch import FetchError, fetch_url
from agentqa.veroniqa.knowledge import KnowledgeBase

# The page Swagger UI's distribution serves (as at petstore3.swagger.io): no text of its own.
SWAGGER_UI = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Swagger UI</title>
<link rel="stylesheet" type="text/css" href="./swagger-ui.css" /></head>
<body><div id="swagger-ui"></div>
<script src="./swagger-ui-bundle.js" charset="UTF-8"> </script>
<script src="./swagger-ui-standalone-preset.js" charset="UTF-8"> </script>
<script src="./swagger-initializer.js" charset="UTF-8"> </script>
</body></html>"""

INITIALIZER = """window.onload = function() {
  window.ui = SwaggerUIBundle({
    url: "/api/v3/openapi.json",
    dom_id: '#swagger-ui',
    deepLinking: true,
    validatorUrl: "https://validator.swagger.io/validator",
    layout: "StandaloneLayout"
  });
};"""

SPEC = {
    "openapi": "3.0.4",
    "info": {"title": "Swagger Petstore - OpenAPI 3.0", "version": "1.0.27"},
    "servers": [{"url": "/api/v3"}],
    "paths": {
        "/pet/{petId}": {
            "get": {
                "tags": ["pet"],
                "summary": "Find pet by ID.",
                "description": "Returns a single pet.",
                "operationId": "getPetById",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "description": "ID of pet to return",
                        "required": True,
                        "schema": {"type": "integer", "format": "int64"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "successful operation",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                        },
                    },
                    "404": {"description": "Pet not found"},
                },
                "security": [{"api_key": []}, {"petstore_auth": ["read:pets"]}],
            }
        },
        "/pet": {
            "post": {
                "summary": "Add a new pet to the store.",
                "operationId": "addPet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                    },
                },
                "responses": {"200": {"description": "Successful operation"}},
            }
        },
    },
    "components": {
        "schemas": {
            "Pet": {
                "required": ["name", "photoUrls"],
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "format": "int64"},
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "description": "pet status in the store",
                        "enum": ["available", "pending", "sold"],
                    },
                },
            }
        },
        "securitySchemes": {
            "petstore_auth": {"type": "oauth2", "flows": {}},
            "api_key": {"type": "apiKey", "name": "api_key", "in": "header"},
        },
    },
}


def public(host: str, port: int) -> list[str]:
    return ["169.254.169.254"] if host == "metadata.internal" else ["93.184.216.34"]


def _client(routes: dict[str, tuple[str, str]], seen: list[str] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url.copy_with(path=request.url.path or "/"))
        if seen is not None:
            seen.append(url)
        found = routes.get(url)
        if found is None:
            return httpx.Response(404, text="not found")
        ctype, body = found
        return httpx.Response(200, headers={"content-type": ctype}, text=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def petstore_site(initializer: str = INITIALIZER) -> httpx.Client:
    return _client(
        {
            "https://petstore3.swagger.io/": ("text/html", SWAGGER_UI),
            "https://petstore3.swagger.io/swagger-initializer.js": (
                "application/javascript",
                initializer,
            ),
            "https://petstore3.swagger.io/api/v3/openapi.json": (
                "application/json",
                json.dumps(SPEC),
            ),
        }
    )


def test_swagger_ui_page_is_followed_to_its_spec() -> None:
    page = fetch_url("https://petstore3.swagger.io/", client=petstore_site(), resolver=public)
    assert page.spec_url == "https://petstore3.swagger.io/api/v3/openapi.json"
    assert page.title == "Swagger Petstore - OpenAPI 3.0"
    text = page.text
    assert "## GET /pet/{petId}: Find pet by ID." in text
    assert "`petId` (in path, integer (int64), required): ID of pet to return" in text
    assert "- 200: successful operation (returns Pet)" in text and "- 404: Pet not found" in text
    assert "Request body (application/json): Pet" in text and "`name` (string, required)" in text
    assert "one of: available, pending, sold" in text
    assert "https://petstore3.swagger.io/api/v3" in text  # the server, made absolute
    assert "**api_key**: apiKey header `api_key`" in text
    assert "Requires authentication: api_key, petstore_auth." in text


def test_direct_spec_links_json_and_yaml() -> None:
    client = _client(
        {
            "https://api.example.com/openapi.json": ("application/json", json.dumps(SPEC)),
            "https://api.example.com/openapi.yaml": ("text/plain", yaml.safe_dump(SPEC)),
        }
    )
    for url in ("https://api.example.com/openapi.json", "https://api.example.com/openapi.yaml"):
        page = fetch_url(url, client=client, resolver=public)
        assert page.spec_url == url and "## POST /pet: Add a new pet to the store." in page.text


def test_spec_found_at_a_usual_location_when_the_page_names_none() -> None:
    client = _client(
        {
            "https://docs.example.com/": (
                "text/html",
                SWAGGER_UI.replace("swagger-initializer", "x"),
            ),
            "https://docs.example.com/openapi.json": ("application/json", json.dumps(SPEC)),
        }
    )
    page = fetch_url("https://docs.example.com/", client=client, resolver=public)
    assert page.spec_url == "https://docs.example.com/openapi.json"


def test_docs_page_cannot_lead_to_a_private_address() -> None:
    evil = INITIALIZER.replace("/api/v3/openapi.json", "http://metadata.internal/openapi.json")
    seen: list[str] = []
    client = _client(
        {
            "https://docs.example.com/": ("text/html", SWAGGER_UI),
            "https://docs.example.com/swagger-initializer.js": ("text/javascript", evil),
        },
        seen,
    )
    with pytest.raises(FetchError, match="no readable text"):
        fetch_url("https://docs.example.com/", client=client, resolver=public)
    assert not any("metadata.internal" in url for url in seen)  # refused before any request
    assert len(seen) <= 2 + 8  # the page, its script, and at most 8 spec probes


def test_bare_host_link_as_typed() -> None:
    page = fetch_url("https://petstore3.swagger.io", client=petstore_site(), resolver=public)
    assert page.spec_url == "https://petstore3.swagger.io/api/v3/openapi.json"
    assert apidocs.candidate_spec_urls("https://host.example")[0] == (
        "https://host.example/openapi.json"
    )


def test_javascript_page_without_a_spec_explains_what_to_link() -> None:
    client = _client({"https://app.example.com/": ("text/html", "<div id='root'></div>")})
    with pytest.raises(FetchError, match="OpenAPI or Swagger file"):
        fetch_url("https://app.example.com/", client=client, resolver=public)


def test_swagger2_spec_is_described() -> None:
    doc = {
        "swagger": "2.0",
        "info": {"title": "Old Petstore", "version": "1.0"},
        "host": "petstore.swagger.io",
        "basePath": "/v2",
        "schemes": ["https"],
        "paths": {
            "/pet": {
                "post": {
                    "summary": "Add a new pet",
                    "parameters": [
                        {"in": "body", "name": "body", "schema": {"$ref": "#/definitions/Pet"}},
                        {"in": "header", "name": "X-Trace", "type": "string"},
                    ],
                    "responses": {"405": {"description": "Invalid input"}},
                }
            }
        },
        "definitions": {"Pet": {"properties": {"name": {"type": "string"}}}},
    }
    title, text = apidocs.spec_to_markdown(doc, "https://petstore.swagger.io/v2/swagger.json")
    assert title == "Old Petstore" and "https://petstore.swagger.io/v2" in text
    assert "Request body (application/json): Pet" in text
    assert "`X-Trace` (in header, string)" in text and "- 405: Invalid input" in text
    assert "### Pet" in text


def test_not_a_spec() -> None:
    assert apidocs.parse_spec("<html></html>") is None
    assert apidocs.parse_spec('{"name": "x"}') is None
    assert apidocs.parse_spec("just: [unbalanced") is None


def test_linking_swagger_ui_indexes_endpoints_and_sets_the_project_spec(tmp_path: Path) -> None:
    project = create_project("Petstore", root=tmp_path)
    fetcher = lambda url: fetch_url(url, client=petstore_site(), resolver=public)  # noqa: E731
    v = VeroniQA(project, profile="simulated", fetcher=fetcher)
    reply = v.chat("add https://petstore3.swagger.io to the knowledge")
    assert "Swagger Petstore" in reply.text and "now this project's spec" in reply.text
    assert project.config.spec == "https://petstore3.swagger.io/api/v3/openapi.json"
    hits = KnowledgeBase(project).retrieve("find pet by ID petId")
    assert any("GET /pet/{petId}" in h.text for h in hits)
    assert "pet" in v.chat("How do I find a pet by its ID?").text.lower()
