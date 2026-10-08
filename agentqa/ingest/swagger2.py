"""Swagger 2.0 -> OpenAPI 3.0 conversion.

Covers what the ingestor and generators read: paths, parameters (including ``in: body`` and
``in: formData``), responses, ``definitions``, ``securityDefinitions`` and ``x-nullable``. It is a
conversion for testing purposes, not a general-purpose converter; see docs/adr/0009.
"""

from __future__ import annotations

import copy
from typing import Any

HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
# Keys that sit directly on a Swagger 2.0 non-body parameter and belong in an OpenAPI 3 schema.
SCHEMA_KEYS = (
    "type",
    "format",
    "enum",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
)
REF_PREFIXES = {
    "#/definitions/": "#/components/schemas/",
    "#/responses/": "#/components/responses/",
}
FLOWS = {
    "implicit": "implicit",
    "password": "password",
    "application": "clientCredentials",
    "accessCode": "authorizationCode",
}


def _rewrite(node: Any) -> Any:
    """Rewrite ``$ref`` targets and ``x-nullable`` throughout a schema or response tree."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                for old, new in REF_PREFIXES.items():
                    if value.startswith(old):
                        value = new + value[len(old) :]
                out[key] = value
            elif key == "x-nullable":
                out["nullable"] = bool(value)
            else:
                out[key] = _rewrite(value)
        return out
    if isinstance(node, list):
        return [_rewrite(v) for v in node]
    return node


def _schema_of(param: dict[str, Any]) -> dict[str, Any]:
    schema = {k: param[k] for k in SCHEMA_KEYS if k in param}
    if schema.get("type") == "file":
        schema = {"type": "string", "format": "binary"}
    if param.get("x-nullable"):
        schema["nullable"] = True
    return _rewrite(schema)


def _convert_params(
    params: list[dict[str, Any]], consumes: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Split Swagger parameters into OpenAPI parameters and an optional request body."""
    converted: list[dict[str, Any]] = []
    body: dict[str, Any] | None = None
    form: dict[str, Any] = {"properties": {}, "required": []}
    for p in params:
        if p.get("in") == "body":
            schema = _rewrite(p.get("schema", {}))
            body = {
                "required": bool(p.get("required")),
                "content": {mime: {"schema": schema} for mime in consumes},
            }
        elif p.get("in") == "formData":
            form["properties"][p["name"]] = _schema_of(p)
            if p.get("required"):
                form["required"].append(p["name"])
        else:
            item: dict[str, Any] = {
                "name": p["name"],
                "in": p["in"],
                "required": bool(p.get("required")) or p["in"] == "path",
                "schema": _schema_of(p),
            }
            if p.get("description"):
                item["description"] = p["description"]
            converted.append(item)
    if form["properties"] and body is None:
        schema = {"type": "object", "properties": form["properties"]}
        if form["required"]:
            schema["required"] = form["required"]
        mime = next((m for m in consumes if "form" in m or "multipart" in m), None)
        body = {
            "required": bool(form["required"]),
            "content": {mime or "application/x-www-form-urlencoded": {"schema": schema}},
        }
    return converted, body


def _security_schemes(defs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, d in defs.items():
        if d.get("type") == "basic":
            out[name] = {"type": "http", "scheme": "basic"}
        elif d.get("type") == "oauth2":
            flow = FLOWS.get(d.get("flow", ""), "implicit")
            body = {k: d[k] for k in ("authorizationUrl", "tokenUrl") if k in d}
            out[name] = {"type": "oauth2", "flows": {flow: {**body, "scopes": d.get("scopes", {})}}}
        else:
            out[name] = {k: v for k, v in d.items() if k in ("type", "in", "name", "description")}
    return out


def to_openapi3(doc: dict[str, Any]) -> dict[str, Any]:
    """Return an OpenAPI 3.0.3 document equivalent to the Swagger 2.0 ``doc``."""
    doc = copy.deepcopy(doc)
    global_params: dict[str, Any] = doc.get("parameters", {})
    consumes_default: list[str] = doc.get("consumes") or ["application/json"]
    produces_default: list[str] = doc.get("produces") or ["application/json"]

    def resolve(params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for p in params:
            ref = p.get("$ref", "")
            out.append(
                global_params[ref.rsplit("/", 1)[-1]] if ref.startswith("#/parameters/") else p
            )
        return out

    def convert_responses(responses: dict[str, Any], produces: list[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for code, resp in responses.items():
            if "$ref" in resp:
                out[str(code)] = _rewrite(resp)
                continue
            item: dict[str, Any] = {"description": resp.get("description", "")}
            if "schema" in resp:
                schema = _rewrite(resp["schema"])
                item["content"] = {mime: {"schema": schema} for mime in produces}
            out[str(code)] = item
        return out

    paths: dict[str, Any] = {}
    for path, item in doc.get("paths", {}).items():
        shared = resolve(item.get("parameters", []))
        new_item: dict[str, Any] = {}
        for method in HTTP_METHODS:
            op = item.get(method)
            if op is None:
                continue
            consumes = op.get("consumes") or consumes_default
            params, body = _convert_params([*shared, *resolve(op.get("parameters", []))], consumes)
            new_op = {
                k: v
                for k, v in op.items()
                if k not in ("parameters", "responses", "consumes", "produces")
            }
            if params:
                new_op["parameters"] = params
            if body is not None:
                new_op["requestBody"] = body
            new_op["responses"] = convert_responses(
                op.get("responses", {}), op.get("produces") or produces_default
            )
            new_item[method] = new_op
        paths[path] = new_item

    components: dict[str, Any] = {}
    if doc.get("definitions"):
        components["schemas"] = _rewrite(doc["definitions"])
    if doc.get("responses"):
        components["responses"] = convert_responses(doc["responses"], produces_default)
    if doc.get("securityDefinitions"):
        components["securitySchemes"] = _security_schemes(doc["securityDefinitions"])

    out: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": doc.get("info", {"title": "API", "version": ""}),
        "paths": paths,
    }
    host, base_path = doc.get("host"), doc.get("basePath", "/")
    if host:
        scheme = (doc.get("schemes") or ["https"])[0]
        out["servers"] = [{"url": f"{scheme}://{host}{base_path}".rstrip("/") or "/"}]
    elif base_path not in ("", "/"):
        out["servers"] = [{"url": base_path}]
    if components:
        out["components"] = components
    for key in ("security", "tags", "externalDocs"):
        if key in doc:
            out[key] = doc[key]
    return out
