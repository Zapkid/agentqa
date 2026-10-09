"""API documentation links: recognise OpenAPI/Swagger specs and the pages that display them.

Swagger UI and ReDoc pages are JavaScript apps: their HTML has no readable text, because the
page fetches the spec and renders it in the browser. So a link to one is followed to the spec
it loads, and the spec is turned into Markdown the knowledge base can index: one section per
endpoint (parameters, request body, responses), plus authentication and the data models.

Rendering is deliberately shallow ($refs are named, not expanded, below the first level), so a
large or hostile spec cannot blow up into an enormous document.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin, urlsplit

import yaml

HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
MAX_ENDPOINTS = 400
MAX_SCHEMAS = 200

# Where documentation pages usually find their spec, relative to the page.
COMMON_SPEC_PATHS = (
    "openapi.json",
    "swagger.json",
    "api/v3/openapi.json",
    "v3/api-docs",
    "openapi.yaml",
    "v2/swagger.json",
)

_UI_MARKERS = ("swagger-ui", "swaggeruibundle", "redoc", "rapi-doc", "stoplight")
_SPEC_URL = re.compile(
    r"""(?:\burl|spec-url|specUrl|spec_url)\s*[:=]\s*["']([^"'\s]+\.(?:json|ya?ml)[^"'\s]*|[^"'\s]*(?:api-docs|openapi|swagger)[^"'\s]*)["']""",
    re.IGNORECASE,
)
_INITIALIZER = re.compile(r"""<script[^>]+src=["']([^"']*swagger-initializer\.js)["']""", re.I)


def parse_spec(text: str) -> dict[str, Any] | None:
    """The spec as a dict if the text is an OpenAPI 3 or Swagger 2 document, else None."""
    stripped = text.lstrip()
    if not stripped or stripped[0] == "<":
        return None
    try:
        data = json.loads(stripped) if stripped[0] in "{[" else yaml.safe_load(stripped)
    except (ValueError, yaml.YAMLError):
        return None
    if isinstance(data, dict) and ("openapi" in data or "swagger" in data) and "paths" in data:
        return data
    return None


def looks_like_docs_ui(html: str) -> bool:
    low = html.lower()
    return any(marker in low for marker in _UI_MARKERS)


def spec_urls_in(text: str, base: str) -> list[str]:
    """Spec addresses a docs page (or its initializer script) points at, made absolute."""
    found = [urljoin(base, m.group(1)) for m in _SPEC_URL.finditer(text)]
    return list(dict.fromkeys(found))


def initializer_url(html: str, base: str) -> str | None:
    match = _INITIALIZER.search(html)
    return urljoin(base, match.group(1)) if match else None


def candidate_spec_urls(page_url: str) -> list[str]:
    """The usual spec locations next to a page ("https://host" counts as "https://host/")."""
    base = page_url if urlsplit(page_url).path else page_url + "/"
    return [urljoin(base, path) for path in COMMON_SPEC_PATHS]


# ------------------------------------------------------------------ spec -> Markdown


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _resolve(node: Any, spec: dict[str, Any]) -> Any:
    """Resolve one local $ref (one level only)."""
    if isinstance(node, dict) and isinstance(node.get("$ref"), str):
        ref = node["$ref"]
        if ref.startswith("#/"):
            target: Any = spec
            try:
                for part in ref[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
            except (KeyError, TypeError):
                return node
            return target
    return node


def _type(schema: Any, spec: dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return ""
    if isinstance(schema.get("$ref"), str):
        return _ref_name(schema["$ref"])
    kind = schema.get("type", "")
    if isinstance(kind, list):
        kind = " or ".join(str(k) for k in kind)
    if kind == "array":
        return f"array of {_type(schema.get('items'), spec) or 'items'}"
    for combo in ("oneOf", "anyOf", "allOf"):
        if isinstance(schema.get(combo), list):
            parts = [_type(s, spec) for s in schema[combo]]
            return f"{combo}({', '.join(p for p in parts if p)})"
    if schema.get("enum"):
        values = ", ".join(str(v) for v in schema["enum"][:12])
        return f"{kind or 'value'} (one of: {values})"
    fmt = schema.get("format")
    return f"{kind} ({fmt})" if kind and fmt else str(kind)


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def _body_schema(content: Any) -> tuple[str, Any] | None:
    if not isinstance(content, dict):
        return None
    for media in ("application/json", *content.keys()):
        entry = content.get(media)
        if isinstance(entry, dict) and "schema" in entry:
            return media, entry["schema"]
    return None


def _fields(schema: Any, spec: dict[str, Any]) -> list[str]:
    schema = _resolve(schema, spec)
    if not isinstance(schema, dict):
        return []
    required = set(schema.get("required") or [])
    lines = []
    for name, prop in list((schema.get("properties") or {}).items())[:60]:
        prop_r = _resolve(prop, spec) if isinstance(prop, dict) else {}
        desc = _one_line(prop_r.get("description") if isinstance(prop_r, dict) else "")
        req = ", required" if name in required else ""
        lines.append(f"  - `{name}` ({_type(prop, spec)}{req}){': ' + desc if desc else ''}")
    return lines


def _swagger2_parts(spec: dict[str, Any], op: dict[str, Any]) -> tuple[list[Any], Any]:
    """Swagger 2.0: parameters without the body, and the body schema (if any)."""
    params, body = [], None
    for p in op.get("parameters", []):
        p = _resolve(p, spec)
        if isinstance(p, dict) and p.get("in") == "body":
            body = p.get("schema")
        elif isinstance(p, dict) and p.get("in") == "formData":
            params.append({**p, "in": "form"})
        else:
            params.append(p)
    return params, body


def spec_to_markdown(spec: dict[str, Any], source_url: str = "") -> tuple[str, str]:
    """(title, Markdown) describing an OpenAPI 3 / Swagger 2 spec for the knowledge base."""
    raw_info = spec.get("info")
    info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
    title = _one_line(info.get("title")) or "API specification"
    version = _one_line(info.get("version"))
    is_v2 = "swagger" in spec
    out = [f"# {title}" + (f" (version {version})" if version else "")]
    if info.get("description"):
        out += ["", str(info["description"]).strip()]
    flavour = f"Swagger {spec.get('swagger')}" if is_v2 else f"OpenAPI {spec.get('openapi')}"
    out += [
        "",
        f"This is the API's {flavour} specification"
        + (f", from {source_url}." if source_url else "."),
    ]

    servers = []
    if is_v2 and spec.get("host"):
        scheme = (spec.get("schemes") or ["https"])[0]
        servers.append(f"{scheme}://{spec['host']}{spec.get('basePath', '')}")
    for server in spec.get("servers") or []:
        if isinstance(server, dict) and server.get("url"):
            servers.append(urljoin(source_url, str(server["url"])) if source_url else server["url"])
    if servers:
        out += ["", "## Servers", "", *[f"- {s}" for s in servers]]

    schemes = (
        spec.get("securityDefinitions")
        if is_v2
        else (spec.get("components") or {}).get("securitySchemes")
    ) or {}
    if isinstance(schemes, dict) and schemes:
        out += ["", "## Authentication", ""]
        for name, scheme in schemes.items():
            if not isinstance(scheme, dict):
                continue
            kind = scheme.get("type", "")
            detail = scheme.get("scheme") or scheme.get("in") or scheme.get("flow") or ""
            where = f" `{scheme['name']}`" if scheme.get("name") else ""
            desc = _one_line(scheme.get("description"))
            out.append(
                f"- **{name}**: {kind} {detail}{where}{'. ' + desc if desc else ''}".rstrip()
            )

    count = 0
    raw_paths = spec.get("paths")
    paths: dict[str, Any] = raw_paths if isinstance(raw_paths, dict) else {}
    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        shared = item.get("parameters", [])
        for method in HTTP_METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            count += 1
            if count > MAX_ENDPOINTS:
                break
            summary = _one_line(op.get("summary"))
            out += ["", f"## {method.upper()} {path}" + (f": {summary}" if summary else "")]
            if op.get("description"):
                out += ["", _one_line(op["description"])]
            facts = []
            if op.get("operationId"):
                facts.append(f"Operation id: `{op['operationId']}`.")
            if op.get("tags"):
                facts.append("Tags: " + ", ".join(str(t) for t in op["tags"]) + ".")
            if op.get("deprecated"):
                facts.append("Deprecated.")
            security = op.get("security", spec.get("security"))
            if security:
                names = sorted({n for req in security if isinstance(req, dict) for n in req})
                facts.append("Requires authentication: " + ", ".join(names) + ".")
            elif security == []:
                facts.append("No authentication required.")
            if facts:
                out += ["", " ".join(facts)]

            if is_v2:
                params, body = _swagger2_parts(
                    spec, {"parameters": [*shared, *op.get("parameters", [])]}
                )
                body_media = (op.get("consumes") or spec.get("consumes") or ["application/json"])[0]
            else:
                params = [_resolve(p, spec) for p in [*shared, *op.get("parameters", [])]]
                request = _resolve(op.get("requestBody"), spec)
                found = _body_schema(request.get("content")) if isinstance(request, dict) else None
                body_media, body = found if found else ("", None)
            if params:
                out += ["", "Parameters:"]
                for p in params:
                    if not isinstance(p, dict) or "name" not in p:
                        continue
                    schema = p.get("schema", p)  # Swagger 2 puts the type on the parameter
                    req = ", required" if p.get("required") else ""
                    desc = _one_line(p.get("description"))
                    out.append(
                        f"- `{p['name']}` (in {p.get('in', '?')}, {_type(schema, spec)}{req})"
                        + (f": {desc}" if desc else "")
                    )
            if body is not None:
                out += ["", f"Request body ({body_media}): {_type(body, spec) or 'object'}"]
                out += _fields(body, spec)
            responses = op.get("responses") if isinstance(op.get("responses"), dict) else {}
            if responses:
                out += ["", "Responses:"]
                for code, resp in responses.items():
                    resp = _resolve(resp, spec)
                    if not isinstance(resp, dict):
                        continue
                    desc = _one_line(resp.get("description"))
                    schema = (
                        resp.get("schema")
                        if is_v2
                        else (_body_schema(resp.get("content")) or ("", None))[1]
                    )
                    shape = _type(schema, spec) if schema is not None else ""
                    out.append(
                        f"- {code}"
                        + (f": {desc}" if desc else "")
                        + (f" (returns {shape})" if shape else "")
                    )
    if count > MAX_ENDPOINTS:
        out += ["", f"(Only the first {MAX_ENDPOINTS} operations are listed.)"]

    models = spec.get("definitions") if is_v2 else (spec.get("components") or {}).get("schemas")
    if isinstance(models, dict) and models:
        out += ["", "## Data models"]
        for name, schema in list(models.items())[:MAX_SCHEMAS]:
            if not isinstance(schema, dict):
                continue
            desc = _one_line(schema.get("description"))
            out += ["", f"### {name}" + (f": {desc}" if desc else "")]
            fields = _fields(schema, spec)
            out += fields if fields else [f"  - {_type(schema, spec) or 'object'}"]
    return title, "\n".join(out).strip() + "\n"
