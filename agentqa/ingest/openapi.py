"""OpenAPI 3.0 / 3.1 parser: spec file or URL -> normalised Endpoint objects.

``$ref``s are resolved inline (with a cycle guard). 3.1 ``anyOf: [X, {type: null}]`` and 3.0
``nullable`` are both understood by the downstream schema checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import yaml

from agentqa.models import Endpoint, Param

HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")


def load_spec(source: str | Path) -> dict[str, Any]:
    text: str
    src = str(source)
    if src.startswith(("http://", "https://")):
        resp = httpx.get(src, timeout=30)
        resp.raise_for_status()
        text = resp.text
    else:
        text = Path(src).read_text(encoding="utf-8")
    data = json.loads(text) if text.lstrip().startswith("{") else yaml.safe_load(text)
    if not isinstance(data, dict) or "openapi" not in data:
        raise ValueError("not an OpenAPI document (missing 'openapi' key)")
    major_minor = str(data["openapi"])[:3]
    if major_minor not in ("3.0", "3.1"):
        raise ValueError(f"unsupported OpenAPI version {data['openapi']}")
    return data


def resolve_refs(node: Any, root: dict[str, Any], seen: tuple[str, ...] = ()) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            ref = node["$ref"]
            if ref in seen:
                return {"type": "object", "description": f"(recursive {ref})"}
            if not ref.startswith("#/"):
                raise ValueError(f"external $ref not supported: {ref}")
            target: Any = root
            for part in ref[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
            merged = {k: v for k, v in node.items() if k != "$ref"}
            resolved = resolve_refs(copy.deepcopy(target), root, (*seen, ref))
            return {**resolved, **merged} if merged else resolved
        return {k: resolve_refs(v, root, seen) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_refs(v, root, seen) for v in node]
    return node


def _json_schema_of(content: dict[str, Any] | None) -> dict[str, Any] | None:
    if not content:
        return None
    for media in ("application/json", *content.keys()):
        if media in content and "schema" in content[media]:
            schema: dict[str, Any] = content[media]["schema"]
            return schema
    return None


def parse_endpoints(spec: dict[str, Any]) -> list[Endpoint]:
    spec = resolve_refs(spec, spec)
    global_security = spec.get("security", [])
    endpoints: list[Endpoint] = []
    for path, item in sorted(spec.get("paths", {}).items()):
        shared_params = item.get("parameters", [])
        for method in HTTP_METHODS:
            op = item.get(method)
            if op is None:
                continue
            params = []
            for p in [*shared_params, *op.get("parameters", [])]:
                params.append(
                    Param(
                        name=p["name"],
                        location=p["in"],
                        required=bool(p.get("required")),
                        schema=p.get("schema", {}),
                        description=p.get("description", ""),
                    )
                )
            body = op.get("requestBody")
            security = op.get("security", global_security)
            schemes = sorted({name for req in security or [] for name in req})
            responses = {
                str(code): _json_schema_of((resp or {}).get("content"))
                for code, resp in op.get("responses", {}).items()
            }
            endpoints.append(
                Endpoint(
                    id=f"{method.upper()} {path}",
                    method=method.upper(),
                    path=path,
                    operation_id=op.get("operationId", ""),
                    summary=op.get("summary", ""),
                    description=op.get("description", ""),
                    tags=op.get("tags", []),
                    params=params,
                    request_schema=_json_schema_of((body or {}).get("content")),
                    responses=responses,
                    requires_auth=bool(schemes),
                    auth_schemes=schemes,
                )
            )
    return endpoints


def endpoint_hash(ep: Endpoint) -> str:
    return hashlib.sha256(ep.model_dump_json().encode()).hexdigest()[:16]


def spec_id(spec: dict[str, Any]) -> str:
    blob = json.dumps(spec, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def match_path(concrete: str, endpoints: list[Endpoint]) -> list[Endpoint]:
    """Endpoints whose path template matches a concrete or placeholder path."""
    concrete = concrete.split("?", 1)[0].rstrip("/") or "/"
    c_parts = concrete.split("/")
    out = []
    for ep in endpoints:
        t_parts = ep.path.split("/")
        if len(t_parts) != len(c_parts):
            continue
        if all(
            t == c
            or (t.startswith("{") and t.endswith("}"))
            or (c.startswith("{") and c.endswith("}"))
            for t, c in zip(t_parts, c_parts, strict=True)
        ):
            out.append(ep)
    return out
