"""The public API at /api, its OpenAPI description, API catalog, developer portal and CLI."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from starlette.applications import Starlette
from starlette.testclient import TestClient

from agentqa.veroniqa import devportal, public_api, server, site

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "clients/veroniqa-cli/src"))
import veroniqa_cli  # noqa: E402

PROBLEM = "application/problem+json"
QUESTION = "What is the p95 latency target for the order list?"


@pytest.fixture(autouse=True)
def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("AGENTQA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AGENTQA_CACHE_MODE", "off")
    public_api.LIMITS.hits.clear()
    yield
    public_api.LIMITS.hits.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(
        Starlette(routes=server.app._user_routes, exception_handlers={404: server.not_found})
    )


def _is_problem(r, status: int, code: str) -> dict:  # type: ignore[no-untyped-def]
    assert r.status_code == status, r.text
    assert r.headers["content-type"] == PROBLEM
    body = r.json()
    assert body["status"] == status and body["code"] == code
    assert body["title"] and body["detail"] and body["hint"]
    assert body["type"] == f"{site.site_url()}/developers#error-{code}"
    return body


# ------------------------------------------------------------------ OpenAPI


@pytest.mark.parametrize("path", ["/openapi.json", "/api/openapi.json"])
def test_openapi_json_is_published(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    assert r.headers["access-control-allow-origin"] == "*"
    doc = r.json()
    assert doc["openapi"].startswith("3.1")
    assert doc["info"]["title"] and doc["info"]["version"] == public_api.API_VERSION
    assert [s["url"] for s in doc["servers"]] == [f"{site.site_url()}/api"]
    assert set(doc["paths"]) >= {"/v1/status", "/v1/results", "/v1/ask", "/v1/demo/openapi.json"}
    ops = [op for item in doc["paths"].values() for op in item.values()]
    assert all(op.get("operationId") and op.get("summary") for op in ops)
    assert len({op["operationId"] for op in ops}) == len(ops)


def test_openapi_documents_problem_errors_and_resolves_every_ref(client: TestClient) -> None:
    doc = client.get("/openapi.json").json()
    schemas = doc["components"]["schemas"]
    assert {"code", "hint", "status", "title", "type"} <= set(schemas["Problem"]["properties"])
    assert "HTTPValidationError" not in schemas
    ask = doc["paths"]["/v1/ask"]["post"]["responses"]
    for status in ("422", "429"):
        assert PROBLEM in ask[status]["content"], status
    refs = re.findall(r'"\$ref": "#/components/schemas/([^"]+)"', json.dumps(doc))
    assert refs and all(ref in schemas for ref in refs)


@pytest.mark.parametrize("path", ["/openapi.yaml", "/api/openapi.yaml"])
def test_openapi_yaml_matches_json(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/yaml")
    assert yaml.safe_load(r.text) == client.get("/openapi.json").json()


# ------------------------------------------------------------------ endpoints


def test_status_results_and_demo_spec(client: TestClient) -> None:
    status = client.get("/api/v1/status")
    assert status.status_code == 200 and status.json()["status"] == "ok"
    assert status.json()["models"] == "simulated"
    results = client.get("/api/v1/results").json()
    assert results["recall_pct"] == site.STATS["recall_pct"]
    assert [(t["value"], t["caption"]) for t in results["tiles"]] == [
        (v, c) for _, v, c in site.stat_tiles()
    ]
    spec = client.get("/api/v1/demo/openapi.json")
    assert spec.status_code == 200 and spec.json() == json.loads(public_api.DEMO_SPEC.read_text())
    index = client.get("/api/")
    assert index.status_code == 200 and index.json()["openapi"].endswith("/openapi.json")
    moved = client.get("/api", follow_redirects=False)
    assert moved.status_code == 308 and moved.headers["location"] == "/api/"


def test_ask_returns_a_cited_answer(client: TestClient) -> None:
    r = client.post("/api/v1/ask", json={"question": QUESTION})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found"] and "300 ms" in body["answer"]
    assert body["citations"] and all(c["source"] for c in body["citations"])
    assert r.headers["cache-control"] == "no-store"


# ------------------------------------------------------------------ errors


def test_unknown_path_and_wrong_method_are_problems(client: TestClient) -> None:
    body = _is_problem(client.get("/api/v1/nope"), 404, "not_found")
    assert body["instance"] == "/api/v1/nope"
    r = client.delete("/api/v1/status")
    _is_problem(r, 405, "method_not_allowed")
    assert r.headers["allow"] == "GET"


@pytest.mark.parametrize(
    ("payload", "field"),
    [({}, "question"), ({"question": "x"}, "question"), ({"question": "a" * 501}, "question")],
)
def test_invalid_requests_list_the_fields(client: TestClient, payload: dict, field: str) -> None:
    body = _is_problem(client.post("/api/v1/ask", json=payload), 422, "invalid_request")
    assert [e["field"] for e in body["errors"]] == [field]


def test_malformed_json_is_a_problem(client: TestClient) -> None:
    r = client.post(
        "/api/v1/ask", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    _is_problem(r, 422, "invalid_request")


def test_rate_limits_are_reported_and_enforced(client: TestClient) -> None:
    quota, window = public_api.LIMITS.limits["default"]
    first = client.get("/api/v1/status")
    assert first.headers["ratelimit-policy"] == f'"default";q={quota};w={window}'
    assert first.headers["ratelimit"].startswith(f'"default";r={quota - 1};t=')
    for _ in range(quota - 1):
        client.get("/api/v1/status")
    r = client.get("/api/v1/status")
    _is_problem(r, 429, "rate_limited")
    assert int(r.headers["retry-after"]) >= 1
    assert r.headers["ratelimit"].startswith('"default";r=0;')


def test_rate_limits_are_per_client_and_per_window() -> None:
    limiter = public_api.RateLimiter({"b": (2, 10)})
    assert limiter.check("b", "a", now=0)[0] and limiter.check("b", "a", now=1)[0]
    allowed, headers = limiter.check("b", "a", now=2)
    assert not allowed and headers["Retry-After"] == "8"
    assert limiter.check("b", "other", now=2)[0]
    assert limiter.check("b", "a", now=10.5)[0]


def test_unexpected_errors_are_problems(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom() -> None:
        raise RuntimeError("secret detail")

    monkeypatch.setattr(public_api, "_demo_agent", boom)
    r = TestClient(client.app, raise_server_exceptions=False).post(
        "/api/v1/ask", json={"question": QUESTION}
    )
    body = _is_problem(r, 500, "internal_error")
    assert "secret detail" not in r.text and body["instance"] == "/api/v1/ask"


def test_site_404s_stay_html_outside_the_api(client: TestClient) -> None:
    r = client.get("/nope", headers={"Accept": "text/html"})
    assert r.status_code == 404 and r.headers["content-type"].startswith("text/html")


# ------------------------------------------------------------------ discovery


def test_api_catalog_is_an_rfc9727_linkset(client: TestClient) -> None:
    r = client.get("/.well-known/api-catalog")
    assert r.status_code == 200
    assert r.headers["content-type"] == server.API_CATALOG
    assert r.headers["link"] == '</.well-known/api-catalog>; rel="api-catalog"'
    linkset = r.json()["linkset"]
    assert len(linkset) == 1 and linkset[0]["anchor"] == f"{site.site_url()}/api"
    hrefs = {
        rel: [t["href"] for t in targets] for rel, targets in linkset[0].items() if rel != "anchor"
    }
    assert hrefs["service-desc"][0] == f"{site.site_url()}/openapi.json"
    for targets in hrefs.values():
        for href in targets:
            assert client.get(href.removeprefix(site.site_url())).status_code == 200, href
    head = client.head("/.well-known/api-catalog")
    assert head.status_code == 200 and head.headers["content-type"] == server.API_CATALOG
    assert head.headers["content-length"] == r.headers["content-length"] and not head.content


def test_pages_point_agents_at_the_api(client: TestClient) -> None:
    home = client.get("/", headers={"Accept": "text/html"})
    link = home.headers["link"]
    for rel in ('rel="service-desc"', 'rel="service-doc"', 'rel="api-catalog"', 'rel="alternate"'):
        assert rel in link, rel
    for part in (
        '<link rel="api-catalog" href="/.well-known/api-catalog">',
        'rel="service-desc"',
        'href="/developers"',
    ):
        assert part in home.text, part
    assert "/developers</loc>" in client.get("/sitemap.xml").text
    llms = client.get("/llms.txt").text
    assert "/openapi.json" in llms and "/developers.md" in llms and "veroniqa-cli" in llms
    assert 'rel="api-catalog"' not in client.get("/nope").headers.get("link", "")


def test_brand_and_product_share_the_one_h1() -> None:
    page = site.home_page()
    assert page.count("<h1") == 1
    h1 = re.search(r"<h1>(.*?)</h1>", page, re.S)
    assert h1
    text = re.sub(r"<[^>]+>", "", h1.group(1))
    assert text.startswith("VeroniQA") and "AI assistant for API testing" in text
    assert "<title>VeroniQA: the AI assistant for API testing</title>" in page
    graph = json.loads(page.split('application/ld+json">')[1].split("</script>")[0])["@graph"]
    app = next(n for n in graph if n["@type"] == "SoftwareApplication")
    assert "VeroniQA AI assistant" in app["alternateName"] and site.REPO_URL in app["sameAs"]
    webapi = next(n for n in graph if n["@type"] == "WebAPI")
    assert webapi["documentation"].endswith("/developers")


# ------------------------------------------------------------------ developer portal


def test_developer_portal_html_and_markdown(client: TestClient) -> None:
    page = client.get("/developers", headers={"Accept": "text/html"})
    md = client.get("/developers", headers={"Accept": "text/markdown"})
    assert page.status_code == md.status_code == 200
    assert page.headers["vary"] == "Accept"
    assert '</developers.md>; rel="alternate"' in page.headers["link"]
    assert client.get("/developers.md").text == md.text
    assert page.text.count("<h1") == 1 and '<link rel="canonical"' in page.text
    for section in (
        "quickstart",
        "authentication",
        "endpoints",
        "errors",
        "rate-limits",
        "sandbox",
    ):
        assert f'id="{section}"' in page.text, section
    for code in public_api.ERRORS:
        assert f'id="error-{code}"' in page.text and f"`{code}`" in md.text
    for method, path, _, _ in devportal.endpoints():
        assert f"`{path}`" in md.text and method in md.text
    assert md.text.startswith("# VeroniQA for developers") and devportal.CLI_GIT in md.text


# ------------------------------------------------------------------ CLI


def _transport(client: TestClient):  # type: ignore[no-untyped-def]
    """Route the CLI's requests into the app instead of the network."""

    def send(method: str, url: str, body: bytes | None, headers: dict, timeout: float):  # type: ignore[no-untyped-def]
        assert url.startswith(veroniqa_cli.DEFAULT_URL)
        r = client.request(
            method, url.removeprefix(veroniqa_cli.DEFAULT_URL), content=body, headers=headers
        )
        return r.status_code, r.content

    return send


def test_cli_status_results_and_ask(client: TestClient, capsys: pytest.CaptureFixture[str]) -> None:
    send = _transport(client)
    assert veroniqa_cli.main(["status"], transport=send) == 0
    assert "ok" in capsys.readouterr().out
    assert veroniqa_cli.main(["results", "--json"], transport=send) == 0
    assert json.loads(capsys.readouterr().out)["recall_pct"] == site.STATS["recall_pct"]
    assert veroniqa_cli.main(["ask", QUESTION], transport=send) == 0
    out = capsys.readouterr().out
    assert "300 ms" in out and "05-performance.md" in out
    assert veroniqa_cli.main(["openapi"], transport=send) == 0
    assert json.loads(capsys.readouterr().out)["openapi"].startswith("3.1")
    assert veroniqa_cli.main(["demo-spec"], transport=send) == 0
    assert "paths" in json.loads(capsys.readouterr().out)


def test_cli_reports_problems_and_network_errors(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert veroniqa_cli.main(["ask", "x"], transport=_transport(client)) == veroniqa_cli.EXIT_API
    err = capsys.readouterr().err
    assert "invalid_request" in err and "question" in err and "openapi.json" in err

    def offline(*_args: object) -> tuple[int, bytes]:
        raise veroniqa_cli.NetworkError("connection refused")

    assert veroniqa_cli.main(["status"], transport=offline) == veroniqa_cli.EXIT_NETWORK
    assert "connection refused" in capsys.readouterr().err
    assert veroniqa_cli.main([]) == veroniqa_cli.EXIT_USAGE


def test_cli_base_url_from_flag_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def record(method: str, url: str, *_args: object) -> tuple[int, bytes]:
        seen.append(url)
        return 200, b'{"status": "ok"}'

    monkeypatch.setenv("VERONIQA_URL", "http://localhost:8502/")
    assert veroniqa_cli.main(["status", "--json"], transport=record) == 0
    assert (
        veroniqa_cli.main(
            ["--json", "--base-url", "https://example.test", "status"], transport=record
        )
        == 0
    )
    assert seen == ["http://localhost:8502/api/v1/status", "https://example.test/api/v1/status"]
    with pytest.raises(veroniqa_cli.NetworkError):
        veroniqa_cli.urllib_transport("GET", "file:///etc/passwd", None, {}, 1)
