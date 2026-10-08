# Security

AgentQA is a proof of concept. This page records the threat model, what is enforced today, what was
tightened in the 2026-10-08 review, and what is still open. If the README sounds stronger than this
page, this page wins.

## What is untrusted
- **The target API's responses.** They end up in evidence, findings and reports.
- **Requirement documents.** They are retrieved into prompts and can try to hijack the agents.
- **Model output.** Generated test code runs on your machine; findings and prose reach reports.
- **The spec.** A hostile spec could try to mislead the planner or the generators.

## Controls in place
| area | control | where |
|---|---|---|
| generated code | static checks (import allowlist, banned calls), grounding against the spec, then a sandboxed pytest process: CPU/memory/time limits, a client- and socket-level host allowlist, read-only HTTP methods unless the target is declared `sandbox: true` | `guards/static_checks.py`, `guards/grounding.py`, `executor/` |
| prompt injection | heuristic scan plus a small-model classifier that fails closed, quarantine, delimited context, fixed tool permissions | `guards/injection.py` |
| secrets and PII | central redaction of logs, spans, reports and generated code; `X-API-Key` and `Authorization` values never reach logs; pre-commit and CI secret scan | `guards/redaction.py`, `scripts/secret_scan.py` |
| spend | token, USD, wall-clock and step budgets; kill switch | `guards/budget.py`, `guards/killswitch.py` |
| load | allowlisted sandbox targets only; caps on users, duration, request rate and seeded data; automatic abort on target errors or host pressure | `perf/runner.py`, `perf/locustfile.py` |
| reports | HTML is escaped; Markdown files have raw HTML neutralised outside code; only http(s), anchor and relative links are rendered as links | `agents/reporter.py` |
| SQL | values are always bound; the few interpolated identifiers come from fixed lists or an allowlist (`Store.RUN_FIELDS`) | `store.py` |
| network exposure | the web UI and the MCP HTTP server bind to `127.0.0.1` by default; every compose port is published on `127.0.0.1` only | `cli.py`, `mcp/server.py`, `deploy/docker-compose.yml` |
| supply chain | every dependency has a floor and an upper bound; `uv.lock` pins the rest; CI installs with `--locked`; Dependabot proposes bumps; weekly `make audit` | `pyproject.toml`, `.github/` |
| lint | ruff security rules (`S`) run in CI; each intentional exception carries a reason in a `noqa` comment | `pyproject.toml` |

Policy tests in `tests/test_hardening.py` fail if a dependency loses its upper bound, a compose port
leaves loopback, a workflow loses least privilege or interpolates inputs into a shell, the target
image runs as root, or the code starts using Chroma's HTTP client.

## Review of 2026-10-08: what changed
- Guardrails that were configured but not enforced now are: `sandbox.read_only_methods`,
  `load.max_rps` and `data_setup_orders_cap`. Options that nothing read were removed
  (`injection.*`, `api_key_env`, `native_*`, `t0_categories`) so the config no longer promises
  behaviour that does not exist.
- `Store.save_run` no longer builds SQL from caller-supplied column names.
- The HTML report no longer turns `[text](javascript:...)` from target-controlled text into a link,
  and Markdown files no longer carry raw HTML.
- Workflows run with `contents: read`; inputs reach shell steps through `env`; model API keys are
  scoped to the steps that call models; the nightly live eval only runs when the repository
  variable `NIGHTLY_EVAL` is `true` (it costs money).
- Compose: loopback-only ports, Grafana anonymous access is `Viewer` (was `Admin`), the target
  container runs as a non-root user with dropped capabilities.

## Known advisories
`make audit` ignores four chromadb advisories (PYSEC-2026-311, -3813, -3814, -3815; five findings
because one is reported twice). They concern Chroma's **server**: its HTTP endpoints,
`trust_remote_code` model repositories, and multi-tenant authorisation. No fixed release exists yet.
AgentQA uses only the embedded `PersistentClient` / `EphemeralClient`, never runs a Chroma server and
never enables `trust_remote_code`; a test fails if the code starts using the HTTP client. Revisit
when a fixed version ships or if a server deployment is ever considered. Any other advisory fails
the audit.

## Still open (client-phase work)
- **Sandbox is process-level, not a container.** Rlimits, import and host allowlists reduce the
  blast radius of a wrong test; they are not isolation against a determined adversary. Run the
  executor in a container or microVM for untrusted specs.
- **GitHub Actions are pinned by tag, not commit SHA.** The SHAs could not be verified from the
  build host. Pin them before use in a client environment (Dependabot keeps them current).
- **cAdvisor runs `privileged` and mounts the host root read-only** in the compose file. Fine for a
  local demo; remove or replace it elsewhere.
- **Local defaults are placeholders**: `TARGET_WEBHOOK_SECRET` and the Grafana admin password both
  default to `change-me-local-only`. Set real values in `.env` for anything shared.
- **No authentication on the web UI or the MCP HTTP transport.** They are loopback-only by default;
  do not expose them. See `docs/ENTERPRISE.md` for SSO/RBAC and audit storage.
- **The compose stack has never been run** (no Docker in the build environment), so the hardening
  there is validated as YAML only.
- **The secret scanner is a regex scanner**, not a substitute for a managed one (GitGuardian also
  runs on pull requests in this repository).
- **Run data is stored unencrypted** under `.agentqa/` (redacted, but it contains target responses).

## Reporting a problem
Open a private security advisory on the repository, or contact the maintainer directly.
