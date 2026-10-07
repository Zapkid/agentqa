# Using AgentQA from an MCP client

AgentQA ships an MCP server (`agentqa-mcp`, MCP Python SDK 2.x) with stdio and streamable-HTTP
transports.

| tool | what it does |
|---|---|
| `ingest_spec` | parse a spec + docs; list endpoints and quarantined (prompt-injection) chunks |
| `plan_tests` | planner agent only, for one endpoint |
| `run_suite` | full pipeline; `local_target="all"` runs against the bundled demo API with every seeded bug |
| `triage_run` | findings of a run with evidence refs and curl repros |
| `get_report` | executive summary or technical report (Markdown) |
| `list_runs` | recent runs |
| `perf_ab` | clean vs candidate load A/B with noise band and diagnosis (local sandbox builds only) |

Resources: `agentqa://runs/{run_id}/report`, `agentqa://runs/{run_id}/summary`.

## Claude Code
```bash
claude mcp add agentqa -- uv --directory /path/to/agentqa run agentqa-mcp
```
or in `.mcp.json` at the project root:
```json
{"mcpServers": {"agentqa": {"command": "uv", "args": ["--directory", "/path/to/agentqa", "run", "agentqa-mcp"],
  "env": {"AGENTQA_PROFILE": "simulated"}}}}
```
Then ask: "Use agentqa to test the demo API with all seeded bugs and summarise what blocks the release."

## Cursor
`~/.cursor/mcp.json` (or `.cursor/mcp.json` in the project):
```json
{"mcpServers": {"agentqa": {"command": "uv", "args": ["--directory", "/path/to/agentqa", "run", "agentqa-mcp"]}}}
```

## Streamable HTTP
```bash
uv run agentqa-mcp --http --port 8765     # endpoint: http://127.0.0.1:8765/mcp
```

## Verified
`tests/test_mcp_server.py` drives the server over stdio with the official MCP client: lists the
tools, ingests, runs a suite against a two-bug build, reads findings and the summary resource.
A screenshot or GIF of a desktop client driving a run was not produced in the build environment
(no GUI client there); see `docs/LIMITATIONS.md`.
