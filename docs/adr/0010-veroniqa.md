# ADR 0010: Veroniqa, a chat interface with project knowledge and RAG

Status: accepted (2026-10-09)

## Context
AgentQA is driven by a CLI, an MCP server and a small read-only web UI. People who are not going
to type commands (a test lead, a product owner, a pre-sales demo) need one place to keep a
project's requirements, ask questions about them, and start a test run.

## Decision
- **One package, one code path.** `agentqa/veroniqa` holds the logic; the Streamlit app
  (`app.py`) only calls it. The UI buttons and the chat use the same functions, and the tests
  drive both (the app through Streamlit's AppTest).
- **Projects are folders** under `<AGENTQA_HOME>/projects/<slug>/`, with a validated slug (no path
  traversal) and a YAML config. No database, so a project can be inspected, backed up or deleted
  with ordinary tools.
- **Reuse the ingestion stack.** Sources become Markdown under `docs/` and go through the existing
  chunker, prompt-injection scan (heuristics plus the classifier) and Chroma + BM25 hybrid store,
  one collection per project. The same `docs/` folder is passed as the requirement docs when a
  test run starts, so the knowledge base and test generation never disagree.
- **Grounded answers.** The answer prompt sees numbered passages and must return the numbers it
  used. Citations to passages that were not retrieved are dropped and reported; an answer with no
  valid citation is marked unverified; "not found" is a first-class outcome.
- **Cheap routing.** Each message is routed to one action by the cheapest tier with a structured
  output schema (`ask`, `add_link`, `run_tests`, `list_runs`, `show_report`, `list_sources`,
  `help`). Model failures fall back to `ask`. Answers use the `veroniqa` role (cheap tier, strong
  on the premium profile).
- **Simulated stand-ins.** Routing is keyword-based and answers are extractive (sentences ranked by
  rare-term overlap). They are deterministic so the demo and tests are stable, and the README says
  they are not a model.
- **Links are an attack surface.** Fetching is http(s) only, public addresses only (re-checked
  after each redirect), 2 MB, text types only, no credentials in URLs. Private networks can be
  allowed explicitly with `AGENTQA_ALLOW_PRIVATE_LINKS=1`.
- **Streamlit** for the interface: fast to build, escapes HTML in Markdown by default, and has a
  headless test harness. It binds to 127.0.0.1 and has no authentication: it is a local tool.

## Consequences
- No multi-user support, authentication or per-user permissions (see docs/ENTERPRISE.md).
- DNS rebinding between the address check and the request is not covered (docs/SECURITY.md).
- Conversation history lives in the browser session only.
