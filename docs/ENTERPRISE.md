# Enterprise readiness notes

Documentation only. This is a proof of concept; it does not claim compliance with any regulation
or standard. Each section describes the controls that exist in the code and the gaps that remain.

## Data residency and PII
- **Exists:** customer specs and docs are processed locally; embeddings are computed locally
  (no embedding API). LLM calls are the only egress, and they go only to the providers configured in
  `config/models.yaml`. Logs, spans, request logs and reports pass through central redaction
  (`agentqa/guards/redaction.py`: API keys, bearer tokens, sensitive headers, e-mail addresses,
  Luhn-valid card numbers); the Collector applies a second redaction pass before export.
  Prompts are normalised so ids and hosts from the target are not sent where not needed.
- **Gaps:** redaction is pattern-based (names, addresses and free-text PII inside API responses are
  not detected); no data-classification step decides what may leave the network; no retention
  policy for run artifacts.

## Model hosting options
- **Exists:** one `LLMClient` interface with per-role routing, so a profile can point every role at a
  private endpoint (Anthropic models on AWS Bedrock / Google Vertex / Azure via their clients,
  Gemini on Vertex, open-weight models behind an OpenAI-compatible server via the OpenRouter-style
  adapter). The simulated provider shows the pipeline can run with no external model at all.
- **Gaps:** only the three public adapters are implemented; private-endpoint configs are not tested.

## Audit trail
- **Exists:** every LLM call is a span with prompt name, version and hash, model, tokens, cost and
  cache status; every prompt/response pair is in the disk cache keyed by its hash; every delegation
  decision (tier, why, verifier outcome, escalation path) is a SQLite row and a span event;
  checkpoints record every task result; reports cite evidence by reference.
- **Gaps:** the audit store is a local SQLite file without tamper evidence (no hash chain or WORM
  storage); no signed reports.

## Human-in-the-loop points
- **Exists:** `needs_review` findings (unsupported or judge-rejected claims) are routed to a person and
  never counted as confirmed bugs; mutations are blocked unless the target is declared a sandbox;
  load tests only run against allowlisted sandbox targets; the kill switch stops everything.
- **Gaps (would add for production):** approval before the first run against a new target; approval
  before publishing a report externally; a review queue UI for `needs_review` items.

## Access control
- **Exists:** none for the tool itself (a non-goal of this PoC). Target credentials come from a
  per-target config; provider keys only from the environment.
- **Gaps:** SSO and RBAC for the UI and MCP server, per-tenant isolation of stores and caches,
  secrets in a vault instead of environment variables, MCP transport authentication.

## How the guardrails map to AI risk-management practice
Framed loosely on the NIST AI RMF functions (govern, map, measure, manage), as descriptive mapping:

| practice | control in this repo |
|---|---|
| Map: know where untrusted input enters | retrieved docs are untrusted: injection heuristics + classifier, quarantine, delimiting, fixed tool permissions |
| Measure: quantify errors | differential ground truth; hallucination rate; triage confusion matrix; false alarms on clean; perf false-alarm rate on clean-vs-clean |
| Measure: track drift | replay gate on every PR; nightly live eval; eval trend panel |
| Manage: limit blast radius | sandbox allowlist and method guard; load guard; budget guard; kill switch; circuit breakers |
| Manage: human oversight | `needs_review`; judge downgrade of unsupported claims; reports show rule-based and model diagnoses side by side |
| Govern: accountability | versioned prompts in the cache key and span attributes; ADRs; delegation ledger |
