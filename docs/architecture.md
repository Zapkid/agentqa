# Architecture

```mermaid
flowchart LR
  S["OpenAPI spec + requirement docs"] --> I["Ingestor (no LLM)<br/>parse · chunk · injection scan"]
  I --> V[("Chroma + BM25<br/>hybrid retrieval")]
  I --> SUP{{"Supervisor<br/>typed task DAG"}}
  V --> P["Planner (T2)"]
  SUP --> T0["Tier 0 synthesizer<br/>0 tokens"]
  SUP --> P
  P --> D["Dispatcher<br/>dedupe · budget · difficulty · tier"]
  D -->|"T1 batch / single"| G1["Generator (T1)"]
  D -->|"hard tasks"| G2["Generator (T2)"]
  G1 --> VER{"Verifiers<br/>static · grounding · confidence · reference build"}
  G2 --> VER
  T0 --> VER
  VER -->|"fail: repair once, then escalate"| D
  VER --> E["Sandboxed executor<br/>allowlist · method guard · rlimits · reruns"]
  E --> TR["Triage<br/>rules → clusters → T1 → T2"]
  TR --> J["Judge (other family) + reporter"]
  J --> R["Report · executive summary · lessons"]
  SUP --> WL["Workload model (T1)"] --> LR["Locust runner<br/>load guard"] --> PA["Perf analysis<br/>knee · SLO · A/B"] --> PT["Perf triage"]
```

| layer | module | what to read first |
|---|---|---|
| provider layer | `agentqa/llm/` | `router.py` (cache, rate limit, retries, fallback, schema repair, cost, span) |
| agents | `agentqa/agents/` | `planner.py`, `generator.py`, `synthesizer.py`, `triage.py`, `reporter.py` |
| guards | `agentqa/guards/` | `grounding.py`, `static_checks.py`, `injection.py`, `budget.py`, `killswitch.py`, `redaction.py` |
| orchestration | `agentqa/orchestrator/` | `graph.py` (DAG engine), `pipeline.py` (the run) |
| delegation | `agentqa/dispatch/` | `policy.py`, `cascade.py`, `savings.py`, `learned.py`, `strategies.py` |
| performance | `agentqa/perf/` | `locustfile.py`, `runner.py`, `analysis.py`, `pipeline.py` |
| evals | `agentqa/evals/` | `groundtruth.py`, `harness.py`, `run.py` |
| observability | `agentqa/obs/`, `deploy/` | `tracing.py`, `metrics.py`, `grafana/build_dashboards.py` |
| surfaces | `agentqa/cli.py`, `agentqa/mcp/server.py`, `agentqa/api/app.py` | |
| simulated models | `agentqa/llm/simulated.py`, `agentqa/sim/` | ADR 0003 |

Agents never call each other. The supervisor passes typed artifacts (`agentqa/models.py`) between
them, checkpoints each task result to SQLite, isolates failures, and checks the kill switch and the
budget before every task.
