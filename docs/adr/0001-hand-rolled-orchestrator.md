# ADR 0001: Hand-rolled supervisor instead of an agent framework

Status: accepted (2026-10-07)

## Context
The brief asks for a supervisor with a typed task DAG, strict pydantic contracts between
agents, failure isolation, SQLite checkpointing, cost-aware delegation and full tracing. The
author must be able to explain every component in an interview.

## Decision
Write the orchestrator by hand: a small DAG of typed tasks, a priority queue, a thread pool
for fan-out, and a SQLite checkpoint table. Agents are plain classes with `run(input) -> output`
contracts. No LangGraph/CrewAI/AutoGen.

## Consequences
- + Every routing, retry and escalation decision is ordinary Python that can be unit-tested
  with the fake adapter and shows up as a span event exactly where we put it.
- + No framework-specific tracing or state model to reconcile with OpenTelemetry.
- − We own scheduling and checkpoint code (~400 lines). A LangGraph adapter stays a stretch goal.
