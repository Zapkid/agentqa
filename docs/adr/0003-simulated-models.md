# ADR 0003: Deterministic simulated models for CI and offline development

Status: accepted (2026-10-07)

## Context
The build environment and CI have no API keys, but the dispatcher, the cascade, the guards
and the eval harness must run end to end on every PR.

## Decision
Add a `simulated` provider with three models (`sim-cheap`, `sim-strong`, `sim-judge`) whose
error rates are *declared* in code (`agentqa/llm/simulated.py`). Responders are seeded by a
hash of the request, so identical inputs give identical outputs. The cheap model makes more
grounding and logic mistakes than the strong one, which gives verification-gated escalation
real failures to catch.

## Consequences
- Results produced with the `simulated` profile measure pipeline mechanics (does the cascade
  escalate, does the grounding guard catch invented endpoints, does the gate trip) and never a
  real model's quality. Every report built from them is labelled "SIMULATED".
- Real-model results come from `make eval-live` with keys present.
