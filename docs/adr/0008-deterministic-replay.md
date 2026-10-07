# ADR 0008: Deterministic replay

Status: accepted (2026-10-07)

## Decision
The CI gate replays committed LLM-cache fixtures in `replay_only` mode, where any cache miss is an
error. That only works if every prompt is byte-identical across runs, so run-specific values are
normalised out of prompts (`agentqa/llm/normalize.py`): local base URLs and ports, generated ids,
timestamps, absolute paths, durations. Evidence shown to people keeps the raw values.
`AGENTQA_CACHE_DEBUG=<dir>` dumps the exact hashed payload of every call, which is how three
sources of drift were found (absolute paths in cascade feedback, path-dependent truncation in
triage symptoms, checkpoint resume of an in-memory task).

## Consequences
Replay catches code regressions for free and without keys. A prompt or model change invalidates
the fixtures by design; the nightly live eval re-scores real models and refreshes them.
