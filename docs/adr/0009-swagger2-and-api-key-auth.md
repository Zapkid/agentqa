# ADR 0009: Swagger 2.0 input and API-key auth

Status: accepted (2026-10-08)

## Context
The first target was an OpenAPI 3 document with bearer-token auth, so the ingestor rejected
Swagger 2.0 and the target config could only send `Authorization: Bearer`. A large share of real
APIs publish Swagger 2.0 and use an API-key header. No public test API could be reached from the
build host (network policy), so the second target had to be built locally.

## Decision
- **Convert, don't parse twice.** `agentqa/ingest/swagger2.py` converts a Swagger 2.0 document to
  OpenAPI 3.0.3 inside `load_spec`. Everything downstream (parser, Tier 0, grounding, prompts)
  keeps one model of an API. Converted: parameters (`body`, `formData`, shared `$ref` parameters),
  responses, `definitions`, `securityDefinitions`, `x-nullable`, `host`/`basePath`/`schemes`.
- **Auth is a config choice, not a code path.** `auth.scheme` is `bearer` (default) or `api_key`
  with `auth.header`. Generated tests are unaffected: they already read headers from the `auth`
  fixture. `X-API-Key` was already in the redaction key list.
- **A second sample target, `examples/tasks_api/`.** A small in-memory FastAPI service whose
  published contract is a hand-written Swagger 2.0 file, with four seeded defects (T01-T04). A test
  keeps the contract and the routes in step. Two defects are visible from the spec alone (Tier 0
  finds them), two need the requirement docs (a model has to write those tests).

## Consequences
- Not supported: `basic` and OAuth token acquisition (the security schemes convert, but AgentQA
  sends only a configured bearer token or API key), external `$ref`s, XML bodies.
- The conversion is tested against a hand-built document and the Tasks API contract, not against
  a corpus of real-world Swagger files. Expect gaps on unusual documents.
- Nothing here was run against a third-party API. See `docs/LIMITATIONS.md`.
