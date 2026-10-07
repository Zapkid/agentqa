# ADR 0004: Local embeddings with a hashing fallback, and hybrid retrieval

Status: accepted (2026-10-07)

## Context
RAG must not consume API quota, and CI / air-gapped installs cannot download embedding weights
(the build host could not reach Hugging Face).

## Decision
- `Embedder` interface. `fastembed` (BAAI/bge-small-en-v1.5, ONNX, local) when selected with
  `AGENTQA_EMBEDDER=fastembed` and its weights are available; otherwise a deterministic
  feature-hashing embedder (unigrams + bigrams, 512 dims, no download).
- Because a hashing embedder is weak on its own, retrieval is **hybrid**: for each endpoint
  two queries (what the endpoint is; the vocabulary of its fields) are run against both the
  vector index and an in-process BM25 index, and the four rankings are merged round-robin
  (rank 1 of each, then rank 2, ...). On a corpus of tens of chunks, reciprocal-rank fusion let
  a chunk that is mediocre everywhere beat the chunk that is first for one query; round-robin
  guarantees each query's best hit is kept. Measured on the bundled docs: with k=5 every chunk
  that states a business rule for an endpoint is retrieved for that endpoint (see
  `tests/test_generation_guards.py::test_planner_simulated_cites_retrieved_chunks`).
- Chroma stores chunks with a content hash; unchanged chunks are not re-embedded (idempotent).

## Consequences
Retrieval quality with real embeddings is likely better; switching is a config change, but
the measurements in this repo were made with the hashing embedder.
