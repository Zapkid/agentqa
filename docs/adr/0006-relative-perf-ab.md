# ADR 0006: Relative A/B performance checks with a measured noise band

Status: accepted (2026-10-07)

## Context
Shared CI runners and a shared build host are noisy; absolute latency thresholds would flap.

## Decision
- Every performance verdict is relative: the same validated WorkloadSpec runs against a baseline
  build and a candidate build back to back on the same host, the target pinned to one CPU core
  (`os.sched_setaffinity`, a stand-in for container CPU limits when Docker is not available), at
  least 3 iterations, warm-up discarded per step, medians with min/max spread.
- The noise band is *measured*: clean repeats give the largest observed log-ratio per metric (and
  the largest memory-growth difference); the band is 2x that, floored at 25% (p95), 15%
  (throughput) and 50 MB (memory growth). Scenario cells with fewer than 50 requests never set the
  band or flag a regression. The benchmark measures false alarms on a held-out clean repeat that
  did not contribute to the band.
- History: the first benchmark run estimated the band from a single clean pair with a 1.5x margin
  and a 15 MB memory floor, and its held-out clean repeat was flagged (false-alarm rate 100% on one
  sample). A diagnostic run showed clean-vs-clean memory growth varying by ~11 MB (SQLite page
  cache), close to that floor. The floor, margin and held-out design above are the fix. (The
  first run's result file was overwritten by the re-run, which had the same date and commit; this
  paragraph is the record.) The re-run's false-alarm test is a single held-out repeat, so "0 of 1"
  is weak evidence; more clean repeats in the nightly job would tighten it.
- Server-side signals from the target's own counters also count as regressions: DB queries per
  request (the "span count per request" signature of N+1), DB time share, memory growth.
- The LLM never sees raw logs: deterministic code builds one compact evidence bundle per
  comparison; a rule-based diagnosis is shown next to the model's so disagreements are visible.
- Absolute numbers appear only with the host described, and are labelled as such.

## Update (2026-10-08)
The `perf-ab-smoke` CI job flagged a clean build against itself: its band came from a single
clean pair, which is too few samples on a shared runner. `agentqa perf ab` now takes
`--clean-repeats` (default 2) and builds the band from all of them. Local check: three
clean-vs-clean runs were not flagged and the P01 candidate was. One local run showed a 3.4x p95
spread between repeats, which widens the band (it takes the worst repeat); a robust statistic or
more repeats is the next step if that matters on CI.
