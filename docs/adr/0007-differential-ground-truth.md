# ADR 0007: Ground truth by differential testing; reference-build verification

Status: accepted (2026-10-07)

## Decision
- No hand labels. A generated suite is executed against the clean build, each single-bug build
  and the all-bugs build. A bug is detected if some test passes on clean and fails on its
  single-bug build; a test failing on clean is a test bug; a test that passes on clean and fails on
  all-bugs and on some single-bug build is a product bug; outcome flips across reruns are flaky.
- Generated tests are verified before execution by running them alone against a *reference build*
  (in the benchmark, the clean build) when one is configured: "executes and passes on the clean
  build" is the strongest cheap verifier the brief lists. In a customer setting the reference is
  the last release or a staging build believed correct; without one, that verifier is skipped and
  static checks, grounding and confidence remain.

## Consequences
- Using the clean build both as the cascade's reference and as the ground-truth baseline makes
  test validity on clean near 100% by construction. That is stated next to every validity number;
  the informative metrics are recall, triage accuracy and what the cascade had to repair.
- Bugs that only show when combined (one bug masking another's setup) are labelled
  `interaction` and counted as product bugs for triage scoring.
