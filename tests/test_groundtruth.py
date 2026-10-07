from __future__ import annotations

from agentqa.evals.groundtruth import BUG_IDS, GroundTruth
from agentqa.evals.harness import aggregate


def gt() -> GroundTruth:
    g = GroundTruth()
    g.outcomes["clean"] = {
        "t_ok": "passed",
        "t_b1": "passed",
        "t_bad": "failed",
        "t_combo": "passed",
        "t_skip": "skipped",
    }
    for b in BUG_IDS:
        g.outcomes[b] = {
            "t_ok": "passed",
            "t_b1": "failed" if b == "B01" else "passed",
            "t_bad": "failed",
            "t_combo": "passed",
        }
    g.outcomes["all"] = {"t_ok": "passed", "t_b1": "failed", "t_bad": "failed", "t_combo": "failed"}
    return g


def test_detection_and_labels() -> None:
    g = gt()
    assert g.detected() == {"B01": ["t_b1"]}
    assert g.label("t_b1") == "product_bug"
    assert g.label("t_bad") == "test_bug"
    assert g.label("t_combo") == "interaction"
    assert g.label("t_ok") == "pass"
    g.flaky_tests.add("t_ok")
    assert g.label("t_ok") == "flaky"
    assert g.validity_rate() == 3 / 4  # skipped tests are not counted


def test_aggregate_mean_and_spread() -> None:
    cases = [
        {
            "recall": 0.5,
            "tokens_total": 100,
            "recall_by_category": {"a": 1.0},
            "bugs_detected": ["B01"],
            "seed": 0,
        },
        {
            "recall": 1.0,
            "tokens_total": 300,
            "recall_by_category": {"a": 0.0},
            "bugs_detected": ["B01", "B02"],
            "seed": 1,
        },
    ]
    a = aggregate(cases)
    assert a["n"] == 2 and a["recall"]["mean"] == 0.75 and a["recall"]["min"] == 0.5
    assert a["tokens_total"]["max"] == 300 and a["recall_by_category"] == {"a": 0.5}
    assert (
        a["bug_detection_frequency"]["B01"] == "2/2"
        and a["bug_detection_frequency"]["B02"] == "1/2"
    )
    assert "seed" not in a
