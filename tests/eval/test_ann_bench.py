import pytest

from playground_retrieval.eval import LatencySummary, latency_summary, recall_at_k


def test_recall_at_k_averages_overlap_per_query():
    truth = [["a", "b", "c"], ["d", "e", "f"]]
    found = [["b", "a", "x"], ["d", "e", "f"]]

    assert recall_at_k(found, truth, 2) == pytest.approx((2 / 2 + 2 / 2) / 2)
    assert recall_at_k(found, truth, 3) == pytest.approx((2 / 3 + 3 / 3) / 2)


def test_recall_at_k_counts_missing_results_as_misses():
    assert recall_at_k([["a"]], [["a", "b"]], 2) == pytest.approx(0.5)


def test_recall_at_k_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        recall_at_k([["a"]], [], 1)


def test_latency_summary_percentiles():
    assert latency_summary([float(i) for i in range(1, 101)]) == LatencySummary(
        p50_ms=50.5, p99_ms=99.01
    )
