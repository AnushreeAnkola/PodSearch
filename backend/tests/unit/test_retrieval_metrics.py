import math

from backend.app.evaluation.retrieval_metrics import hit_at_k, mrr, ndcg_at_k


def test_hit_at_k_first_position():
    assert hit_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0


def test_hit_at_k_at_boundary_k():
    assert hit_at_k(["x", "y", "a"], {"a"}, k=3) == 1.0
    assert hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_hit_at_k_no_match():
    assert hit_at_k(["x", "y", "z"], {"a"}, k=5) == 0.0


def test_mrr_first_hit():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_third_hit():
    assert mrr(["x", "y", "a"], {"a"}) == pytest_approx(1 / 3)


def test_mrr_no_hit():
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


def test_ndcg_perfect_ranking():
    # Single relevant doc at position 1: DCG = 1/log2(2) = 1, IDCG = 1
    assert ndcg_at_k(["a", "b"], {"a"}, k=2) == 1.0


def test_ndcg_partial_match():
    # Relevant doc at position 2: DCG = 1/log2(3), IDCG = 1
    score = ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert score == pytest_approx(1 / math.log2(3))


def test_ndcg_two_relevant_at_top():
    # Both at positions 1 and 2: DCG = 1 + 1/log2(3); IDCG same → 1.0
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_ndcg_no_relevant_in_topk():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_ndcg_empty_relevant_set():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


def pytest_approx(expected: float, tol: float = 1e-9):
    class _Approx:
        def __eq__(self, other):
            return abs(other - expected) < tol

        def __repr__(self):
            return f"~{expected}"

    return _Approx()
