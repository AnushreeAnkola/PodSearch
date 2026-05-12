import math
from collections.abc import Iterable


def hit_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    return 1.0 if any(rid in relevant_set for rid in retrieved[:k]) else 0.0


def mrr(retrieved: list[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for rank, rid in enumerate(retrieved, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for i, rid in enumerate(retrieved[:k], start=1):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
