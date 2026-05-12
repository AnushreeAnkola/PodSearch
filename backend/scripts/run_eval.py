"""Compare retriever modes on hand-labeled queries.

Usage:
    python -m backend.scripts.run_eval --queries backend/data/eval/queries.yaml
    python -m backend.scripts.run_eval --queries ... --top-k 5 --out eval.md
"""
import argparse
import asyncio
import time
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.evaluation.dataset import EvalQuery, load_eval_set
from backend.app.evaluation.retrieval_metrics import hit_at_k, mrr, ndcg_at_k
from backend.app.providers.factory import (
    build_embedding_provider,
    build_vector_store,
)
from backend.app.services.retrieval.base import Retriever
from backend.app.services.retrieval.bm25 import BM25Retriever
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.retrieval.semantic import SemanticRetriever


async def _evaluate_mode(
    name: str,
    retriever: Retriever,
    queries: list[EvalQuery],
    top_k: int,
    ndcg_k: int,
) -> dict:
    hits, mrrs, ndcgs, latencies_ms = [], [], [], []
    for q in queries:
        start = time.perf_counter()
        results = await retriever.retrieve(q.query, top_k=max(top_k, ndcg_k))
        latencies_ms.append((time.perf_counter() - start) * 1000)
        retrieved_ids = [r.chunk.chunk_id for r in results]
        hits.append(hit_at_k(retrieved_ids, q.relevant_chunk_ids, top_k))
        mrrs.append(mrr(retrieved_ids, q.relevant_chunk_ids))
        ndcgs.append(ndcg_at_k(retrieved_ids, q.relevant_chunk_ids, ndcg_k))
    n = max(len(queries), 1)
    return {
        "mode": name,
        "hit_at_k": sum(hits) / n,
        "mrr": sum(mrrs) / n,
        "ndcg": sum(ndcgs) / n,
        "avg_latency_ms": sum(latencies_ms) / n,
    }


def _format_table(rows: list[dict], top_k: int, ndcg_k: int) -> str:
    header = (
        f"| Mode | hit@{top_k} | MRR | NDCG@{ndcg_k} | avg_latency_ms |\n"
        f"|------|--------|-----|---------|----------------|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['hit_at_k']:.3f} | {r['mrr']:.3f} "
            f"| {r['ndcg']:.3f} | {r['avg_latency_ms']:.1f} |"
        )
    return header + "\n".join(lines)


async def run(queries_path: Path, top_k: int, ndcg_k: int, out_path: Path | None) -> None:
    settings = get_settings()
    queries = load_eval_set(queries_path)
    if not queries:
        print(f"No queries loaded from {queries_path}")
        return

    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings)
    chunks = await store.fetch_all()
    if not chunks:
        print(
            "Vector store is empty. Run "
            "`python -m backend.scripts.ingest_folder backend/data/transcripts/` first."
        )
        return

    semantic = SemanticRetriever(embedder, store)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(semantic, bm25, k_rrf=settings.rrf_k)

    rows = []
    for name, retriever in (("semantic", semantic), ("bm25", bm25), ("hybrid", hybrid)):
        rows.append(await _evaluate_mode(name, retriever, queries, top_k, ndcg_k))

    table = _format_table(rows, top_k, ndcg_k)
    print(table)
    if out_path is not None:
        out_path.write_text(table + "\n")
        print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retriever modes.")
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(run(args.queries, args.top_k, args.ndcg_k, args.out))


if __name__ == "__main__":
    main()
