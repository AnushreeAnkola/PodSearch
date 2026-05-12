from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EvalQuery:
    query: str
    relevant_chunk_ids: list[str]
    best_chunk_id: str | None = None
    metadata: dict = field(default_factory=dict)


def load_eval_set(path: str | Path) -> list[EvalQuery]:
    raw = yaml.safe_load(Path(path).read_text())
    if not raw:
        return []
    queries: list[EvalQuery] = []
    for entry in raw:
        queries.append(
            EvalQuery(
                query=entry["query"],
                relevant_chunk_ids=list(entry["relevant_chunk_ids"]),
                best_chunk_id=entry.get("best_chunk_id"),
                metadata=entry.get("metadata", {}),
            )
        )
    return queries
