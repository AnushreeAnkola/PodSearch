import time
from contextlib import contextmanager
from typing import Generator

from backend.app.core.logging import get_logger

log = get_logger(__name__)


@contextmanager
def track_latency(operation: str) -> Generator[dict, None, None]:
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        result["elapsed_ms"] = elapsed_ms
        log.info("latency", operation=operation, elapsed_ms=round(elapsed_ms, 2))
