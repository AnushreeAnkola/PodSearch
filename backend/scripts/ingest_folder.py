"""Batch-ingest all .txt transcripts from a folder."""
import asyncio
import sys
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging, get_logger
from backend.app.providers.factory import build_embedding_provider, build_vector_store
from backend.app.services.ingestion import ingest_text

log = get_logger(__name__)


async def main(folder: Path) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    embedder = build_embedding_provider(settings)
    store = build_vector_store(settings)

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        log.warning("no_transcripts_found", folder=str(folder))
        return

    total_chunks = 0
    for f in txt_files:
        episode_id = f.stem
        raw_text = f.read_text()
        chunks = await ingest_text(episode_id, raw_text, embedder, store)
        total_chunks += len(chunks)
        log.info("ingested_file", file=f.name, chunks=len(chunks))

    log.info("ingest_complete", files=len(txt_files), total_chunks=total_chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m backend.scripts.ingest_folder <folder_path>")
        sys.exit(1)
    asyncio.run(main(Path(sys.argv[1])))
