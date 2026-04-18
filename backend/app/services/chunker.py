from backend.app.models.domain import Chunk, TranscriptLine
from backend.app.services.transcript_parser import format_timestamp


def chunk_naive(lines: list[TranscriptLine], episode_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, line in enumerate(lines):
        next_start = lines[i + 1].start_seconds if i + 1 < len(lines) else line.start_seconds + 30
        chunk = Chunk(
            chunk_id=f"{episode_id}_chunk_{i:04d}",
            text=f"{format_timestamp(line.start_seconds)} {line.text}",
            episode_id=episode_id,
            start_seconds=line.start_seconds,
            end_seconds=next_start,
        )
        chunks.append(chunk)
    return chunks
