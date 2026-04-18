from backend.app.models.domain import TranscriptLine
from backend.app.services.chunker import chunk_naive


def _make_lines() -> list[TranscriptLine]:
    return [
        TranscriptLine(text="First line", start_seconds=0, episode_id="ep1", line_index=0),
        TranscriptLine(text="Second line", start_seconds=30, episode_id="ep1", line_index=1),
        TranscriptLine(text="Third line", start_seconds=60, episode_id="ep1", line_index=2),
    ]


def test_chunk_naive_count():
    lines = _make_lines()
    chunks = chunk_naive(lines, "ep1")
    assert len(chunks) == 3


def test_chunk_naive_ids():
    chunks = chunk_naive(_make_lines(), "ep1")
    assert chunks[0].chunk_id == "ep1_chunk_0000"
    assert chunks[2].chunk_id == "ep1_chunk_0002"


def test_chunk_naive_timestamps():
    chunks = chunk_naive(_make_lines(), "ep1")
    assert chunks[0].start_seconds == 0
    assert chunks[0].end_seconds == 30
    assert chunks[1].start_seconds == 30
    assert chunks[1].end_seconds == 60
    # Last chunk gets start + 30 as end
    assert chunks[2].end_seconds == 90


def test_chunk_naive_text_includes_timestamp():
    chunks = chunk_naive(_make_lines(), "ep1")
    assert chunks[0].text.startswith("[00:00]")
    assert "First line" in chunks[0].text
