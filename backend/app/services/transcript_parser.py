import re

from backend.app.models.domain import TranscriptLine

_TIMESTAMP_RE = re.compile(r"^\[(\d{1,2}):(\d{2})\]\s*(.+)$")


def parse_timestamp(ts: str) -> int:
    m = re.match(r"(\d{1,2}):(\d{2})", ts)
    if not m:
        raise ValueError(f"Invalid timestamp: {ts}")
    return int(m.group(1)) * 60 + int(m.group(2))


def format_timestamp(seconds: int) -> str:
    return f"[{seconds // 60:02d}:{seconds % 60:02d}]"


def parse_transcript(raw_text: str, episode_id: str) -> list[TranscriptLine]:
    lines: list[TranscriptLine] = []
    for i, raw_line in enumerate(raw_text.strip().splitlines()):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        match = _TIMESTAMP_RE.match(raw_line)
        if not match:
            continue
        minutes, seconds, text = int(match.group(1)), int(match.group(2)), match.group(3)
        start_seconds = minutes * 60 + seconds
        lines.append(
            TranscriptLine(
                text=text,
                start_seconds=start_seconds,
                episode_id=episode_id,
                line_index=i,
            )
        )
    return lines
