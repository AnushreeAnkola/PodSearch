from dataclasses import dataclass, field


@dataclass
class TranscriptLine:
    text: str
    start_seconds: int
    episode_id: str
    line_index: int


@dataclass
class Chunk:
    chunk_id: str
    text: str
    episode_id: str
    start_seconds: int
    end_seconds: int
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    episode_id: str
    start_seconds: int
    end_seconds: int
    text: str
    timestamp: str  # formatted [MM:SS]


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    model: str
    input_tokens: int
    output_tokens: int
