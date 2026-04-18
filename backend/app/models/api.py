from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    episode_id: str | None = None


class CitationResponse(BaseModel):
    episode_id: str
    timestamp: str
    start_seconds: int
    end_seconds: int
    text: str


class SearchResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    model: str
    input_tokens: int
    output_tokens: int


class IngestRequest(BaseModel):
    episode_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    episode_id: str
    chunks_created: int
