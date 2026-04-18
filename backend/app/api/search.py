from fastapi import APIRouter, Depends

from backend.app.api.deps import get_app_settings, get_llm, get_retriever
from backend.app.core.config import Settings
from backend.app.core.telemetry import track_latency
from backend.app.models.api import CitationResponse, SearchRequest, SearchResponse
from backend.app.providers.llm.base import LLMProvider
from backend.app.services.generation import generate_answer
from backend.app.services.retrieval.base import Retriever

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
    llm: LLMProvider = Depends(get_llm),
    settings: Settings = Depends(get_app_settings),
):
    with track_latency("search_total"):
        results = await retriever.retrieve(body.query, top_k=body.top_k)
        answer = await generate_answer(
            query=body.query,
            results=results,
            llm=llm,
            model_name=settings.llm_model,
        )

    return SearchResponse(
        answer=answer.text,
        citations=[
            CitationResponse(
                episode_id=c.episode_id,
                timestamp=c.timestamp,
                start_seconds=c.start_seconds,
                end_seconds=c.end_seconds,
                text=c.text,
            )
            for c in answer.citations
        ],
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )
