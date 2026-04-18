import re

from backend.app.core.logging import get_logger
from backend.app.core.telemetry import track_latency
from backend.app.models.domain import Answer, Citation, RetrievalResult
from backend.app.providers.llm.base import LLMProvider
from backend.app.services.transcript_parser import format_timestamp

log = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a podcast search assistant. Answer the user's question using ONLY the \
provided transcript excerpts. Follow these rules strictly:

1. Only use information present in the provided excerpts.
2. Cite your sources using the [MM:SS] timestamp format that appears at the start of each excerpt.
3. If the excerpts do not contain enough information to answer, say so explicitly.
4. Keep answers concise and directly relevant to the question.
5. Include at least one timestamp citation in your answer."""


def build_context(results: list[RetrievalResult]) -> str:
    parts: list[str] = []
    for r in results:
        parts.append(f"[Episode: {r.chunk.episode_id}] {r.chunk.text}")
    return "\n\n".join(parts)


def build_user_prompt(query: str, context: str) -> str:
    return f"Transcript excerpts:\n\n{context}\n\nQuestion: {query}"


def extract_citations(
    answer_text: str, results: list[RetrievalResult]
) -> list[Citation]:
    timestamps_in_answer = re.findall(r"\[(\d{1,2}:\d{2})\]", answer_text)
    chunk_by_ts: dict[str, RetrievalResult] = {}
    for r in results:
        ts = format_timestamp(r.chunk.start_seconds)
        chunk_by_ts[ts] = r

    citations: list[Citation] = []
    seen: set[str] = set()
    for ts_raw in timestamps_in_answer:
        ts_formatted = f"[{ts_raw}]"
        if ts_formatted in seen:
            continue
        seen.add(ts_formatted)
        if ts_formatted in chunk_by_ts:
            r = chunk_by_ts[ts_formatted]
            citations.append(
                Citation(
                    episode_id=r.chunk.episode_id,
                    start_seconds=r.chunk.start_seconds,
                    end_seconds=r.chunk.end_seconds,
                    text=r.chunk.text,
                    timestamp=ts_formatted,
                )
            )
    return citations


async def generate_answer(
    query: str,
    results: list[RetrievalResult],
    llm: LLMProvider,
    model_name: str,
) -> Answer:
    context = build_context(results)
    user_prompt = build_user_prompt(query, context)

    with track_latency("llm_generate"):
        response = await llm.complete(SYSTEM_PROMPT, user_prompt)

    citations = extract_citations(response.text, results)
    log.info(
        "generated_answer",
        citation_count=len(citations),
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )

    return Answer(
        text=response.text,
        citations=citations,
        model=model_name,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
