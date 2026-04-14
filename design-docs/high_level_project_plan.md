# PodSearch — Project Plan

## What PodSearch Is

PodSearch is a semantic search engine over podcast transcripts. A user enters a natural-language query — *"what did the guest say about hallucinations?"* — and gets back a generated answer with clickable timestamp citations that jump to the exact moment in the episode.

The system is a Retrieval-Augmented Generation (RAG) application built on a hybrid retriever (semantic + lexical), a cross-encoder reranker, and a grounded LLM generator that produces cited answers. Every retrieved chunk carries timestamp metadata so citations resolve to a real `[MM:SS]` location in the source episode.

## Current State

- FastAPI app at `backend/app/main.py` with a stub `GET /search` doing substring matching against `backend/data/transcript.txt`.
- A scaffolded folder tree under `backend/app/` (`api/`, `core/`, `service/`, `models/`) — directories exist, files are mostly empty.
- One sample transcript with `[MM:SS]` timestamps (10 lines).
- Empty `frontend/` folder.
- `.env.example` with placeholders for OpenAI, Anthropic, and Chroma.
- No tests, no ingestion pipeline, no vector store wiring.

## Stack

| Layer            | Choice                          | Notes                                                    |
|------------------|---------------------------------|----------------------------------------------------------|
| API              | FastAPI                         | Async, typed, matches existing scaffold                  |
| Config           | pydantic-settings               | Typed env loading, fail-fast at startup                  |
| Embeddings       | OpenAI `text-embedding-3-small` | 1536-d, ABC-wrapped for swap                             |
| Vector DB        | ChromaDB (persisted local)      | Zero-ops dev; ABC allows Qdrant later                    |
| LLM              | Claude (Anthropic)              | Strong instruction following, native streaming           |
| Lexical search   | `rank_bm25`                     | In-memory, pure Python                                   |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CPU-friendly, standard baseline                  |
| Frontend         | Vite + React + TypeScript       | Single-page UI                                           |
| Observability    | structlog + custom telemetry    | Structured logs, request IDs                             |
| Tests            | pytest + pytest-asyncio         |                                                          |

---

## Target Architecture

The end-state folder structure (built up across phases):

```
backend/
├── app/
│   ├── main.py                         # FastAPI app + lifespan (wires DI, loads providers)
│   ├── api/
│   │   ├── deps.py                     # FastAPI Depends() wiring for singletons
│   │   ├── health.py                   # GET /health, GET /ready
│   │   ├── search.py                   # POST /search, POST /search/stream
│   │   ├── ingest.py                   # POST /ingest
│   │   └── metrics.py                  # GET /metrics (Phase 5)
│   ├── core/
│   │   ├── config.py                   # Settings via pydantic-settings (env-driven)
│   │   ├── logging.py                  # Structured JSON logging + request IDs
│   │   ├── telemetry.py                # Latency timers, token/cost counters
│   │   ├── pricing.py                  # Per-model token costs (Phase 5)
│   │   ├── caching.py                  # Embedding + answer caches (Phase 5)
│   │   └── security.py                 # Rate limit, API key, input sanitize (Phase 5)
│   ├── models/
│   │   ├── domain.py                   # Internal: Chunk, Citation, RetrievalResult
│   │   └── api.py                      # Public Pydantic request/response schemas
│   ├── providers/                      # Swappable backends — the "plug" boundary
│   │   ├── embeddings/
│   │   │   ├── base.py                 # EmbeddingProvider ABC
│   │   │   ├── openai.py               # text-embedding-3-small impl
│   │   │   └── local_st.py             # sentence-transformers impl (Phase 2)
│   │   ├── vector_store/
│   │   │   ├── base.py                 # VectorStore ABC
│   │   │   ├── chroma.py               # ChromaDB impl
│   │   │   └── qdrant.py               # Qdrant impl (Phase 5 stretch)
│   │   ├── llm/
│   │   │   ├── base.py                 # LLMProvider ABC (complete + stream)
│   │   │   └── anthropic.py            # Claude impl
│   │   └── factory.py                  # Build providers from Settings — the only
│   │                                   # file that imports concrete SDKs
│   ├── services/                       # Business logic — depends on provider ABCs only
│   │   ├── transcript_parser.py        # [MM:SS] → TranscriptLine objects
│   │   ├── chunker.py                  # Timestamp-aware, overlapping, configurable
│   │   ├── ingestion.py                # parse → chunk → embed → upsert pipeline
│   │   ├── query_rewriter.py           # Optional LLM query expansion (Phase 3)
│   │   ├── retrieval/
│   │   │   ├── base.py                 # Retriever ABC
│   │   │   ├── semantic.py             # Vector search
│   │   │   ├── bm25.py                 # In-memory BM25 (Phase 2)
│   │   │   ├── hybrid.py               # Reciprocal Rank Fusion (Phase 2)
│   │   │   ├── reranker.py             # Cross-encoder reranker (Phase 3)
│   │   │   └── pipeline.py             # Composable retriever wrapper (Phase 3)
│   │   ├── generation.py               # Prompt build, LLM call, citation extraction
│   │   └── groundedness.py             # Faithfulness check (Phase 4)
│   └── evaluation/
│       ├── dataset.py                  # Loads eval query set
│       ├── retrieval_metrics.py        # hit@k, MRR, NDCG
│       └── answer_quality.py           # LLM-as-judge faithfulness scoring
├── data/
│   ├── transcripts/                    # Raw episode .txt files
│   ├── chroma_db/                      # Persisted vector DB (gitignored)
│   └── eval/
│       └── queries.yaml                # Labeled eval queries
├── scripts/
│   ├── ingest_folder.py                # Batch ingest CLI
│   ├── run_eval.py                     # Eval pipeline CLI
│   └── compute_stats.py                # Telemetry rollup CLI (Phase 5)
└── tests/
    ├── unit/                           # Fast, fake providers
    └── integration/                    # End-to-end on tmp Chroma

frontend/                               # Vite + React + TS (Phase 4)
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── api/client.ts                   # Typed fetch + SSE wrapper
    ├── types.ts                        # Mirrors backend api.py
    └── components/
        ├── SearchBar.tsx
        ├── AnswerView.tsx              # Streaming tokens + inline citation pills
        ├── CitationList.tsx            # Clickable [MM:SS] cards
        └── EpisodePlayer.tsx           # <audio> seek-to-timestamp (stretch)
```

### Architectural Principles

1. **`providers/` is the only layer that imports SDKs.** Business logic in `services/` depends on ABCs (`EmbeddingProvider`, `VectorStore`, `LLMProvider`), never on `openai`, `chromadb`, or `anthropic` directly. Replacing OpenAI embeddings with local sentence-transformers means adding one file and flipping one env var — no service code changes.

2. **`services/` vs `providers/` is business logic vs infrastructure.** Services own *decisions* (how to chunk, how to fuse hybrid scores, what goes in a prompt); providers own *mechanics* (how to call an API). Services are unit-testable against in-memory fakes with zero network.

3. **`providers/factory.py` is the single composition root.** No module-level `openai_client = OpenAI()` anywhere else. Every other file gets providers through function arguments or FastAPI `Depends()`.

4. **Domain models vs API models are separate.** Internal `Chunk` and `Citation` types differ from the public wire format. This prevents API contracts from being coupled to internal data shapes.

5. **FastAPI lifespan + `Depends()` for singletons.** Vector store connections, loaded models, and LLM clients are created once at startup via `lifespan`, stored on `app.state`, and injected via `Depends()`. No cold start on first request, fully testable.

6. **Configuration via `pydantic-settings`.** A single `Settings` class reads all env vars with types and defaults. Misconfiguration fails at startup.

7. **No LangChain or LlamaIndex.** Every abstraction is hand-written. The codebase is small, transparent, and has no hidden behavior.

8. **Structured logging with request IDs from day one.** Every log line is JSON with a `request_id` propagated through retrieval and generation. Cheap to add early, painful to retrofit.

9. **Evaluation is a first-class module.** `evaluation/` lives in the app package and imports real services. Built in Phase 2 so every later phase is measured, not guessed at.

10. **Chunks carry timestamp metadata from the first line of ingest code.** `start_seconds`, `end_seconds`, and `episode_id` ride into Chroma alongside every embedding. Citations are not an afterthought.

---

## Phase 1 — Minimum Viable RAG Loop

Prove the full RAG pipeline works end-to-end with the simplest possible implementation of every stage.

### Scope
- ABCs for `EmbeddingProvider`, `VectorStore`, `LLMProvider`.
- Concrete impls: `OpenAIEmbeddingProvider`, `ChromaVectorStore`, `AnthropicLLMProvider`.
- Naive chunker: one transcript line → one chunk (intentional baseline).
- Transcript parser for `[MM:SS]` format.
- `POST /ingest` and `POST /search` endpoints.
- Ingest CLI script.
- Structured logging with request IDs.
- Unit tests with fake providers, integration test on tmp Chroma.

### Files

| File | Status | Purpose |
|---|---|---|
| `backend/requirements.txt` | rewrite | `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `openai`, `anthropic`, `chromadb`, `python-dotenv`, `structlog`, `pyyaml` |
| `backend/app/core/config.py` | fill in | `Settings(BaseSettings)` reading every env var; cached via `lru_cache` |
| `backend/app/core/logging.py` | new | `configure_logging()` + request-ID middleware helper |
| `backend/app/core/telemetry.py` | new | `track_latency` context manager (expanded later) |
| `backend/app/models/domain.py` | new | `TranscriptLine`, `Chunk`, `RetrievalResult`, `Citation`, `Answer` |
| `backend/app/models/api.py` | new | `SearchRequest`, `SearchResponse`, `IngestRequest`, `IngestResponse` |
| `backend/app/providers/embeddings/base.py` | new | `EmbeddingProvider` ABC: `embed`, `embed_query`, `dim` |
| `backend/app/providers/embeddings/openai.py` | new | `OpenAIEmbeddingProvider` with batching + retries |
| `backend/app/providers/vector_store/base.py` | new | `VectorStore` ABC: `upsert`, `query`, `count`, `delete` |
| `backend/app/providers/vector_store/chroma.py` | new | `ChromaVectorStore` persistent client; stores timestamp metadata |
| `backend/app/providers/llm/base.py` | new | `LLMProvider` ABC: `complete` returning `LLMResponse(text, in_tok, out_tok)` |
| `backend/app/providers/llm/anthropic.py` | new | `AnthropicLLMProvider` |
| `backend/app/providers/factory.py` | new | `build_embedding_provider`, `build_vector_store`, `build_llm_provider` |
| `backend/app/services/transcript_parser.py` | new | Regex parser for `[MM:SS]` lines |
| `backend/app/services/chunker.py` | fill in | `chunk_naive(lines, episode_id)` — one chunk per line |
| `backend/app/services/ingestion.py` | new | `ingest_text(episode_id, raw_text, ...)` pipeline |
| `backend/app/services/retrieval/semantic.py` | new | `SemanticRetriever(embedder, store)` |
| `backend/app/services/generation.py` | fill in | `build_prompt`, `generate_answer`; system prompt enforces grounding + citations |
| `backend/app/api/deps.py` | new | `get_settings`, `get_embedder`, `get_store`, `get_llm`, `get_retriever` |
| `backend/app/api/health.py` | new | `GET /health` and `GET /ready` (checks `store.count()`) |
| `backend/app/api/search.py` | fill in | `POST /search` orchestrating retrieve → generate |
| `backend/app/api/ingest.py` | fill in | `POST /ingest` accepting `{episode_id, text}` |
| `backend/app/main.py` | rewrite | `lifespan` builds providers; mounts routers; deletes the stub `/search` |
| `backend/scripts/ingest_folder.py` | new | CLI: ingest all `data/transcripts/*.txt` |
| `backend/data/transcripts/episode_001.txt` | move | Renamed from existing `data/transcript.txt` |
| `backend/tests/unit/test_transcript_parser.py` | new | |
| `backend/tests/unit/test_chunker_naive.py` | new | |
| `backend/tests/unit/test_semantic_retriever.py` | new | Uses fake providers |
| `backend/tests/integration/test_search_endpoint.py` | new | End-to-end on tmp Chroma |
| `.env.example` | update | Add `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `LLM_PROVIDER`, `LLM_MODEL`, `CHROMA_COLLECTION`, `TOP_K` |

### Milestone

```bash
python -m backend.scripts.ingest_folder backend/data/transcripts/
uvicorn backend.app.main:app --reload
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "what is RAG?"}'
```

Returns a Claude-generated answer with a `citations` array of real `[MM:SS]` timestamps. `pytest backend/tests/` is green. Logs print structured JSON with request IDs.

---

## Phase 2 — Hybrid Retrieval and Evaluation

Add the lexical retriever, hybrid fusion, metadata filtering, a second embedding provider, and the evaluation harness that will measure every change from this phase forward.

### Scope
- `BM25Retriever` — in-memory `rank_bm25` index, rebuilt at startup and on ingest.
- `HybridRetriever` — semantic + BM25 merged via Reciprocal Rank Fusion (`k=60`).
- `Retriever` ABC unifying all three modes; selectable via `RETRIEVER_MODE` env var.
- Metadata filtering (`episode_id`, time range) plumbed through the ABC and Chroma.
- `LocalSTEmbeddingProvider` (sentence-transformers) — added now to validate the embedding abstraction works under a real swap.
- Evaluation harness: YAML-labeled query set, retrieval metrics, comparison CLI.

### Files

| File | Status | Purpose |
|---|---|---|
| `backend/requirements.txt` | update | Add `rank_bm25`, `sentence-transformers`, `numpy` |
| `backend/app/providers/embeddings/local_st.py` | new | `LocalSTEmbeddingProvider`, `dim=384` |
| `backend/app/providers/factory.py` | update | Branch on `EMBEDDING_PROVIDER` |
| `backend/app/providers/vector_store/base.py` | update | `query()` accepts `filters: dict \| None` |
| `backend/app/providers/vector_store/chroma.py` | update | Translate domain filters to Chroma `where` clause |
| `backend/app/services/retrieval/base.py` | new | `Retriever` ABC |
| `backend/app/services/retrieval/semantic.py` | update | Implement new ABC |
| `backend/app/services/retrieval/bm25.py` | new | `BM25Retriever` with `rebuild(chunks)` |
| `backend/app/services/retrieval/hybrid.py` | new | `HybridRetriever(semantic, bm25, k_rrf=60)` with RRF |
| `backend/app/api/deps.py` | update | `get_retriever()` returns configured retriever |
| `backend/app/api/search.py` | update | Accepts `filters` and `retriever_mode` in request |
| `backend/app/models/api.py` | update | Add `filters`, `retriever_mode` to `SearchRequest` |
| `backend/app/evaluation/__init__.py` | new | Package marker |
| `backend/app/evaluation/dataset.py` | new | `load_eval_set(path)` |
| `backend/app/evaluation/retrieval_metrics.py` | new | `hit_at_k`, `mrr`, `ndcg_at_k` (pure functions) |
| `backend/data/eval/queries.yaml` | new | 15–25 hand-labeled queries |
| `backend/scripts/run_eval.py` | new | CLI: runs all retrievers, prints markdown comparison table |
| `backend/tests/unit/test_retrieval_metrics.py` | new | |
| `backend/tests/unit/test_rrf_fusion.py` | new | |
| `backend/tests/unit/test_bm25_retriever.py` | new | |

### Milestone

`python -m backend.scripts.run_eval` prints a table comparing semantic, BM25, and hybrid on hit@5, MRR, NDCG@10. `POST /search` with `{"retriever_mode": "hybrid", "filters": {"episode_id": "ep_001"}}` works. Setting `EMBEDDING_PROVIDER=local_st` keeps the system fully functional with no code changes.

---

## Phase 3 — Smart Chunking and Reranking

Replace the naive chunker, add cross-encoder reranking and optional query rewriting, and use the eval harness to prove each change improves retrieval quality.

### Scope
- **Timestamp-overlap chunker**: configurable `target_tokens` (~200) + `overlap_tokens` (~40), boundaries snapped to transcript lines, every chunk carries `start_s`/`end_s`.
- **Cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`): retrieve top-50 via hybrid, rerank to top-5.
- **`RerankedRetriever` decorator** — wraps any `Retriever` with a reranker stage. Composable.
- **Query rewriter**: short Claude call expanding terse queries; gated behind env flag.
- **Chunk-strategy A/B**: eval iterates over `naive`, `fixed_window`, `timestamp_overlap`.
- README "architecture of a query" section tracing a real request through every stage.

### Files

| File | Status | Purpose |
|---|---|---|
| `backend/requirements.txt` | update | Add `tiktoken` |
| `backend/app/services/chunker.py` | rewrite | Add `chunk_timestamp_overlap`; expose `build_chunker(strategy)` factory |
| `backend/app/core/config.py` | update | Add `CHUNKING_STRATEGY`, `CHUNK_TARGET_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `RERANKER_ENABLED`, `RERANKER_FETCH_K`, `QUERY_REWRITE_ENABLED` |
| `backend/app/services/retrieval/reranker.py` | new | `CrossEncoderReranker` with `rerank(query, results, top_k)` |
| `backend/app/services/retrieval/pipeline.py` | new | `RerankedRetriever(inner, reranker, fetch_k, top_k)` |
| `backend/app/services/query_rewriter.py` | new | `QueryRewriter(llm)`; falls back to original query on failure |
| `backend/app/api/search.py` | update | Pipeline becomes `rewrite? → retrieve → rerank → generate` |
| `backend/scripts/run_eval.py` | update | Adds chunking strategy × retriever mode grid; toggles reranker and rewrite |
| `backend/tests/unit/test_chunker_overlap.py` | new | |
| `backend/tests/unit/test_reranker.py` | new | Fake `CrossEncoder` |
| `backend/tests/integration/test_search_pipeline.py` | new | End-to-end with full pipeline |

### Milestone

Eval shows timestamp-overlap chunking + hybrid + reranking beats the Phase 2 baseline on hit@5 and MRR. The `/search` endpoint returns visibly better answers on ambiguous queries. Corpus contains at least three episodes for diversity.

---

## Phase 4 — Grounded Generation and Frontend

Build the generation half of the system properly — citation validation, hallucination defenses, streaming — then ship the React UI that turns the backend into a demoable product.

### Scope
- **Advanced generation prompt**: explicit grounding rules, citation format, "say so if no answer is supported".
- **Deterministic citation extraction**: parse `[MM:SS]` from LLM output, validate each against retrieved chunks, drop hallucinated timestamps.
- **Groundedness check** (env-gated): second LLM call scoring faithfulness; failures flagged as low-confidence.
- **Streaming via SSE**: token-by-token `POST /search/stream`.
- **LLM-as-judge eval**: faithfulness, relevance, citation accuracy added to the eval harness.
- **React + TypeScript frontend**: search bar, streaming answer view with inline citation pills, citation cards. Stretch: audio player that seeks to citation timestamp.

### Backend Files

| File | Status | Purpose |
|---|---|---|
| `backend/app/services/generation.py` | rewrite | New grounded prompt; `stream_answer()`; `extract_and_validate_citations()` |
| `backend/app/services/groundedness.py` | new | `check_groundedness(answer, chunks, llm)` returns `FaithfulnessScore` |
| `backend/app/providers/llm/base.py` | update | Add `stream(...)` returning `AsyncIterator[str]` |
| `backend/app/providers/llm/anthropic.py` | update | Implement streaming via `messages.stream` |
| `backend/app/api/search.py` | update | Add `POST /search/stream` (SSE `StreamingResponse`) |
| `backend/app/models/api.py` | update | `SearchResponse` gains `groundedness_score`, `model`, `retrieval_mode_used` |
| `backend/app/evaluation/answer_quality.py` | new | LLM-as-judge: faithfulness, relevance, citation accuracy |
| `backend/scripts/run_eval.py` | update | Adds answer-quality columns |
| `backend/tests/unit/test_citation_extraction.py` | new | Hallucinated timestamps filtered |
| `backend/tests/unit/test_groundedness.py` | new | |
| `backend/tests/integration/test_streaming.py` | new | |

### Frontend Files (all new)

| File | Purpose |
|---|---|
| `frontend/package.json` | Vite + React 18 + TypeScript |
| `frontend/vite.config.ts` | Proxies `/api/*` → `localhost:8000` |
| `frontend/tsconfig.json` | Strict mode |
| `frontend/index.html` | Mount point |
| `frontend/src/main.tsx` | React root |
| `frontend/src/App.tsx` | Layout: header, `<SearchBar />`, `<AnswerView />`, `<CitationList />` |
| `frontend/src/types.ts` | Mirrors backend `api.py` schemas |
| `frontend/src/api/client.ts` | Typed `searchStream(query, onToken, onDone)` |
| `frontend/src/components/SearchBar.tsx` | Input + submit on Enter |
| `frontend/src/components/AnswerView.tsx` | Streamed tokens + clickable inline `[MM:SS]` pills |
| `frontend/src/components/CitationList.tsx` | Cards: episode, timestamp, excerpt, jump button |
| `frontend/src/components/EpisodePlayer.tsx` (stretch) | `<audio>` with `currentTime` set on citation click |

### Milestone

`localhost:5173` shows a working UI. Type a query, watch the answer stream in token by token with `[MM:SS]` pills inline. Click a pill, the citation card scrolls into view (or audio jumps to that moment if the stretch is done). Hallucinated timestamps never appear — they're filtered post-generation. Eval reports faithfulness, relevance, and citation-accuracy scores.

---

## Phase 5 — Production Readiness

Everything that turns the working demo into something you'd put in front of real users: observability, cost/latency tracking, security, caching, CI, and deployment.

### Scope
- **Telemetry**: every query logs `{request_id, query, retrieval_mode, chunks_retrieved, rerank_ms, llm_ms, total_ms, input_tokens, output_tokens, cost_usd, groundedness_score}` as one JSON line.
- **Cost & latency rollup**: `GET /metrics` returns recent telemetry from an in-memory ring buffer; CLI computes p50/p95 latency and total cost.
- **Caching**: query-embedding LRU cache, answer cache with short TTL — both env-gated, both bypassed during eval.
- **Rate limiting + API key auth** on `/search` and `/ingest`.
- **Input guardrails**: max query length, PII regex on ingest, soft prompt-injection check.
- **Continuous eval**: `run_eval` runs in CI on every PR with a regression threshold.
- **Docker**: multi-stage `Dockerfile` + `docker-compose.yml` (backend + frontend + Chroma volume).
- **Qdrant provider** (stretch): proves the vector store abstraction survives a real swap.
- **Story-driven README**: problem, architecture, eval deltas across phases, production decisions.

### Files

| File | Status | Purpose |
|---|---|---|
| `backend/app/core/telemetry.py` | expand | `QueryTelemetry` dataclass; `TelemetryBuffer(capacity=1000)`; latency + cost decorators |
| `backend/app/core/pricing.py` | new | Per-model `COST_PER_1M_INPUT`/`OUTPUT_TOKENS`; `compute_cost(model, in, out)` |
| `backend/app/core/security.py` | new | `check_api_key`, `rate_limit` (token bucket), `sanitize_query`, `detect_prompt_injection` |
| `backend/app/core/caching.py` | new | `EmbeddingCache(LRU)`, `AnswerCache(TTL)` |
| `backend/app/api/metrics.py` | new | `GET /metrics` (API-key-gated) |
| `backend/app/api/deps.py` | update | Wires rate limiter + API key as `Depends()` |
| `backend/app/api/search.py` | update | Emits `QueryTelemetry` per request |
| `backend/app/providers/llm/anthropic.py` | update | Fills token counts and cost on `LLMResponse` |
| `backend/app/providers/vector_store/qdrant.py` | new (stretch) | Qdrant impl of the ABC |
| `backend/scripts/compute_stats.py` | new | Reads structured logs → p50/p95 latency, total cost |
| `backend/Dockerfile` | new | Multi-stage; runs uvicorn as non-root |
| `frontend/Dockerfile` | new | Builds static, serves via nginx |
| `docker-compose.yml` | new | Backend + frontend + Chroma volume |
| `.github/workflows/ci.yml` | new | `pytest` + `run_eval` with regression threshold |
| `README.md` | rewrite | Story: problem → architecture → eval deltas → production decisions |
| `backend/tests/unit/test_pricing.py` | new | |
| `backend/tests/unit/test_security.py` | new | |

### Milestone

`docker-compose up` brings the entire stack up on a fresh machine. `GET /metrics` returns real telemetry. CI passes on a clean push and fails on a deliberate eval regression. If the Qdrant stretch is done, swapping `VECTOR_STORE_TYPE=qdrant` keeps the system fully functional.

---

## Critical Files (Load-Bearing Across Phases)

- `backend/app/providers/factory.py` — only file that imports concrete SDKs
- `backend/app/providers/*/base.py` — the swap contracts
- `backend/app/services/retrieval/pipeline.py` — retriever composition root
- `backend/app/services/generation.py` — grounding + citation logic
- `backend/app/evaluation/` — ground truth for "is this better?"
- `backend/app/core/config.py` — single source of runtime truth
- `backend/data/eval/queries.yaml` — handwritten ground truth, grown each phase

## Verification Strategy

- **Unit tests** use fake providers; <1s; run on every save during dev.
- **Integration tests** spin up the app on a tmp Chroma dir with a 2-episode fixture corpus.
- **Eval script** (`run_eval.py`) is the source of truth for retrieval and answer quality. Run it at the end of every phase from Phase 2 onward; record numbers in a README "Eval history" section. Never ship a phase where metrics regress without explanation.
- **Manual browser demo** (Phase 4+) on three queries: one clear semantic win, one clear keyword win, one ambiguous — sanity check that the frontend tells the story.
