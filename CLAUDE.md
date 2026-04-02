# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Run development server
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_main.py

# Run a single test
uv run pytest tests/test_main.py::test_root

# Run evaluation (standalone, no server required)
uv run python evaluation/run_eval.py --level L1   # 빠른 검증 (5건)
uv run python evaluation/run_eval.py --level L2   # 전체 평가 (15건)

# Lint
uv run ruff check .

# Format
uv run black .
```

## Environment Setup

Copy `env.sample` to `.env` and set:
- `OPENAI_API_KEY` — required
- `OPENAI_MODEL` — default `gpt-4o`
- `DEEPAGENT_RECURSION_LIMIT` — max agent recursion, default `20`
- `ES_URL` — Elasticsearch URL, default `http://localhost:9200`
- `ES_USERNAME` / `ES_PASSWORD` — ES basic auth credentials
- `ES_INDEX_PREFIX` — index name prefix to avoid conflicts on shared clusters, default `dev`
- `OPIK__URL_OVERRIDE` — Opik self-hosted URL (e.g. `https://your-opik-host/api`)
- `OPIK__PROJECT` — Opik project name
- `OPIK__WORKSPACE` — Opik workspace name, default `default`
- `ES_RERANKER_INFERENCE_ID` — ES Inference API rerank endpoint ID (e.g. `.rerank-v1-elasticsearch`); optional, fallback to score sort if unset

## Architecture

**FastAPI + LangChain educational template** for building streaming AI agents.

### Request Flow

```
POST /api/v1/chat {thread_id, message}
  → chat.py route
  → agent_service.py (async streaming)
  → agent (LangGraph-compatible interface)
  → SSE stream: data: {step, content, metadata}\n\n
```

SSE steps: `model` (tool decision) → `tools` (tool results) → `done` (final answer with metadata).

### Key Layers

- **`app/api/routes/`** — FastAPI endpoints: `chat.py` (streaming SSE), `threads.py` (conversation history)
- **`app/services/`** — Business logic: `agent_service.py` (LLM orchestration), `conversation_service.py` (in-memory history), `threads_service.py` (JSON data access)
- **`app/agents/`** — Agent implementations: `stock_agent.py` (LangGraph ReAct agent), `tools/` (yfinance real-time tools + `_rag_common.py` ES/OpenAI singletons), `es_tools.py` (Elasticsearch historical tools), `sec_search_agent.py` (LangGraph StateGraph subagent: BM25+kNN fan-out → merge → ES rerank, exposed as `search_sec_filing` tool), `prompts.py` (system prompts)
- **`scripts/`** — Offline data pipelines: `ingest_10k.py` (SEC EDGAR → section extraction → tiktoken chunking → OpenAI embedding → ES bulk upsert)
- **`app/elasticsearch/`** — Elasticsearch integration: `client.py` (singleton client), `ingester.py` (yfinance → ES bulk upsert, runs on startup), `retriever.py` (ElasticsearchRetriever + document_mapper)
- **`app/core/config.py`** — Pydantic-Settings config loaded from `.env` (nested via `__` delimiter)
- **`app/data/`** — JSON-based persistence: `threads.json` (index), `threads/{thread_id}.json` (messages), `favorite_questions.json`

### Extending the Agent

The `dummy.py` agent is a mock that echoes input. Replace or extend it in `app/agents/` with a real LangGraph graph and register it in `agent_service.py`. The service layer handles streaming and error handling — agent implementations only need to yield state chunks.

### Conversation Memory

`conversation_service.py` keeps conversation state in-memory (keyed by `thread_id`). For persistence, swap the in-memory store for a database backend while preserving the same interface.

### Logger Utility

`app/utils/logger.py` provides a `@log` decorator that works on sync functions, async functions, sync generators, and async generators. It measures execution time and logs start/end/error events automatically.
