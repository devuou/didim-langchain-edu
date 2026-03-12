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
- **`app/agents/`** — Agent implementations: `dummy.py` (echo mock, used when no real agent is wired), `prompts.py` (system prompts)
- **`app/core/config.py`** — Pydantic-Settings config loaded from `.env` (nested via `__` delimiter)
- **`app/data/`** — JSON-based persistence: `threads.json` (index), `threads/{thread_id}.json` (messages), `favorite_questions.json`

### Extending the Agent

The `dummy.py` agent is a mock that echoes input. Replace or extend it in `app/agents/` with a real LangGraph graph and register it in `agent_service.py`. The service layer handles streaming and error handling — agent implementations only need to yield state chunks.

### Conversation Memory

`conversation_service.py` keeps conversation state in-memory (keyed by `thread_id`). For persistence, swap the in-memory store for a database backend while preserving the same interface.

### Logger Utility

`app/utils/logger.py` provides a `@log` decorator that works on sync functions, async functions, sync generators, and async generators. It measures execution time and logs start/end/error events automatically.
