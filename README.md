# 주식 분석 AI Agent

FastAPI + LangGraph 기반의 주식 분석 특화 AI Agent 서버입니다.
자연어로 주식 정보를 질문하면 yfinance 도구를 활용해 실시간 데이터를 조회하고 분석 결과를 스트리밍으로 반환합니다.

## 기술 스택

- **FastAPI** — API 서버
- **LangChain / LangGraph** — ReAct 에이전트 및 대화 흐름 관리
- **OpenAI GPT-4** — LLM
- **yfinance** — 실시간 주식 데이터 조회
- **Elasticsearch** — OHLCV 히스토리컬 데이터 저장 및 검색 / SEC 10-K RAG (BM25 + kNN + ES rerank)
- **uv** — 패키지 관리

## 에이전트 기능

| Tool | 데이터 소스 | 설명 | 반환 예시 |
|---|---|---|---|
| `get_stock_price` | yfinance (실시간) | 현재 주가 및 전일 대비 등락률 | `AAPL 현재가: $260.83 \| 등락률: +0.37%` |
| `get_company_info` | yfinance (실시간) | 시가총액, PER, 업종 | `AAPL \| 시가총액: 3.83조 달러 \| PER: 33.02 \| 업종: Technology` |
| `get_recent_news` | yfinance (실시간) | 최근 뉴스 최대 3건 (제목 + 링크) | — |
| `get_stock_history` | Elasticsearch | OHLCV 히스토리컬 데이터 조회 (지원 종목: AAPL, MSFT, TSLA, NVDA) | — |
| `search_sec_filing` | Elasticsearch (10-K RAG) | SEC 10-K 공시 기반 정성 정보 검색 — BM25 + kNN 병렬 검색 후 ES rerank (지원 종목: AAPL, MSFT, TSLA, NVDA) | — |

- 여러 tool을 조합한 복합 질문 처리 (예: "AAPL 주가랑 최근 뉴스 알려줘")
- thread_id 기반 멀티턴 대화 (대화 이력 유지)
- 주식/금융 외 질문은 답변하지 않음

## 환경 준비 및 설치 가이드

본 프로젝트는 파이썬 패키지 매니저로 **`uv`** 를 사용합니다.

### 1. 사전 요구사항

- Python 3.11 이상 3.13 이하
- `uv` 패키지 매니저 설치:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. 의존성 설치

```bash
uv sync
```

완료 시 프로젝트 디렉토리에 `.venv` 폴더가 생성됩니다.

### 3. 환경 변수 설정

```bash
cp env.sample .env
```

`.env` 파일을 열고 아래 값을 입력합니다:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o

# Elasticsearch
ES_URL=https://your-elasticsearch-host
ES_USERNAME=elastic
ES_PASSWORD=your_es_password
ES_INDEX_PREFIX=dev

# SEC 10-K RAG (선택 — 설정 시 search_sec_filing 도구에서 리랭킹 활성화)
ES_RERANKER_INFERENCE_ID=.rerank-v1-elasticsearch
```

서버 시작 시 Elasticsearch에 4개 종목(AAPL, MSFT, TSLA, NVDA)의 1년치 OHLCV 데이터가 자동으로 적재됩니다.

SEC 10-K 공시 데이터는 별도 스크립트로 적재합니다:
```bash
uv run python scripts/ingest_10k.py
```

### 4. 서버 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

서버 실행 후 `http://localhost:8000/docs` 에서 API 문서를 확인할 수 있습니다.

## 아키텍처 흐름

### 질문 → 응답 전체 흐름

```
사용자 질문 (POST /api/v1/chat)
│
▼
┌─────────────────────────────────────────────────────────┐
│  chat.py (FastAPI Route)                                │
│  AgentService.process_query()                           │
│  → agent.astream(..., stream_mode="updates")            │
└────────────────────────┬────────────────────────────────┘
                         │  SSE 스트리밍
                         ▼
┌─────────────────────────────────────────────────────────┐
│  【메인 에이전트】 LangChain ReAct (create_agent)         │
│                                                         │
│  ┌─ model 스텝 ──────────────────────────────────────┐  │
│  │  ChatOpenAI(GPT-4o) + system_prompt               │  │
│  │  → 어떤 도구를 쓸지 결정                           │  │
│  │  → tool_calls: [도구명, args]                     │  │
│  └───────────────────────────────────────────────────┘  │
│            │                                            │
│            ▼  (도구가 ChatResponse면 → done 이벤트)     │
│  ┌─ tools 스텝 ──────────────────────────────────────┐  │
│  │  [일반 도구]                [서브 에이전트 도구]   │  │
│  │  get_stock_price            search_sec_filing      │  │
│  │  get_company_info     ────▶ (LangGraph로 위임)     │  │
│  │  get_recent_news                                   │  │
│  │  get_stock_history                                 │  │
│  └───────────────────────────────────────────────────┘  │
│            │                                            │
│            └── 도구 결과를 messages에 추가 → 다시 model │
└─────────────────────────────────────────────────────────┘
                         │
         search_sec_filing 호출 시
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  【서브 에이전트】 LangGraph StateGraph                   │
│  (subagent-as-tool 패턴 — @tool로 래핑됨)               │
│                                                         │
│       START                                             │
│       /    \        ← fan-out (병렬 실행)               │
│      ▼      ▼                                           │
│  bm25_    vector_                                       │
│  search   search                                        │
│  (ES      (OpenAI embed → ES kNN)                       │
│  match)                                                 │
│      \    /         ← fan-in (둘 다 완료 후 진입)       │
│       ▼  ▼                                              │
│  merge_results                                          │
│  (_id 기준 중복 제거, 높은 score 채택)                  │
│       │                                                 │
│       ▼                                                 │
│     rerank                                              │
│  (seen_ids 제외 → ES Inference API / score 내림차순 fallback) │
│  → 상위 5개 청크 추출 → result + seen_ids 반환          │
│       │                                                 │
│      END                                                │
└─────────────────────────────────────────────────────────┘
                         │
             메인 에이전트 messages에 추가
                         │
                         ▼
             ChatOpenAI → ChatResponse 도구 호출
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  SSE 이벤트 스트림 (클라이언트 수신)                     │
│                                                         │
│  {"step": "model",  "tool_calls": ["get_stock_price"]}  │
│  {"step": "tools",  "name": "get_stock_price", ...}     │
│  {"step": "model",  "tool_calls": ["ChatResponse"]}     │
│  {"step": "done",   "message_id": "...", "content": "..."} │
└─────────────────────────────────────────────────────────┘
```

### 메인 에이전트 vs 서브 에이전트

| 구분 | 메인 에이전트 | 서브 에이전트 |
|---|---|---|
| 프레임워크 | LangChain `create_agent()` | LangGraph `StateGraph` |
| 흐름 제어 | LLM이 동적으로 도구 선택 (ReAct) | 개발자가 노드·엣지를 정적 정의 |
| 병렬 실행 | 불가 | BM25 + kNN fan-out |
| 연결 방식 | — | `@tool` 래핑 → 메인 에이전트 tools 목록에 등록 |
| LLM 사용 | GPT-4o (질문 해석 + 최종 답변) | 없음 (검색·정렬 처리만) |
| 대화 이력 | MemorySaver (`thread_id` 기반 전체 대화) | `_seen_ids_cache` (`thread_id:ticker` 기반 seen_ids만 보존) |

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|---|---|---|
| `GET` | `/` | API 정보 |
| `GET` | `/health` | 헬스 체크 |
| `GET` | `/api/v1/threads` | 최근 대화 목록 |
| `GET` | `/api/v1/threads/{thread_id}` | 대화 상세 내역 |
| `POST` | `/api/v1/chat` | 에이전트 질의 (SSE 스트리밍) |

### POST /api/v1/chat

**Request**
```json
{
  "thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "message": "AAPL 현재가 알려줘"
}
```

**Response** (Server-Sent Events)
```
data: {"step": "model", "tool_calls": ["get_stock_price"]}

data: {"step": "tools", "name": "get_stock_price", "content": "AAPL 현재가: $260.83 | 등락률: +0.37%"}

data: {"step": "done", "message_id": "...", "role": "assistant", "content": "...", "metadata": {}}
```

## 프로젝트 구조

```
agent/
├── app/
│   ├── agents/
│   │   ├── tools/
│   │   │   ├── __init__.py       # yfinance 기반 실시간 tool 3종
│   │   │   └── _rag_common.py    # ES / OpenAI 싱글톤, embed_query, rerank_hits, format_hits
│   │   ├── es_tools.py           # Elasticsearch 기반 히스토리컬 tool
│   │   ├── sec_search_agent.py   # LangGraph StateGraph 서브 에이전트 (BM25+kNN → merge → rerank)
│   │   │                         # search_sec_filing @tool로 래핑 (subagent-as-tool 패턴)
│   │   ├── stock_agent.py        # LangGraph ReAct 에이전트 (search_sec_filing 포함)
│   │   └── prompts.py            # 시스템 프롬프트
│   ├── elasticsearch/
│   │   ├── client.py         # ES 클라이언트 싱글턴
│   │   ├── ingester.py       # yfinance → ES bulk upsert (앱 시작 시 실행)
│   │   └── retriever.py      # ElasticsearchRetriever + document_mapper
│   ├── api/routes/
│   │   ├── chat.py           # 스트리밍 채팅 엔드포인트
│   │   └── threads.py        # 대화 이력 엔드포인트
│   ├── core/
│   │   └── config.py         # 환경 변수 설정
│   ├── services/
│   │   └── agent_service.py  # 에이전트 실행 및 스트리밍 처리
│   └── main.py               # FastAPI 앱 진입점 (lifespan으로 ES 적재)
├── scripts/
│   └── ingest_10k.py         # SEC EDGAR 다운로드 → 섹션 추출 → 청킹 → 임베딩 → ES 적재
├── evaluation/
│   ├── data/
│   │   └── dataset.json      # 평가 질문 15건 (expected_output 포함)
│   ├── metrics/
│   │   ├── stock_hallucination.py     # 환각 감지 메트릭
│   │   ├── stock_answer_relevance.py  # 답변 관련성 메트릭
│   │   └── stock_task_completion.py   # 작업 완료 메트릭 (GEval 기반)
│   └── run_eval.py           # Opik 평가 실행 진입점 (서버 불필요)
├── docs/
│   ├── spec.md               # API 명세
│   └── daily-record/         # 개발 일지
├── tests/
├── env.sample
└── pyproject.toml
```
