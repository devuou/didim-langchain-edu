# 주식 분석 AI Agent

FastAPI + LangGraph 기반의 주식 분석 특화 AI Agent 서버입니다.
자연어로 주식 정보를 질문하면 yfinance 도구를 활용해 실시간 데이터를 조회하고 분석 결과를 스트리밍으로 반환합니다.

## 기술 스택

- **FastAPI** — API 서버
- **LangChain / LangGraph** — ReAct 에이전트 및 대화 흐름 관리
- **OpenAI GPT-4** — LLM
- **yfinance** — 주식 데이터 조회
- **uv** — 패키지 관리

## 에이전트 기능

| Tool | 설명 | 반환 예시 |
|---|---|---|
| `get_stock_price` | 현재 주가 및 전일 대비 등락률 | `AAPL 현재가: $260.83 \| 등락률: +0.37%` |
| `get_company_info` | 시가총액, PER, 업종 | `AAPL \| 시가총액: 3.83조 달러 \| PER: 33.02 \| 업종: Technology` |
| `get_recent_news` | 최근 뉴스 최대 3건 (제목 + 링크) | — |

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
```

### 4. 서버 실행

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

서버 실행 후 `http://localhost:8000/docs` 에서 API 문서를 확인할 수 있습니다.

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
│   │   ├── tools.py          # yfinance 기반 tool 3종
│   │   ├── stock_agent.py    # LangGraph ReAct 에이전트
│   │   └── prompts.py        # 시스템 프롬프트
│   ├── api/routes/
│   │   ├── chat.py           # 스트리밍 채팅 엔드포인트
│   │   └── threads.py        # 대화 이력 엔드포인트
│   ├── core/
│   │   └── config.py         # 환경 변수 설정
│   ├── services/
│   │   └── agent_service.py  # 에이전트 실행 및 스트리밍 처리
│   └── main.py               # FastAPI 앱 진입점
├── docs/
│   ├── spec.md               # API 명세
│   └── daily-record/         # 개발 일지
├── tests/
├── env.sample
└── pyproject.toml
```
