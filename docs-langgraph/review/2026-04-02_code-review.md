# 에이전트 코드 평가 보고서
날짜: 2026-04-02

## 종합 평가

LangGraph StateGraph 기반의 fan-out/fan-in 패턴, subagent-as-tool 설계, lazy init 싱글톤 등 핵심 아이디어는 올바르게 구현되어 있다. 다만 `create_agent` / `ToolStrategy`가 표준 LangChain API가 아닌 내부 라이브러리(DeepAgents)임에도 표준인 것처럼 문서화된 점, `agent_service.py`에 JSON 인젝션 버그와 인스턴스별 `progress_queue` 설계 문제가 존재하며, 개선이 필요하다.

---

## 1. 메인 에이전트 (stock_agent.py / prompts.py)

### 잘된 점

- **`@tool` 데코레이터** (`tools/__init__.py:5,35,71`, `es_tools.py:8`): `langchain_core.tools.tool` 사용, docstring에 Args 섹션 포함 — 공식 문서 권장 패턴 정확히 준수
- **응답 강제 이중 방어**: 시스템 프롬프트에서 "반드시 ChatResponse 도구를 호출" 명시 + `ToolStrategy(ChatResponse)` 코드 레벨 강제 — 신뢰성 있는 구조
- **팩토리 함수 분리** (`stock_agent.py:39`): 모델과 checkpointer를 외부 주입받아 테스트 용이성 확보

### 개선 필요 사항

**[심각] `create_agent` / `ToolStrategy`는 표준 LangChain API가 아님**

```python
# stock_agent.py:6-7
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
```

공식 LangChain 패키지에 이 심볼들은 존재하지 않는다. `agent_service.py:61` 주석("DeepAgents 라이브러리")이 실체를 암시하지만 `stock_agent.py`에는 언급이 없다. 팀원이 공식 문서로 확인하려 하면 찾을 수 없다.

표준 LangGraph 방식:
```python
# 공식 권장: langgraph.prebuilt
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model, tools, prompt=system_prompt, checkpointer=checkpointer)
```

**[보통] `checkpointer` 기본값 문서와 코드 불일치**

```python
# stock_agent.py:39,52
def create_stock_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None,):
    """...checkpointer: 대화 이력을 저장할 checkpointer (기본값: MemorySaver)"""
```

실제 기본값은 `None`이지만 docstring은 `MemorySaver`라 기재되어 있다.

**[보통] `MemorySaver` 불필요한 임포트**

```python
# stock_agent.py:10
from langgraph.checkpoint.memory import MemorySaver
```

주석 처리된 코드에서만 쓰이던 임포트가 잔존. 제거 필요.

**[경미] `dataclass` vs Pydantic for `ChatResponse`**

```python
# stock_agent.py:22-33
@dataclass
class ChatResponse:
    message_id: str
    content: str
    metadata: dict[str, Any]
```

DeepAgents `ToolStrategy`의 스키마 추출 방식에 따라 `dataclass` 사용 가능 여부가 달라진다. 표준 LangChain의 structured output은 Pydantic 모델 기반(`BaseModel`)을 권장한다.

### 권장 수정 방향

- `stock_agent.py` 상단 주석에 "이 파일은 DeepAgents 라이브러리(`create_agent`, `ToolStrategy`) 기반" 명시
- docstring의 `기본값: MemorySaver` → `기본값: None (caller가 MemorySaver를 주입)`으로 수정
- 미사용 `MemorySaver` 임포트 제거

---

## 2. 서브 에이전트 StateGraph (sec_search_agent.py)

### 잘된 점

- **`TypedDict` State 정의** (`sec_search_agent.py:34-44`): LangGraph 공식 문서 권장 방식 정확히 준수. 각 키의 역할 주석도 명확
- **fan-out/fan-in 패턴** (`sec_search_agent.py:161-168`): `add_edge(START, "bm25_search")` + `add_edge(START, "vector_search")` → 둘 다 `merge_results`로 수렴하는 구조가 공식 문서의 병렬 노드 패턴과 일치
- **subagent-as-tool 래핑** (`sec_search_agent.py:180-191`): 서브 그래프를 `@tool`로 래핑하여 메인 에이전트에서 단순 도구로 사용 — 합성(composition)의 좋은 예
- **모듈 수준 1회 컴파일** (`sec_search_agent.py:175`): 그래프 컴파일 비용을 요청마다 지불하지 않는 합리적 선택

### 개선 필요 사항

**[보통] Import 위치가 파일 중간에 존재**

```python
# sec_search_agent.py:142-144 — 노드 함수 정의 이후 등장
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
```

PEP 8 및 Python 관례상 모든 임포트는 파일 최상단에 위치해야 한다. 현재 구조는 "노드 정의 → 임포트 → 그래프 조립"으로 읽기 흐름이 끊긴다.

**[경미] `_INDEX_NAME`을 모듈 수준에서 즉시 평가**

```python
# sec_search_agent.py:48
_INDEX_NAME = f"{settings.ES_INDEX_PREFIX}-10k-docs"
```

모듈 임포트 시점에 `settings`가 평가된다. 대부분의 경우 문제없지만, 테스트에서 `ES_INDEX_PREFIX`를 오버라이드하려면 임포트 전에 환경변수를 설정해야 하는 제약이 생긴다.

**[경미] `SecSearchState`의 일부 키가 Optional이어야 함**

```python
# sec_search_agent.py:39-44
bm25_hits: list[dict]    # 그래프 초기 invoke 시 존재하지 않음
vector_hits: list[dict]
merged_hits: list[dict]
result: str
```

`invoke({"ticker": ..., "query": ...})`로 진입 시 `bm25_hits` 등은 없다. LangGraph는 런타임에 누락 키를 허용하지만, TypedDict 타입 정의와 실제 초기 상태가 불일치한다. `Annotated`로 기본값 지정하거나 `total=False` 사용을 고려할 수 있다.

### 권장 수정 방향

- `from langgraph.graph import ...`, `from langchain_core.tools import tool`을 파일 최상단으로 이동
- 초기화되지 않은 키(`bm25_hits`, `vector_hits`, `merged_hits`, `result`)에 `total=False` 또는 `NotRequired` 타입 표기 추가

---

## 3. RAG 파이프라인 (_rag_common.py)

### 잘된 점

- **lazy init 패턴** (`_rag_common.py:24-52`): asyncio 기반 단일 프로세스 환경에서 thread-safety 문제 없음. ES/OpenAI 클라이언트 연결 비용을 첫 호출 시에만 지불하는 합리적 선택
- **`rerank_hits` fallback** (`_rag_common.py:66-97`): `ES_RERANKER_INFERENCE_ID` 미설정 시 `None` 반환 → 호출부(`_rerank_fn`)에서 score 정렬 fallback — 옵셔널 기능의 깔끔한 분리
- **`_merge_results_fn` / `merge_results` 분리** (`sec_search_agent.py:98-112`): `_merge_results_fn`은 plain dict를 받아 단독 단위 테스트 가능. StateGraph 노드 래퍼(`merge_results`)와 로직 분리 — 테스트 편의성을 고려한 좋은 구조

### 개선 필요 사항

**[보통] `TYPE_CHECKING` 블록 목적이 주석과 불일치**

```python
# _rag_common.py:11-15
# TYPE_CHECKING 블록: 타입 힌트 전용 import.
# 런타임에는 실행되지 않아 무거운 라이브러리(elasticsearch, openai 등)를
# 모듈 로드 시점에 불러오지 않아도 된다.
if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from openai import OpenAI
```

이 설명은 오해를 유발한다. `TYPE_CHECKING` 블록은 실제로 런타임 임포트를 막지 않는다 — `Elasticsearch`와 `OpenAI` 런타임 임포트는 이미 각 함수 body 안에 있다 (`_rag_common.py:26,49`). `TYPE_CHECKING` 블록의 임포트는 **문자열 어노테이션**(`"Elasticsearch | None"`)의 타입 체커(mypy/pyright) 해석을 위한 것이다. 주석을 사실에 맞게 수정해야 한다.

**[보통] `rerank_hits` 예외를 조용히 삼킴**

```python
# _rag_common.py:96
except Exception:
    return None
```

어떤 오류가 발생했는지(ES 연결 실패, 인증 오류, API 형식 변경 등) 전혀 알 수 없다. 최소한 `logger.warning`이라도 남겨야 운영 시 디버깅이 가능하다.

**[경미] `embed_query`에 오류 처리 없음**

`embed_query`(`_rag_common.py:55-63`)는 OpenAI API 실패 시 예외를 그대로 전파한다. `vector_search` 노드에서 catch하지 않으면 그래프 전체가 실패한다. `bm25_search`는 성공했더라도 결과를 활용하지 못하게 된다.

### 권장 수정 방향

- `TYPE_CHECKING` 블록 주석을 "타입 체커를 위한 전방 참조 임포트, 런타임에는 실행 안 됨"으로 정확히 기재
- `rerank_hits` except에 `logger.warning("rerank failed, falling back: %s", exc)` 추가
- `vector_search` 노드에 `try/except` 추가하여 임베딩 실패 시 빈 hits 반환 처리

---

## 4. 코드 품질 전반

### 잘된 점

- `langchain_core.tools.tool` 사용으로 LangChain 표준 도구 프로토콜 준수
- 모든 public 함수에 docstring 및 Args 섹션 존재, 가독성 양호
- `settings`의 `pydantic_settings` 기반 환경변수 관리, `env_nested_delimiter` 활용한 중첩 설정(`OPIK__URL_OVERRIDE`) 깔끔
- 도구 함수들의 `try/except` + 사용자 친화적 오류 문자열 반환 패턴 일관성 있음

### 개선 필요 사항

**[심각] JSON 인젝션 버그 — `agent_service.py:170`**

```python
# agent_service.py:170
yield f'{{"step": "tools", "name": {json.dumps(message.name)}, "content": {message.content}}}'
```

`message.content`를 `json.dumps` 없이 f-string에 직접 삽입하고 있다. 툴 결과에 `"` 또는 `}` 문자가 포함되면 JSON이 깨진다. 비교:

```python
# 166번 줄은 올바르게 처리:
"content": {json.dumps(args.get("content"), ensure_ascii=False)}
# 170번 줄은 누락:
"content": {message.content}  # ← 버그
```

**[심각] `progress_queue`가 인스턴스 공유 — 동시 요청 시 이벤트 혼선**

```python
# agent_service.py:56
self.progress_queue: asyncio.Queue = asyncio.Queue()
```

`AgentService`가 싱글톤으로 사용되는 구조(`process_query`가 async generator)에서, 요청 A와 요청 B가 동시에 실행되면 두 요청의 progress 이벤트가 같은 큐에 섞인다. 요청별 큐가 필요하다.

**[보통] 타입 힌트 혼용**

- `config.py:19`: `List[str]` (레거시 `typing.List`)
- `agent_service.py:5`: `Optional` (레거시 `typing.Optional`)
- `_rag_common.py:34`: `dict` (현대식)

Python 3.9+ 프로젝트라면 `list[str]`, `str | None` 등 현대식으로 통일 권장.

**[보통] `_create_agent` 동시 호출 경쟁 조건**

```python
# agent_service.py:62-63
if self.agent is not None:
    return
```

asyncio 환경에서 `process_query`가 두 요청에 의해 동시에 처음 호출되면 두 호출 모두 `self.agent is None`을 보고 중복 초기화할 수 있다. `asyncio.Lock` 사용 권장.

**[경미] `config.py:19` `List` 임포트만 `typing`에서, 나머지는 `|` 문법**

```python
from typing import List  # List만 사용
...
CORS_ORIGINS: List[str] = ["*"]
```

`list[str]`으로 교체하면 `from typing import List` 임포트도 제거 가능.

---

## 우선순위 개선 목록

| 순위 | 중요도 | 항목 | 파일:라인 |
|------|--------|------|-----------|
| 1 | 🔴 버그 | `message.content` JSON 인젝션 — `json.dumps` 누락 | `agent_service.py:170` |
| 2 | 🔴 설계 | `progress_queue` 인스턴스 공유로 동시 요청 시 이벤트 혼선 | `agent_service.py:56` |
| 3 | 🟠 오해 유발 | `create_agent`/`ToolStrategy`가 DeepAgents 라이브러리임을 명시 | `stock_agent.py:6-7` |
| 4 | 🟠 디버깅 | `rerank_hits` 예외 로깅 없이 `return None` | `_rag_common.py:96` |
| 5 | 🟠 안전성 | `embed_query` 실패 시 그래프 전체 중단 — `vector_search` 노드 try/except 추가 | `sec_search_agent.py:75-95` |
| 6 | 🟡 문서 불일치 | docstring "기본값: MemorySaver" vs 실제 `None` | `stock_agent.py:52` |
| 7 | 🟡 주석 오류 | `TYPE_CHECKING` 블록 주석이 역할을 잘못 설명 | `_rag_common.py:11-16` |
| 8 | 🟡 관례 위반 | 임포트 위치가 파일 중간 (PEP 8) | `sec_search_agent.py:142-144` |
| 9 | 🟡 잠재적 경쟁 | `_create_agent` 비동기 중복 초기화 | `agent_service.py:62-63` |
| 10 | 🟢 정리 | 미사용 `MemorySaver` 임포트 제거 | `stock_agent.py:10` |
| 11 | 🟢 일관성 | 타입 힌트 `List`/`Optional` → 현대식 통일 | `config.py:19`, `agent_service.py:5` |
