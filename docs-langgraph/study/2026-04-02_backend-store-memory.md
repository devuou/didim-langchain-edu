# Backend / Store / Memory 개념 정리

> 작성일: 2026-04-02  
> 출처: LangChain 공식 문서 (MCP 실시간 참조)

---

## 한눈에 보기

| 개념 | 프레임워크 | 범위 | 역할 |
|------|-----------|------|------|
| **Checkpointer** (Persistence) | LangGraph | Thread 내 | 그래프 State 스냅샷 저장 → 단기 메모리 |
| **Store** | LangGraph | Thread 간 | 네임스페이스 기반 장기 데이터 저장 → 장기 메모리 |
| **Memory** | LangGraph/LangChain | 개념 | 단기(checkpointer) + 장기(store)를 아우르는 상위 개념 |
| **Backend** | DeepAgent | — | 에이전트에게 파일시스템 인터페이스를 제공하는 플러그형 레이어 |

---

## 1. Memory (메모리) — LangGraph / LangChain 공통 개념

메모리는 **이전 상호작용을 기억하는 시스템**이다. 범위에 따라 두 가지로 나뉜다.

### 1-1. 단기 메모리 (Short-term Memory)

- **범위**: Thread 내 (대화 세션 내)
- **구현**: LangGraph의 **Checkpointer**
- 그래프 State의 일부로 관리되며, 각 스텝마다 DB에 저장
- Thread가 재개될 때 이전 대화 이력 그대로 복원

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# thread_id로 대화 이력 연결
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": [...]}, config)
```

### 1-2. 장기 메모리 (Long-term Memory)

- **범위**: Thread 간 (여러 대화 세션에 걸쳐)
- **구현**: LangGraph의 **Store**
- Namespace로 구조화된 JSON 문서 저장
- 어느 Thread에서든 읽고 쓸 수 있음

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = (user_id, "memories")
store.put(namespace, memory_id, {"food_preference": "I like pizza"})
memories = store.search(namespace)
```

### 1-3. 메모리 유형 (심리학적 분류)

공식 문서는 인간의 기억 유형을 AI 에이전트에 매핑한다.

| 유형 | 저장 내용 | 인간 예시 | 에이전트 예시 |
|------|----------|----------|-------------|
| **Semantic** (의미적) | 사실/개념 | 학교에서 배운 것 | 사용자 선호도, 프로필 |
| **Episodic** (에피소드) | 경험/행동 | 과거에 한 일 | 과거 에이전트 행동, few-shot 예시 |
| **Procedural** (절차적) | 수행 방법/규칙 | 자전거 타는 법 | 시스템 프롬프트 자체 수정 |

**Semantic Memory 구현 방식:**
- **Profile**: 사용자에 대한 단일 JSON 문서를 지속적으로 업데이트
- **Collection**: 개별 메모리 문서 컬렉션을 계속 추가/수정 (더 높은 recall)

**Episodic Memory 구현:** Few-shot 예시로 프롬프트에 주입

**Procedural Memory 구현:** Reflection/메타 프롬프팅으로 에이전트가 자신의 시스템 프롬프트를 직접 수정

### 1-4. 메모리 업데이트 방식

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **Hot path** | 실행 중 실시간 업데이트 | 즉각 반영, 사용자 알림 가능 | 레이턴시 증가 |
| **Background** | 비동기 백그라운드 태스크 | 레이턴시 없음 | 타이밍 관리 필요 |

---

## 2. Store (스토어) — LangGraph

**Thread 간 데이터를 공유하기 위한 인터페이스** (`BaseStore`).

Checkpointer가 "이 대화 내에서" State를 저장한다면,  
Store는 "여러 대화에 걸쳐" 정보를 유지한다.

### 핵심 구조

```
Store
├── namespace: ("user_id", "memories")   # 폴더처럼 데이터 구분
│   ├── key: "memory-uuid-1"             # 파일명처럼 개별 항목 식별
│   │   └── value: {"food": "pizza"}    # 실제 저장 데이터 (JSON)
│   └── key: "memory-uuid-2"
│       └── value: {"hobby": "coding"}
```

### 주요 메서드

```python
# 저장
store.put(namespace, key, value)

# 단건 조회
item = store.get(namespace, key)

# 검색 (필터 + 시맨틱 서치)
items = store.search(namespace, query="food preferences", limit=3)
items = store.search(namespace, filter={"my-key": "my-value"})
```

### 시맨틱 서치 설정

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["$"]   # 임베딩할 필드 지정
    }
)
```

### 구현체

| 구현체 | 용도 |
|--------|------|
| `InMemoryStore` | 개발/테스트 |
| `PostgresStore` | 프로덕션 |
| `RedisStore` | 프로덕션 |

### 그래프에서 사용

```python
# Store는 Runtime 객체를 통해 노드 내에서 접근
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    namespace = (runtime.context.user_id, "memories")
    await runtime.store.aput(namespace, str(uuid.uuid4()), {"memory": "..."})

# 컴파일 시 checkpointer와 store를 함께 전달
graph = builder.compile(checkpointer=checkpointer, store=store)
```

---

## 3. Checkpointer (Persistence) — LangGraph

**그래프 State를 각 슈퍼스텝마다 스냅샷으로 저장**하는 레이어.

### 핵심 개념

- **Thread**: `thread_id`로 식별되는 대화 단위. 누적된 실행 State를 담음
- **Checkpoint**: 특정 시점의 그래프 State 스냅샷 (`StateSnapshot`)
- **Super-step**: 그래프의 "한 틱". 모든 병렬 노드가 실행되는 단위

```
START → node_a → node_b → END
  ↓        ↓        ↓
체크포인트  체크포인트  체크포인트  (각 슈퍼스텝마다 저장)
```

### 용도

| 용도 | 설명 |
|------|------|
| **대화 메모리** | 이전 메시지 이력 유지 |
| **Human-in-the-loop** | 실행 중단 후 사람의 승인/수정 후 재개 |
| **Time travel** | 과거 체크포인트로 돌아가거나 분기 |
| **장애 복구** | 실패 시 마지막 성공 스텝부터 재시작 |

### 기본 사용

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar": []}, config)

# 현재 State 조회
state = graph.get_state(config)

# State 히스토리 조회 (최신순)
history = list(graph.get_state_history(config))
```

### 구현체

| 구현체 | 특징 | 설치 |
|--------|------|------|
| `InMemorySaver` | 메모리 내 저장, 재시작 시 소멸 | 기본 포함 |
| `SqliteSaver` | SQLite, 로컬 개발 | `langgraph-checkpoint-sqlite` |
| `PostgresSaver` | PostgreSQL, 프로덕션 | `langgraph-checkpoint-postgres` |
| `CosmosDBSaver` | Azure Cosmos DB | `langgraph-checkpoint-cosmosdb` |

### StateSnapshot 구조

```python
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},   # 채널 값
    next=(),                                    # 다음에 실행될 노드 (비어있으면 완료)
    config={'configurable': {'thread_id': '1', 'checkpoint_id': '...'}},
    metadata={'source': 'loop', 'writes': {...}, 'step': 2},
    created_at='2024-08-29T19:19:38...',
    parent_config={...},
    tasks=()
)
```

---

## 4. Backend (백엔드) — DeepAgent 전용 개념

DeepAgent에서만 등장하는 고유 개념.  
**에이전트에게 파일시스템 인터페이스를 제공하는 플러그형 레이어**다.

에이전트는 `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` 등의 도구로 "파일"을 다루는데, 이 도구들이 실제로 어디에 데이터를 저장할지를 Backend가 결정한다.

```
에이전트 도구 (ls, read_file, write_file ...)
         ↓
      Backend (플러그형)
    ↙  ↓  ↓  ↓  ↘
State Disk Store Sandbox LocalShell
```

### Backend 종류

#### StateBackend (기본값) — Ephemeral
```python
agent = create_deep_agent()  # 기본값이 StateBackend
```
- LangGraph State 안에 파일 저장
- 같은 Thread 내에서만 유지 (체크포인트로 인해 동일 thread 내 다음 턴에도 접근 가능)
- Thread가 끝나면 소멸
- **용도**: 에이전트의 임시 작업 공간, 중간 결과물 저장

#### FilesystemBackend — 로컬 디스크
```python
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/my/project", virtual_mode=True)
)
```
- 실제 로컬 파일시스템 읽기/쓰기
- `virtual_mode=True` 권장 (경로 탈출 방지)
- **주의**: 프로덕션 웹서버에는 부적합 (시크릿 노출 위험)
- **용도**: 로컬 개발 CLI, CI/CD 파이프라인

#### LocalShellBackend — 로컬 셸 (매우 주의)
```python
from deepagents.backends import LocalShellBackend

agent = create_deep_agent(
    backend=LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"})
)
```
- FilesystemBackend + `execute` 도구 (임의 셸 명령 실행)
- **극도로 주의**: 프로덕션 환경에서 절대 사용 금지
- **용도**: 개인 개발 환경, 로컬 코딩 어시스턴트

#### StoreBackend — LangGraph Store (영속적, Thread 간)
```python
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=lambda rt: StoreBackend(
        rt,
        namespace=lambda ctx: (ctx.runtime.context.user_id,),
    ),
    store=InMemoryStore()   # LangSmith 배포 시에는 생략 (자동 제공)
)
```
- LangGraph `BaseStore`에 파일 저장
- Thread가 바뀌어도 데이터 유지 (장기 기억용)
- **namespace factory**로 사용자/테넌트별 데이터 격리
- **용도**: 에이전트 장기 메모리, 사용자별 설정 저장

#### CompositeBackend — 경로별 라우팅
```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

composite_backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),       # 기본: 임시 저장
    routes={
        "/memories/": StoreBackend(rt),   # /memories/ 경로만 영속 저장
    }
)
```
- 경로 prefix에 따라 다른 Backend로 라우팅
- **용도**: 임시 작업공간은 StateBackend, 장기 메모리는 StoreBackend로 분리

#### Custom Backend — 가상 파일시스템
- `BackendProtocol`을 구현하면 S3, Postgres 등 어떤 저장소도 파일시스템처럼 노출 가능
- 필수 구현 메서드: `ls_info`, `read`, `grep_raw`, `glob_info`, `write`, `edit`

---

## 5. 프레임워크별 개념 비교 요약

```
LangChain
├── Memory (개념)
│   ├── Short-term: 대화 히스토리 관리 (메시지 트리밍 등)
│   └── Long-term: LangGraph Store 기반

LangGraph
├── Checkpointer (= Persistence)
│   └── Thread 내 State 스냅샷 → 단기 메모리, Human-in-the-loop, Time travel
├── Store
│   └── Thread 간 데이터 공유 → 장기 메모리 (semantic/episodic/procedural)
└── Memory (개념)
    ├── Short-term = Checkpointer 활용
    └── Long-term = Store 활용

DeepAgent
└── Backend (고유 개념)
    ├── StateBackend → LangGraph State 활용 (단기, Ephemeral)
    ├── FilesystemBackend → 로컬 디스크
    ├── LocalShellBackend → 로컬 디스크 + 셸 실행
    ├── StoreBackend → LangGraph Store 활용 (장기, Cross-thread)
    ├── CompositeBackend → 경로 기반 라우팅
    └── Custom Backend → 가상 파일시스템 (S3, DB 등)
```

### Checkpointer vs Store 핵심 차이

| | Checkpointer | Store |
|--|-------------|-------|
| **저장 단위** | 그래프 State 전체 스냅샷 | 개별 JSON 문서 |
| **범위** | Thread 내 (단기) | Thread 간 (장기) |
| **접근 방식** | `thread_id` + `checkpoint_id` | `namespace` + `key` |
| **검색** | 히스토리 순회 | 필터 + 시맨틱 서치 |
| **주요 용도** | 대화 이력, 실행 재개 | 사용자 프로필, 장기 기억 |

---

## 참고 링크

- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Memory 개념 개요](https://docs.langchain.com/oss/python/concepts/memory)
- [DeepAgent Backends](https://docs.langchain.com/oss/python/deepagents/backends)

---

## 6. 현재 프로젝트의 기억장치 분석

> 분석 대상: `app/services/agent_service.py`, `app/services/conversation_service.py`, `app/services/threads_service.py`, `app/agents/stock_agent.py`, `app/agents/sec_search_agent.py`

### 전체 구조 요약

```
현재 프로젝트의 기억장치
├── [단기] LangGraph MemorySaver (Checkpointer)   ← LLM이 실제 참조하는 대화 이력
├── [단기] ConversationService (Python dict)       ← UI/API 응답용 대화 목록
├── [영속] ThreadsService (JSON 파일)              ← 읽기 전용 정적 데이터
└── [영속] Elasticsearch                           ← 외부 지식베이스 (SEC 10-K, 주가 히스토리)

※ LangGraph Store (장기 메모리) → 미사용
```

---

### 1) LangGraph `MemorySaver` — 단기 대화 메모리

**파일**: `app/services/agent_service.py`, `app/agents/stock_agent.py`

```python
# agent_service.py - AgentService._create_agent()
from langgraph.checkpoint.memory import MemorySaver
self.checkpointer = MemorySaver()

# process_query() - thread_id로 대화 문맥 연결
agent_stream = self.agent.astream(
    {"messages": [HumanMessage(content=user_messages)]},
    config={"configurable": {"thread_id": str(thread_id)}},
)
```

```python
# stock_agent.py - create_stock_agent()
agent = create_agent(
    model=model,
    tools=[...],
    checkpointer=checkpointer,  # ← AgentService가 주입
)
```

| 항목 | 내용 |
|------|------|
| **구현체** | `MemorySaver` (In-memory) |
| **범위** | 같은 `thread_id` 내 대화 이력 |
| **생존 기간** | 서버 프로세스 실행 중 |
| **서버 재시작 시** | **소멸** (영속화 없음) |
| **역할** | LLM이 이전 대화를 참조하기 위한 단기 메모리 |

**한계**: 프로덕션에서 서버가 재시작되면 모든 대화 이력이 사라진다. `PostgresSaver`나 `SqliteSaver`로 교체하면 영속화 가능.

---

### 2) `ConversationService` — UI용 대화 목록 저장소

**파일**: `app/services/conversation_service.py`

```python
class ConversationService:
    def __init__(self):
        self._conversations: Dict[str, Dict[str, Any]] = {}  # 대화 메타데이터
        self._messages: Dict[str, List[LangChainMessage]] = {}  # 메시지 이력
```

| 항목 | 내용 |
|------|------|
| **구현체** | Python `dict` (순수 인메모리) |
| **범위** | `conversation_id` (= `thread_id`) 단위 |
| **생존 기간** | 서버 프로세스 실행 중 |
| **서버 재시작 시** | **소멸** |
| **역할** | API 응답용 대화 목록/이력 제공 (LLM이 직접 참조하지 않음) |

**주의**: LangGraph `MemorySaver`와 **별개로 존재**한다. 같은 `thread_id`를 공유하지만 서로 독립적으로 관리된다.
- `MemorySaver` → LLM이 다음 턴에 이전 메시지를 기억하기 위한 저장소
- `ConversationService` → 프론트엔드에 대화 이력을 보여주기 위한 저장소

---

### 3) `ThreadsService` — JSON 파일 기반 읽기 전용 데이터

**파일**: `app/services/threads_service.py`, `app/data/threads.json`, `app/data/threads/{thread_id}.json`

```python
async def get_thread_by_id_json(thread_id: uuid.UUID):
    json_data = read_json(f"threads/{str(thread_id)}.json")
    return RootBaseModel[ThreadDataResponse](response=ThreadDataResponse(**json_data))
```

| 항목 | 내용 |
|------|------|
| **구현체** | JSON 파일 (`app/data/`) |
| **범위** | `thread_id` 단위 |
| **생존 기간** | 파일이 존재하는 한 영속 |
| **역할** | **읽기 전용** 정적 데이터 (즐겨찾기 질문 등) |

---

### 4) Elasticsearch — 외부 지식베이스 (영속)

**파일**: `app/agents/sec_search_agent.py`, `app/agents/es_tools.py`

에이전트가 호출하는 도구 2종이 ES를 영속 저장소로 사용한다.

| 도구 | 인덱스 | 데이터 | 적재 방식 |
|------|--------|--------|----------|
| `search_sec_filing` | `{prefix}-10k-docs` | SEC 10-K 공시 문서 청크 | 오프라인 (`scripts/ingest_10k.py`) |
| `get_stock_history` | ES 주가 히스토리 인덱스 | OHLCV 과거 주가 데이터 | 서버 시작 시 (`elasticsearch/ingester.py`) |

`sec_search_agent`는 LangGraph `StateGraph`로 구성된 서브에이전트로, **BM25 + kNN 병렬 fan-out → merge → rerank** 파이프라인을 실행한다. 이 서브에이전트는 자체 `SecSearchState`를 가지지만 체크포인터는 없다(단일 실행 후 결과 반환).

---

### 현황 요약 및 개선 포인트

```
현재 상태
├── 단기 메모리: MemorySaver (in-memory) → 서버 재시작 시 소멸
├── 장기 메모리: 없음 (Store 미사용)
├── UI용 이력: ConversationService (in-memory) → 서버 재시작 시 소멸
└── 지식베이스: Elasticsearch (영속, 읽기 전용)

개선 방향
├── MemorySaver → PostgresSaver/SqliteSaver 교체 시 대화 이력 영속화
├── Store 도입 시 사용자별 장기 기억 (선호 종목, 투자 성향 등) 가능
└── ConversationService → DB 백엔드 교체 시 서버 재시작 후에도 이력 유지
    (conversation_service.py 클래스 주석에도 "향후 DB로 확장 가능" 명시됨)
```
