# LangGraph 핵심 개념 (Concepts)

> **출처:** https://docs.langchain.com/oss/python/langgraph/graph-api
> [← 메인 목차로](2026-03-23_docs.md)

---

## Graph API의 세 가지 핵심 구성요소

LangGraph에서 에이전트는 세 가지 요소로 구성됩니다:

```
┌────────────────────────────────────────────────────────────┐
│                         Graph                              │
│                                                            │
│  State (공유 데이터)                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  messages: [...]  │  user_info: {...}  │  ...        │  │
│  └──────────────────────────────────────────────────────┘  │
│           ↓ 읽기/쓰기              ↓ 읽기/쓰기               │
│  Nodes (작업 함수)      Edges (라우팅 규칙)                   │
│  ┌─────────────┐       ┌─────────────────────────────┐     │
│  │ node_a      │──────▶│ 조건: 다음 노드는 어디?          │     │
│  │ node_b      │       │ - 조건 엣지 (conditional)     │     │
│  │ node_c      │       │ - 일반 엣지 (fixed)           │     │
│  └─────────────┘       └─────────────────────────────┘     │
└────────────────────────────────────────────────────────────┘
```

> **핵심 원칙**: **노드는 작업을 하고, 엣지는 다음을 결정한다.**

---

## 실행 모델: 슈퍼스텝(Super-step)

LangGraph는 Google의 **Pregel** 시스템에서 영감을 받아 **메시지 패싱(message passing)** 방식으로 동작합니다.

- **슈퍼스텝**: 그래프 노드에 대한 한 번의 반복(iteration)
- 병렬로 실행되는 노드들은 같은 슈퍼스텝에 속함
- 순차 실행되는 노드들은 별도의 슈퍼스텝에 속함
- 모든 노드가 비활성 상태이고 전송 중인 메시지가 없으면 실행 종료

---

## 1. State (상태)

State는 그래프의 모든 노드가 읽고 쓸 수 있는 **공유 데이터 구조**입니다.

### 스키마 정의 방법

**TypedDict** (가장 일반적):
```python
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list
    user_name: str
    count: int
```

**dataclass** (기본값 설정 필요 시):
```python
from dataclasses import dataclass

@dataclass
class State:
    messages: list
    count: int = 0
```

**Pydantic BaseModel** (유효성 검증 필요 시):
```python
from pydantic import BaseModel

class State(BaseModel):
    messages: list
    count: int
```

---

### 리듀서(Reducer): 상태 업데이트 방식 제어

리듀서는 각 상태 키가 어떻게 업데이트될지 결정합니다.

#### 기본 동작 (덮어쓰기)

리듀서를 지정하지 않으면 새 값이 기존 값을 **덮어씁니다.**

```python
class State(TypedDict):
    foo: int
    bar: list[str]

# 초기 상태: {"foo": 1, "bar": ["hi"]}
# 노드가 {"bar": ["bye"]} 반환
# 결과: {"foo": 1, "bar": ["bye"]}  ← "hi"가 사라짐!
```

#### 커스텀 리듀서 (추가하기)

`Annotated`를 사용해 리듀서 함수를 지정합니다.

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

# 초기 상태: {"foo": 1, "bar": ["hi"]}
# 노드가 {"bar": ["bye"]} 반환
# 결과: {"foo": 1, "bar": ["hi", "bye"]}  ← "hi"가 유지됨!
```

---

### MessagesState: 메시지 관리 전용 상태

채팅 메시지 관리가 워낙 흔한 패턴이라 전용 상태 클래스가 있습니다.

```python
from langgraph.graph import MessagesState

# MessagesState를 그대로 사용
graph = StateGraph(MessagesState)

# 또는 상속해서 필드 추가
class MyState(MessagesState):
    documents: list[str]
    user_name: str
```

`MessagesState`의 `messages` 필드는 `add_messages` 리듀서를 사용합니다:
- 메시지 ID로 중복 제거
- `HumanMessage` 객체와 딕셔너리 형태 모두 지원

```python
# 둘 다 동작함
{"messages": [HumanMessage(content="안녕")]}
{"messages": [{"type": "human", "content": "안녕"}]}
```

---

### 다중 스키마

내부 노드 간 통신에만 쓰이는 private 상태를 별도로 정의할 수 있습니다.

```python
class InputState(TypedDict):    # 입력용
    user_input: str

class OutputState(TypedDict):   # 출력용
    graph_output: str

class OverallState(TypedDict):  # 내부 전체 상태
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):  # 특정 노드 간 private 통신용
    bar: str

def node_1(state: InputState) -> OverallState:
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(
    OverallState,
    input_schema=InputState,    # 입력 스키마 지정
    output_schema=OutputState   # 출력 스키마 지정
)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input": "My"})
# 출력: {'graph_output': 'My name is Lance'}
```

---

## 2. Nodes (노드)

노드는 상태를 받아 처리하고 업데이트된 상태를 반환하는 **Python 함수**입니다.

### 기본 노드 정의

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

class State(TypedDict):
    input: str
    results: str

# 기본 노드 (state만 받기)
def plain_node(state: State):
    return {"results": f"Hello, {state['input']}!"}

# config 접근이 필요한 노드
def node_with_config(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    return {"results": f"Thread {thread_id}: Hello, {state['input']}!"}

# Runtime 접근이 필요한 노드
def node_with_runtime(state: State, runtime: Runtime):
    return {"results": f"Hello, {state['input']}!"}
```

### 노드 추가 방법

```python
builder = StateGraph(State)

# 이름 명시
builder.add_node("my_node", plain_node)

# 이름 생략 (함수 이름 자동 사용)
builder.add_node(plain_node)  # "plain_node"로 참조됨
```

### START / END 노드

```python
from langgraph.graph import START, END

# START: 사용자 입력이 들어오는 시작점
graph.add_edge(START, "first_node")

# END: 그래프 종료 지점
graph.add_edge("last_node", END)
```

### 노드 캐싱

비용이 큰 노드의 결과를 캐시할 수 있습니다:

```python
import time
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

def expensive_node(state: State) -> dict[str, int]:
    time.sleep(2)  # 오래 걸리는 작업 시뮬레이션
    return {"result": state["x"] * 2}

builder.add_node(
    "expensive_node",
    expensive_node,
    cache_policy=CachePolicy(ttl=3)  # 3초 캐시
)

graph = builder.compile(cache=InMemoryCache())

# 첫 번째 호출: 2초 소요
graph.invoke({"x": 5})
# 두 번째 호출: 캐시에서 즉시 반환
# 결과에 '__metadata__': {'cached': True} 포함
graph.invoke({"x": 5})
```

---

## 3. Edges (엣지)

엣지는 한 노드에서 다음 노드로의 **흐름**을 정의합니다.

### 일반 엣지 (Normal Edge)

항상 같은 노드로 이동합니다.

```python
graph.add_edge("node_a", "node_b")  # node_a → node_b (항상)
```

### 조건부 엣지 (Conditional Edge)

함수의 반환값에 따라 다음 노드가 결정됩니다.

```python
def routing_function(state: State) -> str:
    if state["score"] > 80:
        return "high_score_node"
    else:
        return "low_score_node"

graph.add_conditional_edges("judge_node", routing_function)

# 매핑 딕셔너리 사용 (함수 반환값 → 노드 이름 매핑)
graph.add_conditional_edges(
    "judge_node",
    routing_function,
    {True: "node_b", False: "node_c"}
)
```

### 진입점 (Entry Point)

```python
# 시작 노드 지정
graph.add_edge(START, "first_node")

# 조건부 시작 노드
graph.add_conditional_edges(START, routing_function)
```

### 병렬 실행

한 노드에서 여러 노드로 엣지를 연결하면 **병렬 실행**됩니다:

```python
# 세 노드가 동시에 실행됨
graph.add_edge(START, "node_a")
graph.add_edge(START, "node_b")
graph.add_edge(START, "node_c")
```

---

## 4. Send API (동적 병렬 실행)

병렬로 실행할 작업의 수를 미리 알 수 없을 때 사용합니다. **Map-Reduce 패턴**에 적합합니다.

```python
from langgraph.types import Send

def continue_to_jokes(state: OverallState):
    # state['subjects']의 각 항목에 대해 별도 노드 실행
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("planning_node", continue_to_jokes)
```

`Send(노드이름, 해당_노드에_전달할_상태)`

---

## 5. Command (명령)

`Command`는 **상태 업데이트**와 **흐름 제어**를 동시에 할 수 있는 강력한 도구입니다.

### 기본 사용법

```python
from langgraph.types import Command
from typing import Literal

def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},      # 상태 업데이트
        goto="my_other_node"        # 다음 노드 지정
    )
```

> **중요**: `Command`를 반환할 때는 반환 타입에 `Command[Literal["노드이름"]]`으로 라우팅 가능한 노드를 명시해야 합니다 (그래프 렌더링에 필요).

### 조건에 따른 동적 라우팅

```python
def my_node(state: State) -> Command[Literal["node_a", "node_b"]]:
    if state["foo"] == "bar":
        return Command(update={"result": "yes"}, goto="node_a")
    else:
        return Command(update={"result": "no"}, goto="node_b")
```

### 서브그래프에서 부모 그래프로 이동

```python
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",   # 부모 그래프의 노드
        graph=Command.PARENT     # 부모 그래프를 대상으로 지정
    )
```

### interrupt 후 재개

```python
from langgraph.types import Command, interrupt

def human_review(state: State):
    answer = interrupt("승인하시겠습니까?")  # 여기서 일시 중단
    return {"messages": [{"role": "user", "content": answer}]}

# 첫 번째 실행 - interrupt에서 중단
result = graph.invoke({"messages": [...]}, config)

# 사람이 검토 후 재개
result = graph.invoke(Command(resume="승인"), config)
```

> **주의**: `Command(resume=...)`은 `invoke()`/`stream()`의 입력으로만 사용하세요. 기존 스레드에서 대화를 계속하려면 일반 딕셔너리를 입력으로 사용하세요.

---

## 6. 그래프 컴파일

그래프를 사용하기 전에 반드시 컴파일해야 합니다.

```python
# 기본 컴파일
graph = graph_builder.compile()

# 체크포인터와 함께 컴파일 (상태 영속성)
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

컴파일 시 수행되는 작업:
- 그래프 구조 기본 검증
- 런타임 인자(체크포인터, 브레이크포인트 등) 설정

---

## 7. Recursion Limit (재귀 제한)

무한 루프 방지를 위한 슈퍼스텝 최대 횟수 설정입니다.

- 기본값: **1000** (v1.0.6 이후)
- 초과 시 `GraphRecursionError` 발생

```python
graph.invoke(inputs, config={"recursion_limit": 50})
```

### RemainingSteps로 사전 감지

```python
from langgraph.managed import RemainingSteps

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    remaining_steps: RemainingSteps  # LangGraph가 자동으로 값 채움

def reasoning_node(state: State) -> dict:
    remaining = state["remaining_steps"]
    if remaining <= 2:
        return {"messages": ["제한에 가까워져 마무리합니다..."]}
    return {"messages": ["계속 생각 중..."]}

def route_decision(state: State) -> Literal["reasoning_node", "fallback_node"]:
    if state["remaining_steps"] <= 2:
        return "fallback_node"  # 남은 스텝이 적으면 폴백으로
    return "reasoning_node"
```

---

## 8. Runtime Context

런타임 시 그래프 전체에 공유되는 설정 값을 주입할 수 있습니다.

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class ContextSchema:
    llm_provider: str = "openai"

graph = StateGraph(State, context_schema=ContextSchema)

# 실행 시 context 주입
graph.invoke(inputs, context={"llm_provider": "anthropic"})
```

노드에서 context 접근:

```python
def node_a(state: State, runtime: Runtime[ContextSchema]):
    llm = get_llm(runtime.context.llm_provider)
    # ...
```

---

## 9. Functional API 핵심 개념

### @entrypoint

워크플로우의 시작점을 표시합니다.

```python
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def my_workflow(some_input: dict) -> int:
    # 장기 실행 작업과 interrupt를 포함할 수 있는 로직
    ...
    return result
```

**인젝션 가능한 파라미터:**

```python
@entrypoint(checkpointer=checkpointer, store=store)
def my_workflow(
    some_input: dict,
    *,
    previous: Any = None,       # 이전 체크포인트의 상태 (단기 메모리)
    store: BaseStore,            # 장기 메모리 스토어
    writer: StreamWriter,        # 스트리밍 (Python 3.11 미만 비동기)
    config: RunnableConfig       # 런타임 설정
) -> ...:
    ...
```

**단기 메모리 활용:**

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> int:
    previous = previous or 0
    return number + previous

my_workflow.invoke(1, config)  # 1  (이전값: None)
my_workflow.invoke(2, config)  # 3  (이전값: 1)
my_workflow.invoke(5, config)  # 8  (이전값: 3)
```

**`entrypoint.final` - 반환값과 저장값 분리:**

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(number: int, *, previous: Any = None) -> entrypoint.final[int, int]:
    previous = previous or 0
    # 호출자에게는 previous를 반환하지만, 체크포인트에는 2 * number를 저장
    return entrypoint.final(value=previous, save=2 * number)

my_workflow.invoke(3, config)  # 반환: 0  (이전값 None), 저장: 6
my_workflow.invoke(1, config)  # 반환: 6  (이전값 6), 저장: 2
```

---

### @task

독립적으로 실행되는 작업 단위입니다.

```python
from langgraph.func import task

@task
def slow_computation(input_value):
    # 오래 걸리는 작업
    return result
```

**특징:**
- 호출 즉시 **future 객체** 반환 (비동기적)
- entrypoint, 다른 task, 또는 StateGraph 노드 내에서만 호출 가능
- 직접 애플리케이션 코드에서는 호출 불가

```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(some_input: int) -> int:
    future = slow_computation(some_input)  # 즉시 반환
    return future.result()  # 여기서 결과를 기다림
```

**언제 task를 사용해야 하나?**

| 상황 | 이유 |
|---|---|
| **체크포인팅** | 오래 걸리는 작업 결과를 저장해 재시작 시 재계산 방지 |
| **Human-in-the-loop** | 무작위성을 캡슐화해 올바른 재개 보장 |
| **병렬 실행** | I/O 바운드 작업의 동시 실행 |
| **재시도 가능한 작업** | 재시도 로직 캡슐화 |

---

### 결정론성 (Determinism) — 중요!

`@entrypoint` 함수 내의 모든 **부수 효과(side effect)** 와 **비결정적 코드**는 반드시 `@task` 안에 넣어야 합니다.

실행이 중단되었다가 재개될 때 동일한 순서로 실행되어야 하기 때문입니다.

**잘못된 예 (재개 시 재실행됨):**
```python
@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    # ❌ interrupt 전 부수 효과 - 재개 시 다시 실행됨!
    with open("output.txt", "w") as f:
        f.write("Side effect executed")
    value = interrupt("question")
    return value
```

**올바른 예 (task 안에 캡슐화):**
```python
@task
def write_to_file():
    with open("output.txt", "w") as f:
        f.write("Side effect executed")

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    write_to_file().result()  # ✅ task 안에서 실행
    value = interrupt("question")
    return value
```

**비결정적 제어 흐름도 task로 감싸야 함:**
```python
@task
def get_time() -> float:
    return time.time()  # ✅ 시간 조회도 task로

@entrypoint(checkpointer=checkpointer)
def my_workflow(inputs: dict) -> int:
    t1 = get_time().result()  # 재개 시 체크포인트 값 재사용
    if t1 > inputs["threshold"]:
        result = fast_task(1).result()
    else:
        result = slow_task(2).result()
    value = interrupt("question")
    return {"result": result, "value": value}
```

---

## Graph API vs Functional API 심층 비교

| 기능 | Graph API | Functional API |
|---|---|---|
| **제어 흐름** | 선언적 노드 + 엣지 | 표준 Python (if/else, 루프, 함수 호출) |
| **단기 메모리** | State와 리듀서로 명시적 관리 | 함수 스코프, 명시적 상태 관리 불필요 |
| **체크포인팅** | 모든 슈퍼스텝 후 새 체크포인트 | entrypoint당 기존 체크포인트에 작업 결과 저장 |
| **시각화** | 지원 (그래프 그리기 가능) | 미지원 (런타임에 동적 결정) |

> **두 API는 같은 런타임을 공유**하므로 함께 사용할 수 있습니다.

---

## 다음 단계

- [워크플로우 패턴 →](2026-03-23_docs_patterns.md) - 실전 패턴들 (프롬프트 체이닝, 병렬화, 에이전트 등)
