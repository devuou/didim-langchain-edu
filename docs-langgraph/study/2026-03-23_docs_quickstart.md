# LangGraph 빠른 시작 (Quickstart)

> **출처:** https://docs.langchain.com/oss/python/langgraph/quickstart
> [← 메인 목차로](2026-03-23_docs.md)

---

## 개요

이 퀵스타트는 **계산기 에이전트**를 예제로 LangGraph의 두 가지 API를 비교합니다.

- **Graph API**: 그래프(노드 + 엣지)로 에이전트를 정의
- **Functional API**: 단일 함수로 에이전트를 정의

사전 준비: Anthropic API 키 필요

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

---

## Graph API로 에이전트 만들기

### 1단계: 도구(Tools)와 모델 정의

LLM이 사용할 수 있는 함수들을 `@tool` 데코레이터로 정의합니다.

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-sonnet-4-6", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

# LLM에 도구 목록 바인딩
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

### 2단계: 상태(State) 정의

그래프의 상태는 에이전트 실행 전체에 걸쳐 유지되는 **공유 메모리**입니다.

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
```

> **핵심 포인트**: `Annotated[list[AnyMessage], operator.add]`의 의미
> - `operator.add` = 리듀서(reducer) 함수
> - 새 메시지가 기존 목록에 **추가(append)** 됨 (덮어쓰지 않음)

### 3단계: 모델 노드 정의

LLM을 호출하여 도구를 쓸지 말지 결정하는 노드입니다.

```python
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM이 도구 호출 여부를 결정"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }
```

### 4단계: 도구 노드 정의

LLM이 요청한 도구를 실제로 실행하는 노드입니다.

```python
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """도구 호출 실행"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

### 5단계: 라우팅 함수 정의

LLM이 도구를 호출했는지 여부에 따라 다음 노드를 결정합니다.

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """도구 호출 여부에 따라 루프 계속 or 종료"""
    messages = state["messages"]
    last_message = messages[-1]
    # LLM이 도구 호출을 했으면 tool_node로
    if last_message.tool_calls:
        return "tool_node"
    # 아니라면 종료
    return END
```

### 6단계: 그래프 구성 및 컴파일

```python
# 그래프 빌더 생성
agent_builder = StateGraph(MessagesState)

# 노드 추가
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# 엣지 연결
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")  # 도구 실행 후 다시 LLM으로

# 컴파일 (반드시 필요!)
agent = agent_builder.compile()
```

**그래프 구조 시각화:**

```
START
  ↓
llm_call ──(도구 호출 없음)──→ END
  ↑              ↓ (도구 호출 있음)
  └────── tool_node
```

### 7단계: 실행

```python
from langchain.messages import HumanMessage

messages = [HumanMessage(content="Add 3 and 4.")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()
```

---

## Functional API로 에이전트 만들기

Functional API는 기존 Python 코드처럼 작성하면서 LangGraph의 기능을 사용하는 방법입니다.

### 1단계: 도구와 모델 정의

Graph API와 동일하게 도구를 정의합니다. (코드 생략, 위와 동일)

### 2단계: `@task`로 모델 노드 정의

```python
from langgraph.func import entrypoint, task
from langchain_core.messages import BaseMessage

@task
def call_llm(messages: list[BaseMessage]):
    """LLM이 도구 호출 여부를 결정"""
    return model_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
        + messages
    )
```

> **`@task` 데코레이터**: 비동기적으로 실행될 수 있는 독립적인 작업 단위를 표시합니다.

### 3단계: `@task`로 도구 노드 정의

```python
from langchain.messages import ToolCall

@task
def call_tool(tool_call: ToolCall):
    """도구 호출 실행"""
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)
```

### 4단계: `@entrypoint`로 에이전트 정의

Graph API의 노드/엣지 대신, 일반 Python 루프와 조건문으로 에이전트 로직을 작성합니다.

```python
from langgraph.graph import add_messages

@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break  # 도구 호출 없으면 종료

        # 도구 병렬 실행
        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages

# 실행
messages = [HumanMessage(content="Add 3 and 4.")]
for chunk in agent.stream(messages, stream_mode="updates"):
    print(chunk)
    print()
```

---

## Graph API vs Functional API 비교 요약

| 구분 | Graph API | Functional API |
|---|---|---|
| **구조** | 노드 + 엣지로 명시적 선언 | 일반 Python 함수 + `@entrypoint` |
| **상태 관리** | TypedDict 명시적 선언 + 리듀서 | 함수 내부에서 변수로 관리 |
| **흐름 제어** | `add_conditional_edges` 등 | `if/else`, `while` 루프 |
| **시각화** | 그래프 시각화 지원 | 미지원 |
| **적합한 경우** | 복잡한 분기, 팀 협업, 디버깅 | 빠른 프로토타이핑, 기존 코드 통합 |

---

## 다음 단계

- [핵심 개념 →](2026-03-23_docs_concepts.md) - State, Node, Edge, Command 상세 이해
- [워크플로우 패턴 →](2026-03-23_docs_patterns.md) - 실전 패턴들
