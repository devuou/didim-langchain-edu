# LangGraph 워크플로우 패턴 (Workflows & Agents)

> **출처:** https://docs.langchain.com/oss/python/langgraph/workflows-agents
> [← 메인 목차로](2026-03-23_docs.md)

---

## 워크플로우 vs 에이전트

| 구분 | 워크플로우 (Workflow) | 에이전트 (Agent) |
|---|---|---|
| **실행 경로** | 사전에 정해진 코드 경로 | 동적으로 자신의 프로세스를 결정 |
| **특징** | 순서대로 작동하도록 설계 | 도구 사용과 프로세스를 스스로 정의 |
| **적합한 경우** | 예측 가능한 작업, 안정성 중요 | 예측 불가능한 문제, 유연성 중요 |

LangGraph는 두 가지 모두에 **지속성(persistence), 스트리밍, 디버깅, 배포**를 제공합니다.

---

## 공통 설정

```bash
pip install langchain_core langchain-anthropic langgraph
```

```python
import os
from langchain_anthropic import ChatAnthropic

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
llm = ChatAnthropic(model="claude-sonnet-4-6")
```

---

## 패턴 1: 프롬프트 체이닝 (Prompt Chaining)

각 LLM 호출이 이전 호출의 출력을 처리합니다.

**적합한 경우:**
- 문서 다국어 번역
- 생성된 콘텐츠의 일관성 검증
- 단계적 콘텐츠 개선

```
START → generate_joke → (조건 분기) → improve_joke → polish_joke → END
                              ↓ (통과 시)
                             END
```

### Graph API 구현

```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

def generate_joke(state: State):
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}

def check_punchline(state: State):
    """조건부 엣지: 유머 품질 확인"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

def polish_joke(state: State):
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke",
    check_punchline,
    {"Fail": "improve_joke", "Pass": END}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

chain = workflow.compile()
state = chain.invoke({"topic": "cats"})
```

### Functional API 구현

```python
from langgraph.func import entrypoint, task

@task
def generate_joke(topic: str):
    return llm.invoke(f"Write a short joke about {topic}").content

@task
def improve_joke(joke: str):
    return llm.invoke(f"Make this joke funnier by adding wordplay: {joke}").content

@task
def polish_joke(joke: str):
    return llm.invoke(f"Add a surprising twist to this joke: {joke}").content

@entrypoint()
def prompt_chaining_workflow(topic: str):
    original_joke = generate_joke(topic).result()

    if "?" not in original_joke and "!" not in original_joke:
        # 조건 미달: 개선 과정 추가
        improved_joke = improve_joke(original_joke).result()
        return polish_joke(improved_joke).result()

    return original_joke  # 바로 통과
```

---

## 패턴 2: 병렬화 (Parallelization)

여러 독립적인 작업을 동시에 실행합니다.

**적합한 경우:**
- 같은 입력으로 여러 다른 결과물 생성
- 독립적인 여러 데이터 소스 조회
- 동일 작업을 여러 번 실행해 결과 비교

```
         ┌→ call_llm_1 (joke) ─┐
START ───┼→ call_llm_2 (story)─┼→ aggregator → END
         └→ call_llm_3 (poem) ─┘
```

### Graph API 구현

```python
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

def call_llm_1(state: State):
    return {"joke": llm.invoke(f"Write a joke about {state['topic']}").content}

def call_llm_2(state: State):
    return {"story": llm.invoke(f"Write a story about {state['topic']}").content}

def call_llm_3(state: State):
    return {"poem": llm.invoke(f"Write a poem about {state['topic']}").content}

def aggregator(state: State):
    combined = f"STORY:\n{state['story']}\n\nJOKE:\n{state['joke']}\n\nPOEM:\n{state['poem']}"
    return {"combined_output": combined}

parallel_builder = StateGraph(State)
parallel_builder.add_node("call_llm_1", call_llm_1)
parallel_builder.add_node("call_llm_2", call_llm_2)
parallel_builder.add_node("call_llm_3", call_llm_3)
parallel_builder.add_node("aggregator", aggregator)

# START에서 세 노드로 → 자동으로 병렬 실행
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
# 세 노드 모두 완료 후 aggregator 실행
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
parallel_builder.add_edge("aggregator", END)

parallel_workflow = parallel_builder.compile()
```

### Functional API 구현

```python
@task
def call_llm_1(topic: str):
    return llm.invoke(f"Write a joke about {topic}").content

@task
def call_llm_2(topic: str):
    return llm.invoke(f"Write a story about {topic}").content

@task
def call_llm_3(topic: str):
    return llm.invoke(f"Write a poem about {topic}").content

@task
def aggregator(topic, joke, story, poem):
    return f"STORY:\n{story}\n\nJOKE:\n{joke}\n\nPOEM:\n{poem}"

@entrypoint()
def parallel_workflow(topic: str):
    # 세 작업을 동시에 시작
    joke_fut = call_llm_1(topic)
    story_fut = call_llm_2(topic)
    poem_fut = call_llm_3(topic)

    # 모두 완료 후 집계
    return aggregator(
        topic,
        joke_fut.result(),
        story_fut.result(),
        poem_fut.result()
    ).result()
```

---

## 패턴 3: 라우팅 (Routing)

입력 내용에 따라 적절한 처리 경로로 분기합니다.

**적합한 경우:**
- 사용자 요청 유형에 따른 다른 처리
- 문서 분류 후 분야별 전문가 처리
- 언어 감지 후 해당 언어 처리기로 라우팅

```
START → router → (poem 요청) → poem_node → END
                 (story 요청) → story_node → END
                 (joke 요청) → joke_node → END
```

### Graph API 구현

```python
from typing_extensions import Literal
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        description="The next step in the routing process"
    )

# 구조화된 출력으로 라우터 생성
router = llm.with_structured_output(Route)

class State(TypedDict):
    input: str
    decision: str
    output: str

def llm_call_1(state: State):  # poem
    return {"output": llm.invoke(state["input"]).content}

def llm_call_2(state: State):  # story
    return {"output": llm.invoke(state["input"]).content}

def llm_call_3(state: State):  # joke
    return {"output": llm.invoke(state["input"]).content}

def llm_call_router(state: State):
    """LLM이 입력을 분류"""
    decision = router.invoke([
        SystemMessage(content="Route the input to story, joke, or poem based on the user's request."),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

def route_decision(state: State):
    """분류 결과에 따라 노드 선택"""
    if state["decision"] == "story": return "llm_call_1"
    elif state["decision"] == "joke": return "llm_call_2"
    elif state["decision"] == "poem": return "llm_call_3"

router_builder = StateGraph(State)
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {"llm_call_1": "llm_call_1", "llm_call_2": "llm_call_2", "llm_call_3": "llm_call_3"}
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

router_workflow = router_builder.compile()
state = router_workflow.invoke({"input": "Write me a joke about cats"})
```

---

## 패턴 4: 오케스트레이터-워커 (Orchestrator-Worker)

오케스트레이터가 작업을 분해하고 워커에게 위임하며, 결과를 합성합니다.

**적합한 경우:**
- 서브 작업을 미리 알 수 없는 경우 (병렬화와 다른 점!)
- 여러 파일에 걸쳐 코드 작성
- 보고서의 각 섹션을 독립적으로 작성

```
START → orchestrator → (Send API로 동적 분배)
           ↓
    [worker_1] [worker_2] [worker_3] ... (동시 실행)
           ↓
       synthesizer → END
```

### Graph API + Send API 구현

```python
from typing import Annotated, List
import operator
from langgraph.types import Send
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage, SystemMessage

class Section(BaseModel):
    name: str = Field(description="Name for this section of the report.")
    description: str = Field(description="Brief overview of the main topics.")

class Sections(BaseModel):
    sections: List[Section] = Field(description="Sections of the report.")

planner = llm.with_structured_output(Sections)

class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]  # 워커들이 병렬로 여기에 씀
    final_report: str

class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]

def orchestrator(state: State):
    """작업 계획 수립"""
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Report topic: {state['topic']}"),
    ])
    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """각 섹션 작성 (워커)"""
    section = llm.invoke([
        SystemMessage(content="Write a report section. Use markdown formatting."),
        HumanMessage(content=f"Section: {state['section'].name}\nDescription: {state['section'].description}"),
    ])
    return {"completed_sections": [section.content]}

def synthesizer(state: State):
    """완성된 섹션들 합치기"""
    combined = "\n\n---\n\n".join(state["completed_sections"])
    return {"final_report": combined}

def assign_workers(state: State):
    """Send API로 각 섹션에 워커 동적 할당"""
    return [Send("llm_call", {"section": s}) for s in state["sections"]]

orchestrator_worker_builder = StateGraph(State)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()
state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})
print(state["final_report"])
```

---

## 패턴 5: 평가자-최적화 (Evaluator-Optimizer)

하나의 LLM이 결과를 생성하고, 다른 LLM이 평가합니다. 평가 결과가 나쁘면 피드백과 함께 재생성합니다.

**적합한 경우:**
- 명확한 성공 기준이 있지만 반복이 필요한 경우
- 글쓰기 품질 개선
- 코드 정확성 검증

```
START → generator → evaluator → (통과) → END
             ↑         ↓ (탈락 + 피드백)
             └─────────┘
```

### Graph API 구현

```python
from typing import Literal
from pydantic import BaseModel, Field

class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str

class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Is the joke funny?")
    feedback: str = Field(description="If not funny, how to improve it.")

evaluator = llm.with_structured_output(Feedback)

def llm_call_generator(state: State):
    """농담 생성 (피드백 있으면 반영)"""
    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a joke about {state['topic']} "
            f"but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def llm_call_evaluator(state: State):
    """농담 평가"""
    grade = evaluator.invoke(f"Grade the joke: {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}

def route_joke(state: State):
    """평가 결과에 따라 분기"""
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

optimizer_builder = StateGraph(State)
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {"Accepted": END, "Rejected + Feedback": "llm_call_generator"}
)

optimizer_workflow = optimizer_builder.compile()
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])
```

---

## 패턴 6: 에이전트 (Agent)

LLM이 어떤 도구를 사용할지 스스로 결정하며 목표를 달성합니다.

**적합한 경우:**
- 문제와 해결책이 예측 불가능한 경우
- 여러 도구를 조합해야 하는 복잡한 작업
- 자율적인 작업 수행이 필요한 경우

```
START → llm_call → (도구 호출 없음) → END
             ↑          ↓ (도구 호출)
             └──── tool_node
```

### Graph API 구현

```python
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`. Args: a: First int, b: Second int"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`. Args: a: First int, b: Second int"""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`. Args: a: First int, b: Second int"""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

def llm_call(state: MessagesState):
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant for arithmetic.")]
                + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

messages = [HumanMessage(content="Add 3 and 4.")]
result = agent.invoke({"messages": messages})
for m in result["messages"]:
    m.pretty_print()
```

---

## 실전 예제: 고객 지원 이메일 에이전트

복잡한 실제 워크플로우 구현 예시입니다. **5단계 접근법**을 보여줍니다.

### 요구사항

```
- 수신 이메일 읽기
- 긴급도와 주제별 분류
- 관련 문서 검색해 질문 답변
- 적절한 응답 초안 작성
- 복잡한 이슈는 사람 에이전트에게 에스컬레이션
- 필요시 후속 조치 스케줄링
```

### Step 1: 상태 설계 (State Design)

> **핵심 원칙: 원시 데이터를 저장하고, 포맷은 노드 안에서 수행하라**

```python
from typing import TypedDict, Literal

class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # 원시 이메일 데이터
    email_content: str
    sender_email: str
    email_id: str

    # 분류 결과
    classification: EmailClassification | None

    # 원시 검색/API 결과
    search_results: list[str] | None
    customer_history: dict | None

    # 생성된 콘텐츠
    draft_response: str | None
    messages: list[str] | None
```

### Step 2: 노드에서 에러 처리

| 에러 유형 | 처리 방법 | 예시 |
|---|---|---|
| **일시적 에러** (네트워크, 레이트 리밋) | 재시도 정책 | `RetryPolicy(max_attempts=3)` |
| **LLM 복구 가능** (도구 실패, 파싱) | 에러를 상태에 저장하고 루프백 | `Command(update={"error": e}, goto="agent")` |
| **사용자 수정 필요** (정보 누락) | `interrupt()`로 일시 중단 | 고객 ID 누락 시 요청 |
| **예상치 못한 에러** | 그대로 전파 | `raise` |

```python
from langgraph.types import RetryPolicy

# 일시적 에러: 재시도 정책
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)

# 사용자 수정 필요: interrupt()
from langgraph.types import interrupt

def lookup_customer_history(state: State) -> Command[Literal["draft_response"]]:
    if not state.get('customer_id'):
        user_input = interrupt({
            "message": "Customer ID needed",
            "request": "Please provide the customer's account ID"
        })
        return Command(
            update={"customer_id": user_input['customer_id']},
            goto="lookup_customer_history"  # 재시도
        )
    customer_data = fetch_customer_history(state['customer_id'])
    return Command(update={"customer_history": customer_data}, goto="draft_response")
```

### Step 3: 핵심 노드 구현

```python
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """이메일 의도 분류 후 라우팅"""
    structured_llm = llm.with_structured_output(EmailClassification)
    classification = structured_llm.invoke(
        f"Analyze this email:\n{state['email_content']}\nFrom: {state['sender_email']}"
    )

    # 라우팅 로직
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug':
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(update={"classification": classification}, goto=goto)

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """사람 검토를 위해 interrupt()로 일시 중단"""
    human_decision = interrupt({
        "email_id": state.get('email_id', ''),
        "original_email": state.get('email_content', ''),
        "draft_response": state.get('draft_response', ''),
        "urgency": state.get('classification', {}).get('urgency'),
        "action": "Please review and approve/edit this response"
    })

    if human_decision.get("approved"):
        return Command(
            update={"draft_response": human_decision.get("edited_response", state.get('draft_response', ''))},
            goto="send_reply"
        )
    else:
        return Command(update={}, goto=END)
```

### Step 4: 그래프 조립

```python
from langgraph.checkpoint.memory import MemorySaver

workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("search_documentation", search_documentation,
                  retry_policy=RetryPolicy(max_attempts=3))
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

# 체크포인터로 상태 영속성 활성화
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

### Step 5: 실행 및 Human-in-the-loop

```python
# 초기 실행
initial_state = {
    "email_content": "I was charged twice for my subscription! This is urgent!",
    "sender_email": "customer@example.com",
    "email_id": "email_123",
    "messages": []
}

config = {"configurable": {"thread_id": "customer_123"}}
result = app.invoke(initial_state, config)

# human_review 노드에서 중단됨
print(result['__interrupt__'])

# 사람이 검토 후 재개
from langgraph.types import Command

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize for the double charge. Refund initiated."
    }
)
final_result = app.invoke(human_response, config)
print("Email sent successfully!")
```

---

## API 선택 가이드

### Graph API를 선택할 때

- **복잡한 의사결정 트리**: 여러 분기 조건
- **여러 컴포넌트 간 상태 공유**: TypedDict로 명시적 관리
- **병렬 처리 + 동기화**: 여러 경로가 합류하는 경우
- **팀 개발**: 시각적 그래프로 구조 공유
- **디버깅이 중요**: 그래프 시각화 활용

### Functional API를 선택할 때

- **기존 Python 코드 통합**: 최소한의 변경
- **단순 선형 워크플로우**: 복잡한 분기 없음
- **빠른 프로토타이핑**: 적은 보일러플레이트
- **함수 스코프 상태**: 명시적 상태 정의 불필요

### 두 API 함께 사용하기

```python
from langgraph.graph import StateGraph
from langgraph.func import entrypoint

# 복잡한 멀티 에이전트 조율: Graph API
coordination_graph = StateGraph(CoordinationState)
coordination_graph.add_node("orchestrator", orchestrator_node)

# 간단한 데이터 처리: Functional API
@entrypoint()
def data_processor(raw_data: dict) -> dict:
    cleaned = clean_data(raw_data).result()
    transformed = transform_data(cleaned).result()
    return transformed

# Graph 내에서 Functional API 결과 활용
def orchestrator_node(state):
    processed_data = data_processor.invoke(state["raw_data"])
    return {"processed_data": processed_data}
```

---

## 전체 패턴 요약

| 패턴 | 특징 | 적합한 케이스 |
|---|---|---|
| **프롬프트 체이닝** | A→B→C 순차 처리 | 단계적 변환, 번역, 검증 |
| **병렬화** | 동시 실행 후 집계 | 독립적 다중 작업 |
| **라우팅** | 분류 후 적절한 경로 선택 | 요청 유형별 처리 |
| **오케스트레이터-워커** | 동적 작업 분배 | 서브 작업 수를 모르는 경우 |
| **평가자-최적화** | 생성 → 평가 → 재생성 루프 | 품질 기준 있는 반복 개선 |
| **에이전트** | LLM이 자율적으로 도구 선택 | 예측 불가능한 복잡한 문제 |

---

## 학습 정리

LangGraph의 핵심은 다음 세 가지입니다:

1. **그래프 구조**: 노드(작업)와 엣지(흐름)로 에이전트 정의
2. **공유 상태**: 모든 노드가 읽고 쓰는 중앙 데이터 저장소
3. **내구성**: 체크포인터로 상태 저장, interrupt()로 인간 개입, 장애 시 재개

이 세 가지를 이해하면 LangGraph로 어떤 복잡한 에이전트도 구축할 수 있습니다.
