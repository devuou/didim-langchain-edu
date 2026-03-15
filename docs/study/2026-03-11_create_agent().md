# `create_agent()` 함수 분석

**출처**: `langchain/agents/factory.py`
**import**: `from langchain.agents import create_agent`

---

## 개요

LangGraph 기반의 ReAct 에이전트 그래프를 생성하는 팩토리 함수입니다.
내부적으로 **model → tools → model 루프**를 구성하고, `tool_calls`가 없어질 때까지 반복한 뒤 최종 응답을 반환합니다.
반환값은 `CompiledStateGraph`로, `.astream()` 등을 바로 호출할 수 있습니다.

---

## 함수 시그니처

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | type | dict | None = None,
    state_schema: type[AgentState] | None = None,
    context_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
```

---

## 파라미터 설명

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `model` | `str \| BaseChatModel` | 에이전트의 두뇌. `"openai:gpt-4o"` 같은 문자열 또는 `ChatOpenAI` 인스턴스 직접 전달 |
| `tools` | `list` | 에이전트가 호출할 수 있는 도구 목록. `None`이면 tool loop 없이 단순 LLM 호출로만 동작 |
| `system_prompt` | `str \| SystemMessage` | LLM에 주입할 시스템 프롬프트. 매 호출 시 메시지 목록 맨 앞에 자동 추가 |
| `response_format` | `ToolStrategy \| ProviderStrategy \| Pydantic` | 최종 응답의 구조화 방식 지정. `ToolStrategy(ChatResponse)`를 전달하면 tool calling 방식으로 구조화된 응답 강제 |
| `middleware` | `Sequence[AgentMiddleware]` | 에이전트 실행 단계에 끼어드는 미들웨어. 로깅, 검증, 행동 수정 등에 활용 |
| `state_schema` | `TypedDict` | `AgentState`를 확장하는 커스텀 상태 스키마. 미들웨어 없이 상태 필드를 추가하고 싶을 때 사용 |
| `context_schema` | `TypedDict` | 런타임 컨텍스트 스키마 |
| `checkpointer` | `BaseCheckpointSaver` | 대화 이력 저장소. thread_id별로 상태를 저장해 멀티턴 대화 가능. `MemorySaver`, `AsyncSqliteSaver` 등 |
| `store` | `BaseStore` | 여러 thread(대화) 간 공유 데이터 저장소. checkpointer가 단일 대화 내 이력이라면, store는 사용자 전체에 걸친 장기 메모리 |
| `interrupt_before` | `list[str]` | 지정한 노드 실행 **전**에 일시 정지. 사용자 확인(Human-in-the-loop) 구현에 활용 |
| `interrupt_after` | `list[str]` | 지정한 노드 실행 **후**에 일시 정지 |
| `debug` | `bool` | `True`이면 각 노드 실행, 상태 변화, 전환 과정을 상세 로깅 |
| `name` | `str` | 생성된 그래프의 이름. 멀티 에이전트 시스템에서 서브그래프로 추가할 때 노드 이름으로 자동 사용 |
| `cache` | `BaseCache` | 그래프 실행 결과 캐싱. 동일 입력 반복 시 LLM 호출 없이 캐시 반환 |

---

## 동작 흐름

```
[START]
  ↓
[model 노드] — LLM 호출, system_prompt 자동 주입
  ↓ tool_calls 있음
[tools 노드] — tools 실행, ToolMessage 생성
  ↓
[model 노드] — 반복
  ↓ tool_calls 없음 (또는 response_format 조건 충족)
[END] — CompiledStateGraph 반환
```

`response_format=ToolStrategy(ChatResponse)` 전달 시, LLM이 최종 답변을 `ChatResponse` 스키마에 맞춰 tool calling 형태로 반환하도록 강제합니다.

---

## 현재 프로젝트에서의 사용

```python
# app/agents/stock_agent.py

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

def create_stock_agent(model: ChatOpenAI, checkpointer=None):
    return create_agent(
        model=model,                                              # AgentService에서 초기화한 ChatOpenAI
        tools=[get_stock_price, get_company_info, get_recent_news],  # 3개 stock tool
        system_prompt=system_prompt,                              # prompts.py의 시스템 프롬프트
        response_format=ToolStrategy(ChatResponse),              # @dataclass ChatResponse로 최종 응답 구조화
        checkpointer=checkpointer,                               # MemorySaver (thread_id별 대화 이력)
    )
```

### 미사용 파라미터 (향후 활용 가능)

| 파라미터 | 활용 시나리오 |
|---|---|
| `store` | 사용자별 즐겨찾기 종목, 투자 성향 등 장기 메모리 저장 |
| `interrupt_before` | 매수/매도 추천 전 사용자 확인 단계 삽입 |
| `middleware` | 요청/응답 로깅, 입력 필터링 |
| `name` | 멀티 에이전트 확장 시 서브그래프 노드 이름 지정 |
