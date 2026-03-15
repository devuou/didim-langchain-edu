# LangChain 이해도 확인 시험

> 교육 기준 문서: https://docs.langchain.com/oss/python/langchain/
> 대상: Agent, Tools, Memory, Middleware 전 섹션

---

## 📋 시험 안내

- 총 15문항 (객관식 10 + 단답/코드교정 5)
- 각 문항 배점 표기
- 공식 문서를 직접 읽고 판단할 수 있어야 함

---

## Part 1. 객관식 (각 6점, 총 60점)

### Q1. `@tool` 데코레이터 적용 위치 ★★★

다음 중 LangChain `@tool` 데코레이터를 **올바르게** 사용한 코드는?

```python
# (A)
class MedicalAgent:
    @tool
    def search_hospital(self, region: str) -> str:
        """병원을 검색합니다."""
        return f"{region} 병원 목록"

# (B)
@tool
def search_hospital(region: str) -> str:
    """병원을 검색합니다."""
    return f"{region} 병원 목록"

# (C)
def search_hospital(region: str) -> str:
    """병원을 검색합니다."""
    return f"{region} 병원 목록"
tool(search_hospital)  # 나중에 변환

# (D)
class MedicalAgent:
    def __init__(self):
        @tool
        def search_hospital(region: str) -> str:
            """병원을 검색합니다."""
            return f"{region} 병원 목록"
        self.search_fn = search_hospital
```

**정답**: **(B)**

> **풀이**: 공식 문서(Tools 페이지)에 따르면 `@tool` 데코레이터는 **모듈 레벨의 일반 함수**에 적용하는 것이 기본 사용법이다.
> - **(A)**: 클래스 메서드에 `@tool`을 직접 적용하면 `self` 파라미터가 tool 스키마에 노출되어 LLM이 `self`를 인자로 채우려 한다. 이는 Q7에서 더 자세히 다룬다.
> - **(B)**: 올바른 사용법. 모듈 레벨의 일반 함수에 `@tool`을 적용한다.
> - **(C)**: `tool(search_hospital)`의 반환값을 변수에 할당하지 않았으므로, `search_hospital`은 여전히 일반 함수로 남는다. `search_hospital = tool(search_hospital)` 형태로 써야 동작하며, 이는 `@tool` 데코레이터와 동일하다.
> - **(D)**: `__init__` 내부에 정의된 tool은 인스턴스마다 재생성되고 관리가 까다롭다. 공식 문서 패턴이 아니다.

---

### Q2. Tool docstring의 역할 ★★★

LangChain에서 `@tool` 데코레이터를 사용할 때 함수의 docstring이 하는 역할로 **가장 정확한** 것은?

**(A)** 개발자가 함수의 동작을 문서화하기 위한 주석일 뿐이다.  
**(B)** LLM이 어떤 Tool을 사용할지 결정하는 근거(메타데이터)로 사용된다.  
**(C)** 런타임에서 타입 검사에 활용된다.  
**(D)** 사용자에게 UI로 표시되는 도움말 텍스트다.

**정답**: **(B)**

> **풀이**: 공식 문서(Tools 페이지)에 다음과 같이 명시되어 있다:
> _"By default, the function's docstring becomes the tool's description that helps the model understand when to use it"_
> docstring은 LLM에게 전달되는 tool description이 되며, **LLM이 어떤 상황에 어떤 tool을 선택할지 판단하는 핵심 메타데이터**로 사용된다.

---

### Q3. Checkpointer 싱글턴 관리 ★★★

다음 코드에서 **버그**가 있는 부분을 고르시오.

```python
# 코드 A
_checkpointer = MemorySaver()

def create_agent_instance(model, tools):
    return create_agent(model, tools, checkpointer=_checkpointer)
```

```python
# 코드 B
def create_agent_instance(model, tools):
    checkpointer = MemorySaver()  # (★)
    return create_agent(model, tools, checkpointer=checkpointer)
```

**(A)** 코드 A — `_checkpointer`를 모듈 레벨에 두면 여러 요청이 메모리를 공유해 버그가 생긴다.  
**(B)** 코드 B — `MemorySaver()`를 매 호출마다 생성하면 `thread_id` 기반 대화 연속성이 파괴된다.  
**(C)** 두 코드 모두 정상이다.  
**(D)** 두 코드 모두 버그가 있다.

**정답**: **(B)**

> **풀이**: 공식 문서(Short-term memory 페이지)의 예제를 보면 `InMemorySaver()` / `MemorySaver()`는 **한 번 생성하여 공유**하는 패턴을 사용한다.
> - **코드 A (정상)**: `_checkpointer`를 모듈 레벨에 싱글턴으로 두는 것이 올바른 패턴이다. 여러 thread_id로 격리되므로 요청 간 메모리가 섞이지 않는다.
> - **코드 B (버그)**: 함수 호출마다 새로운 `MemorySaver()` 인스턴스를 생성하면, 이전 대화 기록이 담긴 체크포인터가 매번 초기화된다. `thread_id`를 전달해도 새 인스턴스는 이전 대화를 기억하지 못하므로 **멀티턴 대화 연속성이 파괴된다**.

---

### Q4. 올바른 import 경로 ★★

다음 중 LangChain v1 공식 문서 기준으로 **권장되는** import 경로를 모두 고르시오. (복수 선택)

**(A)** `from langchain.tools import tool`  
**(B)** `from langchain_core.tools import tool`  
**(C)** `from langchain.agents.structured_output import ToolStrategy`  
**(D)** `from langchain.agents import create_agent`  
**(E)** `from langchain_core.messages import HumanMessage, SystemMessage`

**정답**: **(A), (C), (D), (E)**

> **풀이**: 공식 문서 코드 예제들을 직접 확인하면:
> - **(A)** `from langchain.tools import tool` → Tools 페이지 예제에서 직접 사용. ✅ 권장
> - **(B)** `from langchain_core.tools import tool` → 공식 문서 예제에서 사용되지 않음. `langchain_core`는 하위 패키지이며, 공식 문서는 `langchain.tools`를 사용.
> - **(C)** `from langchain.agents.structured_output import ToolStrategy` → Agents 및 Structured output 페이지 예제에서 직접 사용. ✅ 권장
> - **(D)** `from langchain.agents import create_agent` → Overview, Agents 페이지 예제에서 직접 사용. ✅ 권장
> - **(E)** `from langchain_core.messages import HumanMessage, SystemMessage` → Agents 페이지 예제(`from langchain.messages import ...` 또는 `from langchain_core.messages import ...`)에서 사용. ✅ 권장
>
> *참고*: (B)의 `langchain_core.tools`는 실제로 존재하는 패키지이나 공식 문서가 권장하는 import 경로는 `langchain.tools`이다.

---

### Q5. `create_agent`의 `system_prompt` 파라미터 ★

다음 중 `create_agent()`의 `system_prompt` 파라미터에 대해 **올바른** 설명은?

**(A)** 반드시 `ChatPromptTemplate` 객체를 전달해야 한다.  
**(B)** 문자열(`str`) 또는 `SystemMessage` 객체 모두 전달 가능하다.  
**(C)** 리스트 형태의 메시지만 허용된다.  
**(D)** `system_prompt` 파라미터는 존재하지 않으며 미들웨어로만 설정해야 한다.

**정답**: **(B)**

> **풀이**: 공식 문서(Agents 페이지, "System prompt" 섹션)에 다음과 같이 명시되어 있다:
> _"The system_prompt parameter accepts either a str or a SystemMessage."_
> 문자열로 간단히 전달하거나, `SystemMessage` 객체를 전달하여 더 세밀하게 제어(예: Anthropic 프롬프트 캐싱)할 수 있다.

---

### Q6. `response_format`과 `ToolStrategy` ★★

LLM이 구조화된 응답(Pydantic 모델 형태)을 반환하도록 강제하려면 어떻게 해야 하는가?

```python
class ChatResponse(BaseModel):
    message_id: str
    content: str

# 방법 A
@tool
def ChatResponse(message_id: str, content: str) -> str:
    """최종 응답을 반환합니다."""
    return f"{message_id}: {content}"

agent = create_agent(model, tools=[search, ChatResponse])

# 방법 B
agent = create_agent(
    model,
    tools=[search],
    response_format=ToolStrategy(ChatResponse)
)
```

**(A)** 방법 A가 올바르다. `ChatResponse`를 tool로 등록하면 LLM이 반드시 호출한다.  
**(B)** 방법 B가 올바르다. `response_format=ToolStrategy(ChatResponse)`가 공식 API다.  
**(C)** 두 방법 모두 동일한 결과를 낸다.  
**(D)** 구조화 출력은 LangChain에서 지원하지 않는다.

**정답**: **(B)**

> **풀이**: 공식 문서(Agents 페이지 "Structured output" 섹션, Structured output 페이지)에 따르면:
> - `response_format=ToolStrategy(MySchema)` 또는 `response_format=ProviderStrategy(MySchema)`가 **공식 구조화 출력 API**이다.
> - 결과는 `result["structured_response"]`에 검증된 Pydantic 인스턴스로 반환된다.
> - **방법 A**는 `ChatResponse`를 일반 tool로 등록하는 것이며, LLM이 반드시 해당 tool을 호출한다는 보장이 없고, 반환값도 문자열이 되어 Pydantic 검증이 이루어지지 않는다.

---

### Q7. 클래스 메서드와 `@tool` ★★

아래 코드를 실행하면 어떤 문제가 발생하는가?

```python
class StockAgent:
    def __init__(self):
        self.api_key = "abc"

    @tool
    def get_stock_price(self, ticker: str) -> str:
        """주식 현재가를 조회합니다."""
        return f"{ticker}: 50000원"

agent_instance = StockAgent()
agent = create_agent(model, tools=[agent_instance.get_stock_price])
```

**(A)** 정상 동작한다. `@tool`은 클래스 메서드에도 적용 가능하다.  
**(B)** `self` 파라미터가 LangChain tool 스키마에 노출되어 LLM이 `self`를 인자로 채우려 시도한다.  
**(C)** `@tool`은 `async def`에만 사용 가능하다.  
**(D)** `create_agent`에 메서드를 직접 전달하면 항상 오류가 발생한다.

**정답**: **(B)**

> **풀이**: 공식 문서(Tools 페이지)에서 `@tool`은 **모듈 레벨의 일반 함수**에 적용하도록 안내한다.
> 클래스 메서드에 `@tool`을 적용하면 Python의 메서드 바인딩 전에 데코레이터가 적용되므로, `self` 파라미터가 tool 입력 스키마에 포함되어 버린다. LLM은 `self`가 무엇인지 알 수 없으므로 이를 채우려 하거나 오류를 낸다. 올바른 해결책은 메서드를 클래스 외부의 일반 함수로 분리하거나, `__init__` 내에서 클로저(closure)로 캡처하는 것이다.

---

### Q8. Checkpointer와 멀티턴 대화 ★★★

`thread_id`를 활용한 멀티턴 대화를 구현할 때 **반드시** 필요한 것은?

**(A)** `create_agent()`에 `memory=True` 플래그 설정  
**(B)** `create_agent()`에 `checkpointer` 파라미터로 `MemorySaver()` 등 전달 + 각 invoke에 `{"configurable": {"thread_id": "..."}}` 전달  
**(C)** 각 요청마다 이전 메시지를 직접 `messages` 리스트에 추가  
**(D)** LangChain v1은 멀티턴을 지원하지 않는다.

**정답**: **(B)**

> **풀이**: 공식 문서(Short-term memory 페이지, "Usage" 섹션) 코드 예제를 보면:
> ```python
> agent = create_agent(
>     "gpt-5",
>     tools=[get_user_info],
>     checkpointer=InMemorySaver(),
> )
> agent.invoke(
>     {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
>     {"configurable": {"thread_id": "1"}},
> )
> ```
> `checkpointer` 파라미터와 `thread_id`를 함께 사용하는 것이 공식 패턴이다. `memory=True` 같은 플래그는 존재하지 않는다.

---

### Q9. AI Hallucination 식별 ★★

다음 코드를 실행할 때 반드시 오류가 발생하는 것을 모두 고르시오. (복수 선택)

**(A)**
```python
from langchain.agents.structured_output import ToolStrategy
```

**(B)**
```python
from langchain_core.tools import tool
```

**(C)**
```python
from langchain.agents import create_agent
agent = create_agent(model, tools=[])
```

**(D)**
```python
from langchain_core.messages import HumanMessage
```

**(E)**
```python
response_fromat = ToolStrategy(ChatResponse)  # 오타
agent = create_agent(model, tools=[], response_format=response_fromat)
```

**정답**: **정답 없음**

> **풀이**: `tests/test_q9_imports.py`를 작성하여 5개 선지를 모두 실제 실행으로 검증하였다.
>
> ```bash
> uv run pytest tests/test_q9_imports.py -v -s
> ```
>
> 결과:
> ```
> test_A_langchain_agents_structured_output_ToolStrategy  (A) ✅ 정상 import 성공  PASSED
> test_B_langchain_core_tools_tool                        (B) ✅ 정상 import 성공  PASSED
> test_C_create_agent_with_empty_tools                    (C) ✅ 정상 import 성공  PASSED
> test_D_langchain_core_messages_HumanMessage             (D) ✅ 정상 import 성공  PASSED
> test_E_typo_variable_name_still_works                   (E) ✅ 문법 오류 없음    PASSED
> ```
>
> 5개 선지 모두 오류 없이 정상 실행되었다. 각 선지별 상세 결과는 다음과 같다:
>
> - **(A)** `from langchain.agents.structured_output import ToolStrategy` → 출제 의도상 존재하지 않는 경로로 설계된 것으로 보이나, 현재 설치된 langchain 버전에서 **실제로 존재하는 경로**이므로 오류가 발생하지 않는다.
> - **(B)** `from langchain_core.tools import tool` → 정상 동작.
> - **(C)** `from langchain.agents import create_agent` → 정상 동작.
> - **(D)** `from langchain_core.messages import HumanMessage` → 정상 동작.
> - **(E)** `response_fromat`은 변수명 오타이지만 Python 문법 오류가 아니다. `create_agent`의 `response_format=` 파라미터에 올바르게 전달되고 있어 런타임 오류도 발생하지 않는다.

---

### Q10. Tool docstring 품질 ★★

다음 두 tool 중 LLM이 올바른 상황에 정확하게 선택할 가능성이 더 높은 것은?

```python
# Tool A
@tool
def search(q: str) -> str:
    """검색"""
    ...

# Tool B
@tool
def search_hospital(region: str, specialty: str = "내과") -> str:
    """지역과 진료과목으로 병원을 검색합니다.
    
    Args:
        region: 검색할 지역명 (예: '서울', '강남구')
        specialty: 진료과목 (예: '내과', '정형외과', '소아청소년과')
    
    Returns:
        해당 지역의 병원 목록과 연락처
    """
    ...
```

**(A)** Tool A — 간결한 docstring이 LLM 토큰을 절약하므로 더 효율적이다.  
**(B)** Tool B — LLM은 docstring을 tool 선택의 근거로 사용하므로, 구체적일수록 정확도가 높다.  
**(C)** 두 tool의 선택 정확도는 동일하다. docstring은 tool 선택에 영향을 주지 않는다.  
**(D)** Tool A — 함수명만으로 LLM이 충분히 판단할 수 있다.

**정답**: **(B)**

> **풀이**: Q2의 연장선으로, docstring은 LLM이 tool 선택을 결정하는 메타데이터이다. 공식 문서는 _"The docstring should be informative and concise to help the model understand the tool's purpose"_ 라고 강조한다. Tool B처럼 파라미터 설명, 예시, 반환값까지 상세히 기술하면 LLM이 정확한 상황에서 올바른 tool을 선택할 확률이 높아진다. Tool A의 `"검색"` 한 단어로는 LLM이 이 tool의 용도를 제대로 파악하기 어렵다.

---

## Part 2. 단답 / 코드 교정 (각 8점, 총 40점)

### Q11. 코드 교정 ★★★

아래 코드에는 **3가지 버그**가 있습니다. 각각 찾아서 수정하시오.

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

class MedicalService:
    def __init__(self):
        self.checkpointer = MemorySaver()  # bug 1

    @tool  # bug 2
    def search_symptom(self, symptom: str) -> str:
        """증상으로 질병 정보를 검색합니다."""
        return f"{symptom} 관련 질병 정보"

    def create(self, model):
        return create_agent(
            model,
            tools=[self.search_symptom],
            response_fromat=ToolStrategy(ChatResponse)  # bug 3
        )
```

**Bug 1**: `self.checkpointer = MemorySaver()`

> **문제점**: `MemorySaver`를 import하지 않았거나, 인스턴스 속성으로 생성하면 `AgentService` 인스턴스마다 새로운 체크포인터가 만들어진다. 이렇게 되면 thread_id 기반 대화 연속성이 인스턴스 간에 공유되지 않는다.
>
> **수정**: `MemorySaver`를 올바르게 import하고, 클래스 외부(모듈 레벨)에서 싱글턴으로 생성하거나 클래스 변수로 공유해야 한다.
> ```python
> from langgraph.checkpoint.memory import InMemorySaver
>
> _checkpointer = InMemorySaver()  # 모듈 레벨 싱글턴
>
> class MedicalService:
>     def __init__(self):
>         self.checkpointer = _checkpointer  # 공유 인스턴스 참조
> ```

**Bug 2**: `@tool`을 클래스 인스턴스 메서드에 직접 적용

> **문제점**: `@tool`을 클래스 메서드에 적용하면 `self` 파라미터가 tool 스키마에 노출되어 LLM이 `self`를 인자로 채우려 시도한다.
>
> **수정**: `search_symptom`을 클래스 외부의 일반 함수로 분리하거나, 클래스 내에서 클로저로 정의한다.
> ```python
> @tool
> def search_symptom(symptom: str) -> str:
>     """증상으로 질병 정보를 검색합니다."""
>     return f"{symptom} 관련 질병 정보"
> ```

**Bug 3**: `response_fromat=` (오타)

> **문제점**: `response_fromat`은 오타이다. `create_agent`는 이를 알 수 없는 키워드 인자로 받아 무시하거나 오류를 낸다. 구조화 출력이 적용되지 않는다.
>
> **수정**: `response_format=ToolStrategy(ChatResponse)` (올바른 철자)
> ```python
> return create_agent(
>     model,
>     tools=[search_symptom],
>     response_format=ToolStrategy(ChatResponse)  # 오타 수정
> )
> ```

---

### Q12. 빈칸 채우기 ★★

공식 문서에 따르면, `create_agent()`의 `system_prompt` 파라미터는 `______` 또는 `______` 타입을 받을 수 있다.

또한 멀티턴 대화를 유지하려면 `create_agent()`에 `checkpointer=______()` 와 같이 **서버 전체에서 단 한 번** 생성된 인스턴스를 전달해야 한다.

**정답**:
- `system_prompt`가 받을 수 있는 타입: **`str`** 또는 **`SystemMessage`**
- 멀티턴 대화용 checkpointer: **`InMemorySaver`** (개발/테스트 환경) 또는 **`PostgresSaver`** (프로덕션 환경)

> **풀이**: 공식 문서(Agents 페이지) 인용:
> _"The system_prompt parameter accepts either a str or a SystemMessage."_
>
> Short-term memory 페이지 예제:
> ```python
> from langgraph.checkpoint.memory import InMemorySaver
> agent = create_agent("gpt-5", tools=[...], checkpointer=InMemorySaver())
> ```
> 프로덕션에서는 `PostgresSaver` 등 DB 기반 체크포인터를 사용하며, 반드시 **단일 인스턴스를 서버 전체에서 공유**해야 한다.

---

### Q13. 설계 오류 설명 ★★★

아래 `AgentService` 클래스의 **설계 문제점**을 서술하고, 올바르게 수정한 코드를 작성하시오.

```python
class AgentService:
    def process_query(self, query: str, thread_id: str) -> str:
        checkpointer = InMemorySaver()  # ← 문제
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=checkpointer
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content
```

**문제점**: `process_query` 메서드 내부에서 매 호출마다 `InMemorySaver()`를 새로 생성하고 있다. `InMemorySaver`는 메모리 내에 대화 기록을 저장하는데, 호출마다 새 인스턴스를 생성하면 이전 대화 내용이 모두 사라진다. 동일한 `thread_id`를 전달해도 새로운 체크포인터 인스턴스는 이전 기록을 알 수 없으므로 **멀티턴 대화 연속성이 완전히 파괴**된다. 또한 매 호출마다 `create_agent`를 재실행하는 것도 비효율적이다.

**수정 코드**:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

# 체크포인터는 모듈 레벨 또는 애플리케이션 시작 시 단 한 번 생성
_checkpointer = InMemorySaver()

class AgentService:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        # agent도 한 번만 생성하여 재사용 (효율적)
        self._agent = create_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=_checkpointer  # 공유 싱글턴 사용
        )

    def process_query(self, query: str, thread_id: str) -> str:
        result = self._agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content
```

> **핵심**: 공식 문서는 `checkpointer`를 **한 번 생성된 인스턴스**로 사용할 것을 명시한다. 프로덕션에서는 `InMemorySaver` 대신 `PostgresSaver` 등 영속성 있는 체크포인터를 사용한다.

---

### Q14. import 경로 교정 ★★

아래에서 **잘못된 import**를 찾아 올바르게 수정하시오. (존재하지 않는 경로이거나 deprecated된 경로)

```python
from langchain.agents.structured_output import ToolStrategy   # (1)
from langchain.tools import tool                               # (2)
from langchain_core.messages import HumanMessage              # (3)
from langchain.agents import create_agent                     # (4)
from langchain_core.tools import tool as core_tool            # (5)
```

`tests/test_q14_imports.py`를 작성하여 5개 항목을 모두 실제 실행으로 검증하였다.

```bash
uv run pytest tests/test_q14_imports.py -v -s
```

결과:
```
test_1_langchain_agents_structured_output_ToolStrategy  (1) ✅ 정상 import 성공  PASSED
test_2_langchain_tools_tool                             (2) ✅ 정상 import 성공  PASSED
test_3_langchain_core_messages_HumanMessage             (3) ✅ 정상 import 성공  PASSED
test_4_langchain_agents_create_agent                    (4) ✅ 정상 import 성공  PASSED
test_5_langchain_core_tools_tool                        (5) ✅ 정상 import 성공  PASSED
```

5개 항목 모두 오류 없이 정상 실행되었다. 교정이 필요한 잘못된 import 경로가 없다.

**(1)** 상태: ✅ **정상** — 실제로 존재하는 경로. (`stock_agent.py`에서도 사용 중)

**(2)** 상태: ✅ **정상** — 공식 문서 Tools 페이지 예제에서 직접 사용하는 경로.

**(3)** 상태: ✅ **정상** — `langchain_core.messages`는 표준 경로.

**(4)** 상태: ✅ **정상** — 공식 문서 전반에서 사용하는 핵심 import.

**(5)** 상태: ✅ **정상** — `langchain_core.tools`도 유효한 경로. 공식 문서에서 더 자주 등장하는 권장 경로는 `langchain.tools`이나, 두 경로 모두 동작한다.

---

### Q15. 자유 서술 ★★

LangChain에서 **`response_format=ToolStrategy(ChatResponse)`** 와 **`ChatResponse`를 `@tool`로 직접 정의** 하는 방식의 차이를 설명하시오. 공식 API 관점에서 어느 쪽이 권장되며, 그 이유는 무엇인가?

**답변**:

`response_format=ToolStrategy(ChatResponse)`는 LangChain의 **공식 구조화 출력 API**이다. 이 방식은 에이전트 내부적으로 `ChatResponse` Pydantic 모델을 tool 호출 방식으로 강제하고, LLM이 해당 스키마에 맞는 응답을 생성하면 이를 자동으로 검증하여 `result["structured_response"]`에 검증된 Pydantic 인스턴스로 반환한다. 스키마 불일치 시 자동 재시도(retry) 로직이 작동하고, `handle_errors` 파라미터로 오류 처리 전략을 세밀하게 제어할 수 있다.

반면 `ChatResponse`를 `@tool`로 직접 등록하는 방식은 LLM이 해당 tool을 반드시 호출한다는 보장이 없으며, tool이 호출되더라도 반환값은 문자열(`str`)이 되어 Pydantic 검증이 이루어지지 않는다. 또한 tool 함수명과 Pydantic 모델명이 충돌하여 코드 가독성과 유지보수성이 저하된다.

따라서 공식 API 관점에서 **`response_format=ToolStrategy(ChatResponse)`가 권장**된다. 이는 구조화 출력을 위한 전용 메커니즘으로, 검증, 재시도, 오류 처리가 통합되어 있어 프로덕션 환경에서 안정적인 구조화 응답을 보장한다. 모델이 provider-native 구조화 출력을 지원하는 경우(OpenAI, Anthropic 등)에는 `response_format=ProviderStrategy(ChatResponse)` 또는 `response_format=ChatResponse`를 사용하는 것이 더욱 권장된다.