# 주식 분석 AI Agent 개발 및 평가 과제

---

## 1. 프로젝트 개요

### 무엇을 만들었나?

자연어로 주식 정보를 질문하면 실시간 데이터를 조회하고 답변하는 AI 에이전트

```
사용자: "AAPL 현재 주가랑 최근 뉴스 알려줘"
에이전트: get_stock_price + get_recent_news 호출 → 답변
```

### 기술 스택

| 계층 | 기술 |
|---|---|
| API 서버 | FastAPI + SSE 스트리밍 |
| 에이전트 | LangGraph ReAct |
| LLM | OpenAI GPT-4o |
| 실시간 데이터 | yfinance |
| 히스토리컬 데이터 | Elasticsearch |
| 평가 | Opik |

**SSE(Server-Sent Events)란?**
- 서버에서 클라이언트로의 단방향 실시간 스트림.
- 에이전트가 "도구 선택 → 도구 실행 → 최종 답변" 각 단계를 완료될 때마다 즉시 클라이언트에 전송한다. 
- WebSocket과 달리 단방향이라 구현이 단순하고 HTTP 표준을 그대로 따른다.

**yfinance란?**
- Yahoo Finance의 주가 데이터를 무료로 조회할 수 있는 비공식 Python 라이브러리.
- API 키 없이 현재가, 재무 정보, 뉴스, 과거 OHLCV 데이터를 가져올 수 있다. 
- 공식 API가 아니기 때문에 Yahoo 서버 구조 변경 시 갑자기 동작하지 않을 수 있다는 한계가 있다.

---

## 2. 에이전트 도구(Tool) 구성

에이전트는 사용자 질문을 분석해 필요한 도구를 선택·조합하여 답변한다.

### Tool 목록 및 내부 동작

**`get_stock_price`** — 현재 주가 + 등락률
- `yfinance.Ticker(ticker).fast_info`에서 `last_price`, `previous_close`를 가져와 등락률 계산
- `yfinance.fast_info`: 전체 재무 데이터가 아닌 가격 관련 핵심 필드만 빠르게 조회하는 경량 API

**`get_company_info`** — 시가총액, PER, 업종
- `yfinance.Ticker(ticker).info`에서 `marketCap`, `trailingPE`, `sector` 필드 추출
- `.info`는 수십 가지 재무 지표를 한 번에 반환하는 전체 조회 API

**`get_recent_news`** — 최신 뉴스 최대 3건
- `yfinance.Ticker(ticker).news`에서 뉴스 목록 조회 (제목 + 링크)
- Yahoo Finance가 해당 티커와 관련이 있다고 판단한 기사를 가져오므로 경쟁사 기사가 섞일 수 있음

**`get_stock_history`** — OHLCV 히스토리컬 데이터
- yfinance로 수집한 4개 종목(AAPL·MSFT·TSLA·NVDA) 1년치 데이터를 앱 시작 시 Elasticsearch에 upsert
- 사용자 질문 시 `ElasticsearchRetriever`가 `term(ticker) + sort(date desc)` Query DSL로 최근 N일 데이터 조회
- ES에서 반환된 문서를 LLM이 읽기 좋은 자연어로 포맷(`[AAPL] 2026-03-13 | 시가:255.40 ...`)하여 컨텍스트로 전달

---

## 3. Opik Evaluation

### 평가의 목표

에이전트를 만들고 나서 "잘 동작한다"는 것을 어떻게 증명할 수 있을까?

- 수동으로 질문을 몇 개 해보는 방식: 재현성 없음, 기록도 없음, 개선됐는지 비교 불가
- 목표: **정량적 점수를 통해 에이전트 품질을 객관적으로 측정하고, 프롬프트 개선 전후를 수치로 비교하자**

### 평가 대상 질문

평가 범위를 설계할 때 두 가지 유형을 모두 포함했다.

| 유형 | 이유 |
|---|---|
| 에이전트가 잘 **답해야** 하는 질문 | 기능이 제대로 동작하는지 검증 |
| 에이전트가 잘 **거절해야** 하는 질문 | 범위 밖 요청(배당, 전망, 투자 추천)을 올바르게 처리하는지 검증 |

"잘 거절하는지"도 품질 지표다.

---

## 4. 메트릭 선정 및 설계

### 왜 이 3가지 메트릭인가?

| 메트릭 | 선정 이유 | 래핑한 Opik 내장 메트릭 |
|---|---|---|
| **Hallucination** | 에이전트가 조회하지 않은 수치를 지어냈는지 탐지 | `opik.evaluation.metrics.Hallucination` |
| **AnswerRelevance** | 주식 질문에 실제로 관련 있는 답변인지 확인 | `opik.evaluation.metrics.AnswerRelevance` |
| **TaskCompletion** | 사용자 요청(주가 조회, 거절 안내 등)을 기대 수준으로 완료했는지 | `opik.evaluation.metrics.GEval` |

### 각 메트릭의 내부 동작 방식

세 메트릭 모두 **LLM-as-a-Judge** 방식이다. 즉, 에이전트를 평가하기 위해 또 다른 LLM(기본값: gpt-4o-mini)을 내부적으로 호출한다. 따라서 평가 실행 자체에도 API 호출과 토큰 비용이 발생한다.

**Hallucination**
- judge LLM에게 "이 질문과 답변을 보고, 답변에 질문에서 근거를 찾을 수 없는 사실이 포함되어 있는가?"를 묻는 프롬프트를 생성해 전달
- 외부 사실 DB를 조회하는 것이 아니라, judge LLM 자신의 추론으로 "이 답변이 입력에 근거하는가"를 판단
- **0.0 = 환각 없음(좋음), 1.0 = 환각 감지(나쁨)** — 직관과 반대 방향이므로 주의

**AnswerRelevance**
- judge LLM에게 "이 질문에 대해 이 답변이 얼마나 관련 있는가?"를 묻는 프롬프트를 생성해 0~1로 채점
- 기본적으로 `context`(참조 문서) 필드를 요구하지만, 우리 에이전트는 RAG 구조가 아니므로 `require_context=False`로 비활성화

**TaskCompletion (GEval 기반)**
- `criteria`(평가 기준 설명)를 기반으로 judge LLM이 자유롭게 점수를 매기는 범용 평가
- GEval의 `score()` 시그니처는 `output` 하나만 받으므로, `input`과 `expected_output`을 페이로드 패킹으로 하나의 문자열로 묶어 전달
  ```python
  payload = f"QUESTION: {input}\nEXPECTED_OUTPUT: {expected_output}\nOUTPUT: {output}"
  self._inner.score(output=payload)
  ```

### 레벨별 평가 전략

| 레벨 | 용도 | 샘플 수 | 메트릭 |
|---|---|---|---|
| L1 | 빠른 검증 (개발 중 수시 실행) | 5건 | hallucination, answer_relevance |
| L2 | 전체 평가 (배포 전) | 15건 | hallucination, answer_relevance, task_completion |

LLM-as-a-Judge 메트릭은 API 호출 비용이 발생하므로 목적에 따라 범위를 조절했다.

### `expected_output` 설계 원칙

정확한 값이 아닌 **기대하는 답변 유형에 대한 설명**으로 작성. LLM 판사가 기준을 알고 채점할 수 있도록.

```json
{
  "input": "MSFT 배당 정보 알려줘",
  "expected_output": "배당 정보는 제공하지 않는다는 안내와 함께 제공 가능한 기능을 안내하는 답변"
}
```

---

## 5. 주요 트러블슈팅

### task_completion 평가

**1단계 — expected_output 없이 평가 (점수: 0.14)**

초기에 `AgentTaskCompletionJudge`를 사용하면서 `expected_output` 없이 에이전트 출력만 전달했다. judge LLM이 "기대하는 답변"에 대한 기준 없이 판단해야 하므로 15건 중 8건이 0점이었다.


**2단계 — expected_output 추가했지만 GEval이 무시 (점수: 0.55)**

작업 완료 여부를 단순하게 yes or no로만 판단하는 AgentTaskCompletionJudge는 이번 평가에 부적합하다고 생각되었다. GEval과 비교해보았다.

| 항목 | AgentTaskCompletionJudge | GEval |
|---|---|---|
| 평가 방식 | "작업을 완료했는가"를 단순 yes/no로 판단 | `criteria`(기준 설명)를 기반으로 연속 점수 산출 |
| expected_output 지원 | 미지원 (input + output만 사용) | criteria 또는 페이로드에 포함 가능 |
| 커스터마이징 | 제한적 — 기준을 직접 설정할 수 없음 | criteria 텍스트로 평가 기준을 자유롭게 정의 |
| 적합한 상황 | 명확한 성공/실패가 있는 단일 작업 | 도메인 특화 평가, 복합 기준이 필요한 경우 |

→ 주식 에이전트는 "잘 거절했는가", "필요한 정보를 모두 담았는가" 등 복합적인 기준이 필요하므로 GEval이 더 적합하다.

메트릭을 GEval로 교체하고 dataset에 `expected_output`을 추가했다. 점수는 올라갔지만 여전히 1건이 0점이었다. 원인을 확인해보니 GEval의 `score()` 시그니처가 `output: str, **ignored_kwargs`로 되어 있어, 전달한 `expected_output`이 내부적으로 아예 무시되고 있었다.

```python
# GEval 내부 시그니처 — 추가 인자는 사실상 버려짐
def score(self, output: str, **ignored_kwargs): ...
```

**3단계 — 페이로드 패킹으로 해결 (점수: 0.61, 0점 항목 0건)**

`input`과 `expected_output`을 `output` 하나의 문자열 안에 함께 담아 전달하는 방식으로 우회했다. judge LLM이 모든 정보를 읽고 채점할 수 있게 되었고 0점 항목이 사라졌다.

```python
payload = f"QUESTION: {input}\nEXPECTED_OUTPUT: {expected_output}\nOUTPUT: {output}"
self._inner.score(output=payload)
```

| 단계 | 변경 내용 | 점수 |
|---|---|---|
| 초기 | AgentTaskCompletionJudge, expected_output 없음 | 0.14 |
| 1차 | GEval 교체 | 0.37 |
| 2차 | expected_output 추가 (but GEval이 무시) | 0.55 |
| 3차 | 페이로드 패킹 적용 | **0.61** |

> 메트릭 교체보다 **데이터셋 설계(expected_output 추가)** 가 더 큰 개선 효과를 가져왔다.

---

## 6. 평가 결과에 따른 프롬프트 개선

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| `get_stock_history` | 언급 없어서 에이전트가 도구를 활용하지 못함 | 언제 사용할지 명시 |
| 수치 생성 금지 | 따로 명시하지 않아서 hallucination 점수 높음 | "도구 조회 없이 수치 직접 생성 금지" 추가 |
| 간결성 | 따로 명시하지 않아서 불필요한 설명이 task_completion 점수 낮춤 | 묻지 않은 개념 설명 금지 추가 |
| 범위 밖 질문 | 단순 거절만 해서 task_completion 기준 미충족 | 거절 + 제공 가능 기능 안내 명시 |

### 최종 평가 결과 (L2, 15건)

| 메트릭 | 점수 |
|---|---|
| `hallucination_metric` | 0.305 (낮을수록 좋음) |
| `answer_relevance_metric` | 0.804 |
| `stock_task_completion` | 0.610 |

---

## 7. 미해결 이슈

### **출처 오표기**: The Motley Fool을 "한 증권사"로 표현
- yfinance가 `publisher` 필드를 반환하지만 에이전트가 이를 활용하지 않고 기사 내용에서 추측함
- **개선 방향**: `get_recent_news` 도구가 제목·링크와 함께 `publisher` 필드도 포함해 반환 → 에이전트가 원본 출처명을 그대로 사용하게 유도

### **무관 기사 포함**: TSLA 뉴스 조회 시 Lucid(경쟁사) 기사가 포함됨
- yfinance가 관련 기사를 넓게 수집하기 때문에 발생하는 데이터 품질 문제
- **개선 방향**: 도구 레벨에서 기사 제목에 티커 심볼 또는 회사명이 포함된 것만 필터링, 또는 프롬프트에 "조회한 뉴스 중 해당 종목과 직접 관련된 기사만 포함하라" 명시

### Hallucination 메트릭이 실시간 데이터 에이전트에 부적합

L2 평가(hallucination_metric: 0.430) 항목별 분석 결과, 높은 점수를 받은 대부분이 에이전트가 도구를 호출해 가져온 정확한 데이터였다.

| 질문 | 점수 | judge LLM 판단 | 실제 |
|---|---|---|---|
| NVDA 오늘 주가 | 0.65 | 출처·타임스탬프 없어 검증 불가 | yfinance 실시간 데이터 |
| NVDA 7일 히스토리 | 0.85 | 2026년 3월 $180대는 말이 안 됨 | ES에 저장된 정확한 데이터 |
| AAPL PER | 0.74 | 출처 없는 정밀 수치 | yfinance 조회값 |

**원인**: `Hallucination` 메트릭은 RAG 시스템에서 "제공된 컨텍스트에 없는 내용을 지어냈는가"를 판단하도록 설계되어 있다. 도구 호출로 실시간 데이터를 가져오는 에이전트에서는 judge LLM이 수치의 출처를 알 수 없어 정확한 데이터도 환각으로 오판한다.

**개선 방향**: 도구 실행 결과(tool output)를 `context`로 메트릭에 함께 전달하면 judge가 "출력이 도구 결과와 일치하는가"로 판단할 수 있다. evaluation task에서 도구 실행 로그를 함께 캡처하는 구조 개선이 필요하다.

---

## 8. 배운 점

### LangChain의 create_agent()를 활용한 에이전트 개발

- LangGraph ReAct 에이전트에 데이터 특성에 맞는 도구를 연결했다. 실시간성이 필요한 현재가·뉴스·재무정보는 yfinance로 직접 조회하는 도구를 사용하고, 과거 OHLCV 데이터는 앱 시작 시 Elasticsearch에 적재한 뒤 ElasticsearchRetriever로 검색하는 도구를 별도로 구성했다. 데이터 특성(실시간 vs 과거)에 따라 저장소와 접근 방식을 분리한 설계가 핵심이었다.

### Opik을 활용한 에이전트 평가

- dataset + metric + evaluation task 세 요소로 구성되는 평가 구조를 이해했다. 평가 품질을 높이려면 메트릭 선택보다 **데이터셋 설계, 특히 `expected_output`** 이 더 중요하다. expected_output이 없으면 judge LLM이 기준 없이 판단하고, 있더라도 메트릭에 제대로 전달되지 않으면 효과가 없다. "잘 답하는 질문"뿐 아니라 "잘 거절해야 하는 질문"도 데이터셋에 포함해야 한다는 점도 배웠다.

### LLM-as-a-Judge 패턴의 개념과 한계

- **개념**: 또 다른 LLM(judge)이 평가자 역할을 맡아 출력 품질을 채점하는 방식. 규칙 기반으로 측정하기 어려운 "관련성", "환각 여부", "작업 완수" 같은 주관적 지표를 정량화할 수 있다.

- **한계**: judge LLM도 완벽하지 않아 동일한 출력에 대해 판단이 달라질 수 있다. 또한 평가 자체에도 API 호출과 토큰 비용이 발생하며, judge LLM 스스로가 환각을 일으켜 잘못된 점수를 줄 가능성도 있다. 이 때문에 LLM-as-a-Judge 점수는 절대적 기준이 아니라 상대적 비교(프롬프트 개선 전후)에 활용하는 것이 적합하다.

### 평가 결과에 따른 프롬프트 개선

- 점수만 보는 것이 아니라 낮은 점수의 원인을 분석하고 프롬프트에 반영하는 반복 개선이 핵심이다. 어떤 질문이 낮은 점수를 받았는지, 그 이유가 도구 미활용인지·불필요한 설명인지·거절 방식의 문제인지를 추적해야 프롬프트를 의미 있게 수정할 수 있다. 이번 경험을 통해 **평가 → 원인 분석 → 프롬프트 수정 → 재평가** 사이클을 직접 경험했다.
