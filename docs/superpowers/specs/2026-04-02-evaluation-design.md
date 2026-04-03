# 평가 시스템 설계 — 메인 에이전트 + SEC 서브 에이전트

날짜: 2026-04-02
대상: 주식 분석 AI 에이전트 (LangChain ReAct + LangGraph StateGraph)

---

## 배경

기존 평가(`run_eval.py`)는 메인 에이전트만 있을 때 설계되었다.

- 데이터셋 15건 (실시간 주가·기업정보·뉴스·히스토리컬·범위 밖)
- 메트릭: hallucination, answer_relevance, task_completion
- 한계: Hallucination 메트릭이 실시간 데이터에 부적합 (context 없어 judge가 수치 출처 불명으로 오판)

LangGraph 기반 SEC 공시 검색 서브 에이전트(`search_sec_filing`) 추가로 평가를 두 레이어로 확장한다.

---

## 목표

1. **Layer 1 (서브 에이전트 단독)**: 검색 파이프라인이 질문에 관련된 10-K 청크를 잘 찾아오는지 측정
2. **Layer 2 (엔드투엔드)**: 메인 에이전트가 SEC 질문에 올바른 도구를 선택하고 10-K 근거에 충실한 답변을 내는지 측정
3. 기존 평가를 그대로 유지하면서 SEC 평가를 추가 (기존 회귀 방지)

---

## 파일 구조

```
evaluation/
├── data/
│   ├── dataset.json          # 기존 15건 → SEC 질문 6건 추가 = 21건
│   └── sec_dataset.json      # 서브 에이전트 전용 신규 10건
├── metrics/
│   ├── stock_hallucination.py     # 기존 유지
│   ├── stock_answer_relevance.py  # 기존 유지
│   ├── stock_task_completion.py   # 기존 유지
│   ├── sec_retrieval_relevance.py # 신규: GEval 검색 관련성 (서브 에이전트용)
│   └── sec_groundedness.py        # 신규: Hallucination + context (메인 에이전트용)
├── run_eval.py               # 수정: SEC 질문 추가, context 캡처, groundedness 레벨 편입
└── run_eval_sec.py           # 신규: 서브 에이전트 단독 평가
```

---

## Layer 1: 서브 에이전트 단독 평가 (`run_eval_sec.py`)

### 데이터셋 (`sec_dataset.json`, 10건)

| 섹션 | 종목 | 질문 | expected_output |
|---|---|---|---|
| Item 1 (사업구조) | AAPL | 애플의 주요 제품과 서비스 부문은? | iPhone, Mac, Services 등 주요 부문 언급된 내용 |
| Item 1 (사업구조) | NVDA | NVDA의 핵심 사업 영역은? | 데이터센터, 게이밍 GPU 등 포함된 내용 |
| Item 1A (리스크) | MSFT | 마이크로소프트가 공시한 주요 리스크는? | 경쟁, 규제, 보안 등 리스크 키워드 포함 내용 |
| Item 1A (리스크) | TSLA | 테슬라의 공급망 리스크는? | 공급망 관련 리스크 내용 |
| Item 7 (MD&A) | AAPL | 애플 최근 사업연도 매출 성과는? | 매출·수익 관련 경영진 분석 내용 |
| Item 7 (MD&A) | NVDA | NVDA의 AI 사업 성과는? | AI·데이터센터 매출 성장 관련 내용 |
| 지원 종목 외 | AMZN | 아마존 리스크 알려줘 | "관련 공시 정보를 찾을 수 없습니다" |
| 지원 종목 외 | GOOGL | 구글 사업구조 알려줘 | "관련 공시 정보를 찾을 수 없습니다" |
| 복합 쿼리 | MSFT | 마이크로소프트 클라우드 전략과 리스크는? | Azure 관련 내용 + 리스크 모두 포함 |
| 복합 쿼리 | TSLA | 테슬라 전기차 시장 전략과 경쟁 리스크는? | 시장전략 + 경쟁 리스크 모두 포함 |

데이터셋 항목 스키마:
```json
{
  "ticker": "AAPL",
  "query": "애플의 주요 제품과 서비스 부문은?",
  "section": "item1",
  "expected_output": "iPhone, Mac, Services 등 주요 부문이 언급된 검색 결과"
}
```

### task 함수

```python
def sec_eval_task(dataset_item: dict) -> dict:
    result = _sec_search_graph.invoke({
        "ticker": dataset_item["ticker"],
        "query": dataset_item["query"],
    })
    return {
        "input": dataset_item["query"],
        "output": result["result"],
        "expected_output": dataset_item.get("expected_output", ""),
    }
```

### 메트릭: `SecRetrievalRelevance` (신규)

- 기반: `GEval` + 페이로드 패킹 (기존 `StockTaskCompletion`과 동일 방식)
- 판단 기준: "반환된 10-K 내용이 질문과 관련 있는가?"
- 점수: 0.0~1.0 (높을수록 좋음)
- 빈 결과("관련 공시 정보를 찾을 수 없습니다") 케이스는 expected_output과 비교해 정상 처리

```python
payload = f"QUESTION: {input}\nEXPECTED: {expected_output}\nRETRIEVED: {output}"
self._inner.score(output=payload)
```

### 레벨 구성

| 레벨 | 건수 | 메트릭 | 용도 |
|---|---|---|---|
| L1 | 4건 (섹션별 1건) | retrieval_relevance | 개발 중 빠른 검증 |
| L2 | 10건 (전체) | retrieval_relevance | 배포 전 전체 검증 |

---

## Layer 2: 메인 에이전트 엔드투엔드 평가 (`run_eval.py` 수정)

### 데이터셋 추가 (`dataset.json`, +6건 → 총 21건)

| 유형 | 건수 | 예시 | expected_output |
|---|---|---|---|
| SEC 단독 질문 | 3 | "AAPL 10-K에서 밝힌 주요 사업 리스크는?" | 리스크 요인 내용 포함 답변 |
| 실시간 + SEC 복합 | 2 | "NVDA 현재 주가랑 공시에서 밝힌 AI 전략 알려줘" | 주가 수치 + 10-K 내용 모두 포함 |
| SEC 범위 밖 | 1 | "삼성전자 10-K 알려줘" | 지원하지 않는 종목임을 안내하는 답변 |

### context 캡처 (`run_eval.py` 수정)

`evaluation_task` 내부의 `_stream()` 함수에서 `search_sec_filing` tool output을 수집한다.

```python
async def _stream():
    final = ""
    tool_outputs = []
    async for chunk in _agent.astream(...):
        for step, event in chunk.items():
            if step == "tools":
                messages = event.get("messages", [])
                for msg in messages:
                    if getattr(msg, "name", None) == "search_sec_filing":
                        tool_outputs.append(msg.content)
            if step == "model":
                # 기존 ChatResponse 감지 로직 유지
                ...
    return final, tool_outputs

output, tool_outputs = asyncio.run(_stream())
return {
    "input": question,
    "output": output,
    "expected_output": dataset_item.get("expected_output", ""),
    "context": tool_outputs,   # SecGroundedness가 이 값을 사용
}
```

### 메트릭: `SecGroundedness` (신규)

- 기반: `Hallucination(context=...)`
- context가 있을 때(search_sec_filing 호출됨): "답변이 검색된 10-K 청크에 근거하는가?" 판단
- context가 없을 때: `None` 반환 → Opik이 해당 항목 skip 처리
- 점수: 0.0=근거 있음(좋음), 1.0=근거 없음(나쁨) — 기존 hallucination과 동일 방향

```python
def score(self, output: str, **kwargs) -> ScoreResult | None:
    context = kwargs.get("context", [])
    if not context:
        return None  # SEC 도구 미호출 항목은 skip
    input_ = kwargs.get("input", "")
    return self._inner.score(input=input_, output=output, context=context)
```

### 레벨 구성

| 레벨 | 건수 | 메트릭 | 용도 |
|---|---|---|---|
| L1 | 5건 | hallucination, relevance | 빠른 검증 |
| L2 | 21건 | hallucination, relevance, task_completion, **groundedness** | 배포 전 전체 검증 |

---

## 메트릭 전체 요약

| 메트릭 | 파일 | 적용 위치 | 점수 방향 | 비고 |
|---|---|---|---|---|
| StockHallucination | 기존 | run_eval.py | 낮을수록 좋음 | 실시간 데이터 항목에는 여전히 한계 있음 |
| StockAnswerRelevance | 기존 | run_eval.py | 높을수록 좋음 | |
| StockTaskCompletion | 기존 | run_eval.py | 높을수록 좋음 | GEval + 페이로드 패킹 |
| SecGroundedness | **신규** | run_eval.py | 낮을수록 좋음 | Hallucination + context, SEC 항목만 채점 |
| SecRetrievalRelevance | **신규** | run_eval_sec.py | 높을수록 좋음 | GEval + 페이로드 패킹 |

---

## 실행 명령

```bash
# 메인 에이전트 평가
uv run python evaluation/run_eval.py --level L1
uv run python evaluation/run_eval.py --level L2

# 서브 에이전트 단독 평가
uv run python evaluation/run_eval_sec.py --level L1
uv run python evaluation/run_eval_sec.py --level L2
```

---

## 구현 순서

1. `sec_dataset.json` 작성
2. `sec_retrieval_relevance.py` 작성
3. `run_eval_sec.py` 작성
4. `dataset.json` SEC 질문 6건 추가
5. `sec_groundedness.py` 작성
6. `run_eval.py` 수정 (context 캡처 + groundedness 레벨 편입)
