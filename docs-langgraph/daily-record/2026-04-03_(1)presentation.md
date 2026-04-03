# 주식 분석 AI Agent 개발 — 최종 발표

---

## 1. 프로젝트 개요

### 무엇을 만들었나?

자연어로 주식 정보를 질문하면 실시간 데이터 또는 공시 문서를 조회해 답변하는 AI 에이전트.

```
사용자: "AAPL의 주요 사업 리스크가 뭐야?"
에이전트: search_sec_filing(AAPL, "사업 리스크") 호출
         → BM25 + kNN 병렬 검색 → 리랭킹 → 상위 5청크 추출
답변:
AAPL의 10-K(2024년 연간 보고서) 기준 주요 사업 리스크는 다음과 같습니다.
1. 글로벌 공급망 집중 리스크: 제조 파트너 및 부품 공급업체가 소수에 집중...
2. 중국 시장 의존도: 매출의 상당 비중이 중국 시장에서 발생...
```

### 기술 스택

| 계층 | 기술 | 비고 |
|---|---|---|
| API 서버 | FastAPI + SSE 스트리밍 | 단계별 실시간 이벤트 전송 |
| 메인 에이전트 | LangChain `create_agent` (ReAct) | 도구 선택 동적 결정 |
| 서브 에이전트 | LangGraph `StateGraph` | BM25+kNN 병렬 검색 파이프라인 |
| LLM | OpenAI GPT-4o | 질문 해석 + 최종 답변 |
| 실시간 데이터 | yfinance | 현재가·뉴스·재무정보 |
| 히스토리컬 데이터 | Elasticsearch | OHLCV 1년치 |
| 공시 RAG | Elasticsearch (BM25 + kNN + ES rerank) | SEC 10-K 4개 종목 |
| 평가 | Opik | LLM-as-a-Judge |

### 에이전트 도구(Tool) 목록

| Tool | 데이터 소스 | 설명 |
|---|---|---|
| `get_stock_price` | yfinance (실시간) | 현재가 + 전일 대비 등락률 |
| `get_company_info` | yfinance (실시간) | 시가총액·PER·업종 |
| `get_recent_news` | yfinance (실시간) | 최근 뉴스 최대 3건 (키워드 필터링 적용) |
| `get_stock_history` | Elasticsearch | OHLCV 히스토리컬 데이터 |
| `search_sec_filing` | Elasticsearch (RAG) | SEC 10-K 기반 정성 정보 — 서브 에이전트로 구현 |

- 복합 질문 처리: `"AAPL 주가랑 최근 뉴스 같이 알려줘"` → `get_stock_price` + `get_recent_news` 병렬 호출
- thread_id 기반 멀티턴 대화 (MemorySaver)
- 주식·금융 외 질문 거절

### 에이전트 페르소나 ([`app/agents/prompts.py`](../../app/agents/prompts.py))

시스템 프롬프트가 에이전트의 성격과 동작 범위를 결정한다. 크게 세 영역으로 구성된다.

**① 역할 및 범위 정의** — 에이전트가 답할 수 있는 것과 없는 것

```
주식/금융과 무관한 질문 → "저는 주식 분석 전용 AI입니다." 안내
지원하지 않는 기능(배당·전망·투자 추천) → 거절 + 제공 가능 기능 안내
```

**② 도구 사용 지침** — 언제 어떤 도구를 써야 하는가 명시

```
도구 조회 없이 수치를 직접 생성하는 것은 금지
get_stock_history  → 주가 추이, 특정 기간 최고가/최저가 분석
search_sec_filing  → 사업 구조, 리스크, MD&A 등 정성 정보
```

**③ 응답 규칙** — Opik 평가 결과를 반영해 추가된 규칙들

| 규칙 | 추가 배경 |
|---|---|
| 도구 조회 없이 수치 생성 금지 | hallucination 점수 높아서 추가 |
| 묻지 않은 개념 설명·추가 안내 문구 금지 | task_completion 점수 낮아서 추가 |
| 거절 시 제공 가능 기능 안내 | 거절만 했을 때 task_completion 기준 미충족 |
| 반드시 한국어로 답변 | 영어 뉴스 제목이 번역 없이 노출되는 문제 |
| search_sec_filing 결과에 회계연도 명시 | 공시 시점 불명확 문제 |

### 아키텍처 흐름

```
POST /api/v1/chat {thread_id, message}
│
▼
┌──────────────────────────────────────────────────┐
│  【메인 에이전트】 LangChain ReAct (create_agent)  │
│                                                  │
│  ┌─ model 스텝 ──────────────────────────────┐   │
│  │  ChatOpenAI(GPT-4o) + system_prompt       │   │
│  │  → 어떤 도구를 쓸지 결정                   │   │
│  └───────────────────────────────────────────┘   │
│            │                                     │
│            ▼                                     │
│  ┌─ tools 스텝 ──────────────────────────────────┐   │
│  │  get_stock_price    (yfinance)                │   │
│  │  get_company_info   (yfinance)                │   │
│  │  get_recent_news    (yfinance)                │   │
│  │  get_stock_history  (ES)                      │   │
│  │  search_sec_filing  ──▶ LangGraph StateGraph  │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
                         │
         search_sec_filing 호출 시
                         ▼
┌──────────────────────────────────────────────────┐
│  【서브 에이전트】 LangGraph StateGraph            │
│                                                  │
│       START                                      │
│       /    \      ← fan-out (병렬)               │
│      ▼      ▼                                    │
│  bm25_    vector_                                │
│  search   search (embed → kNN)                   │
│      \    /       ← fan-in                       │
│       ▼  ▼                                       │
│  merge_results (_id 중복 제거, 높은 score 채택)   │
│       │                                          │
│       ▼                                          │
│     rerank (seen_ids 제외 → Cohere Rerank API)   │
│  → 상위 5청크 추출 → result + seen_ids 반환       │
│       │                                          │
│      END                                         │
└──────────────────────────────────────────────────┘
                         │
             메인 에이전트 → 최종 답변 생성
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│  SSE 이벤트 스트림                                │
│  {"step":"model",  "tool_calls":["get_stock_price"]}
│  {"step":"tools",  "name":"get_stock_price", ...}│
│  {"step":"done",   "content":"..."}              │
└──────────────────────────────────────────────────┘
```

---

## 2. LangGraph 구현 방식

### 왜 서브 에이전트를 별도로 설계했나?

메인 에이전트는 LLM이 다음 도구를 동적으로 선택하는 ReAct 구조다. SEC 10-K 검색처럼 "BM25 검색 → kNN 검색 → 병합 → 리랭킹"의 **고정된 파이프라인**은 LLM에게 맡길 이유가 없다. 개발자가 흐름을 정적으로 정의하고, 그 전체를 단일 도구처럼 노출하는 것이 효율적이다.

| 항목 | 메인 에이전트 (`create_agent`) | 서브 에이전트 (`StateGraph`) |
|---|---|---|
| 흐름 제어 | LLM이 동적으로 도구 선택 (ReAct) | 개발자가 노드·엣지를 정적 정의 |
| 병렬 실행 | 불가 | BM25 + kNN fan-out |
| 적합한 용도 | 범용 추론, 다단계 질의응답 | 고정 파이프라인, 검색·처리 흐름 |
| 대화 이력 | MemorySaver (thread_id 기반) | `_seen_ids_cache` (중복 청크 제거용) |

### subagent-as-tool 패턴

핵심 아이디어: **StateGraph 전체를 `@tool` 함수로 감싸면 메인 에이전트 입장에서는 일반 도구와 동일하다.**

```python
_sec_search_graph = _build_graph()  # 모듈 로드 시 1회 컴파일

@tool
def search_sec_filing(
    ticker: str,
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,  # LLM에 노출 안 됨
) -> str:
    """SEC 10-K 공시 기반 정성 정보 검색 (지원: AAPL, MSFT, TSLA, NVDA)"""
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    cache_key = f"{thread_id}:{ticker.upper()}" if thread_id else None
    seen_ids = _seen_ids_cache.get(cache_key, []) if cache_key else []

    result = _sec_search_graph.invoke({
        "ticker": ticker.upper(),
        "query": query,
        "seen_ids": seen_ids,
    })
    if cache_key:
        _seen_ids_cache[cache_key] = result.get("seen_ids", seen_ids)
    return result["result"]
```

- `InjectedToolArg`: `config`는 LangChain이 자동 주입하며 LLM 스키마에는 노출되지 않음
- `_seen_ids_cache`: `thread_id:ticker` 키로 이미 반환한 청크 ID를 보관 → "추가로 더 없어?" 요청 시 중복 청크 방지

### 데이터 준비 — RAG 파이프라인 ([`scripts/ingest_10k.py`](../../scripts/ingest_10k.py))

서브 에이전트가 검색하는 데이터는 사전에 오프라인 파이프라인으로 ES에 적재된다. 서버 실행과 분리된 1회성 스크립트(`uv run python scripts/ingest_10k.py`)로 동작한다.

```
SEC EDGAR 다운로드 (sec-edgar-downloader)
  → primary-document.html
  → BeautifulSoup.get_text() — HTML 태그 제거, 순수 텍스트 추출
  → extract_sections() — Item 1 / Item 1A / Item 7 섹션별 분리
  → chunk_text() — tiktoken 512 tokens, overlap 50
  → embed_texts() — text-embedding-3-small (batch 100)
  → ES bulk upsert — _id = chunk_id (멱등성 보장)
```

**왜 이 세 섹션인가?**

| 섹션 | 내용 | 검색 활용 |
|---|---|---|
| Item 1 (Business) | 사업 구조, 제품·서비스, 경쟁 환경 | "AAPL 주요 사업은?" |
| Item 1A (Risk Factors) | 투자·사업 리스크 요인 | "TSLA 리스크가 뭐야?" |
| Item 7 (MD&A) | 경영진의 재무 성과 분석 | "NVDA 매출 성장 배경은?" |

**청킹: 문자 기반 vs 토큰 기반**

| | 문자 기반 (`RecursiveCharacterTextSplitter`) | 토큰 기반 (tiktoken, 이번 구현) |
|---|---|---|
| 단위 | 문자 수 (500자 등) | 토큰 수 (512 tokens) |
| LLM 정합성 | 간접적 (문자 수 ≠ 토큰 수) | 직접적 (context window = 토큰 단위) |
| 문장 보존 | 문단·문장 경계 우선 | 토큰 경계에서 분할 (문장 중간 잘림 가능) |

tiktoken 방식을 선택한 이유: LLM context window는 토큰 단위로 제한되므로 청크 크기를 토큰으로 직접 제어하는 것이 예측 가능하다. 최종적으로 `_FINAL_TOP_N = 5`청크만 LLM에 전달되므로 청크 경계 품질의 영향이 제한적이기도 하다.

**chunk_id 구조와 멱등성**

```python
# chunk_id = "{ticker}_{fiscal_year}_{section}_{index}"
# 예: "AAPL_2024_item1a_003"
# _id로 ES에 upsert → 동일 ID 재실행 시 덮어쓰기 (중복 적재 없음)
```

### 검색 방식 — BM25 vs kNN

두 방식을 병렬로 실행하고 결과를 합치는 이유: 각각 잘 잡는 질문 유형이 다르다.

| | BM25 | kNN 벡터 검색 |
|---|---|---|
| 원리 | 단어 빈도 기반 키워드 매칭 (TF-IDF 개선) | 의미 유사도 기반 최근접 이웃 탐색 |
| 강점 | 정확한 용어 포함 문서 ("CUDA", "Blackwell" 등 고유명사) | 의미적으로 관련된 문서 ("GPU 경쟁력" → "computing platform" 청크) |
| 약점 | 동의어·패러프레이징에 취약 | 희귀 용어·신조어에 취약 |
| ES 구현 | `match` 쿼리 + `term` filter | `knn` 쿼리 + `text-embedding-3-small` 1536차원 벡터 |

```python
# 상수 (sec_search_agent.py)
_TOP_K = 20            # BM25 / kNN 각각 최대 20개 검색
_NUM_CANDIDATES = 100  # kNN 후보 탐색 범위 (클수록 정확하지만 느림, 최소 _TOP_K 이상)
_FINAL_TOP_N = 5       # 리랭킹 후 LLM에 전달할 최종 청크 수

# bm25_search 노드 — ES match 쿼리
body = {
    "size": _TOP_K,  # 20개
    "query": {
        "bool": {
            "must":   [{"match": {"text": state["query"]}}],   # BM25 점수 계산
            "filter": [{"term": {"ticker": state["ticker"]}}], # 종목 필터 (점수 무영향)
        }
    },
}

# vector_search 노드 — OpenAI 임베딩 → ES kNN
query_vector = embed_query(state["query"])  # text-embedding-3-small, 1536차원
body = {
    "knn": {
        "field": "embedding",
        "query_vector": query_vector,
        "k": _TOP_K,                  # 20개 — 반환할 최근접 이웃 수
        "num_candidates": _NUM_CANDIDATES,  # 100개 — 탐색 후보 범위
        "filter": [{"term": {"ticker": state["ticker"]}}],
    }
}
```

### 리랭킹 방식

BM25와 kNN의 score는 **서로 다른 척도**다. 단순 합산이나 score 비교로 최종 순위를 정할 수 없다. 두 결과를 병합한 후 Cohere Rerank API(cross-encoder 기반)로 질문과의 실제 관련도를 재평가한다.

```
BM25 결과 (20개)  ┐
                  ├─▶ merge (중복 제거, 높은 score 채택) ─▶ rerank ─▶ Top 5
kNN  결과 (20개)  ┘                                        ↑
                                              Cohere Rerank API (rerank-v3.5)
                                              질문과 각 청크 텍스트 간 관련도 점수 재산출
                                              미설정 시 score 내림차순 정렬로 fallback
```

```python
# rerank 노드 (_rag_common.py:rerank_hits)
co = cohere.ClientV2(settings.COHERE_API_KEY)
resp = co.rerank(
    model="rerank-v3.5",
    query=query,
    documents=texts,  # texts: 병합된 청크 텍스트 목록
)
# resp.results: [RerankResponseResultsItem(index=2, relevance_score=0.94), ...]
return [hits[item.index] for item in resp.results]
```

### BM25 + kNN 병렬 fan-out 파이프라인

```python
builder = StateGraph(SecSearchState)
builder.add_node("bm25_search",    _bm25_search_fn)
builder.add_node("vector_search",  _vector_search_fn)
builder.add_node("merge_results",  _merge_fn)
builder.add_node("rerank",         _rerank_fn)

builder.add_edge(START, "bm25_search")    # fan-out: 두 노드 동시 실행
builder.add_edge(START, "vector_search")
builder.add_edge("bm25_search",   "merge_results")  # fan-in: 둘 다 완료 후 진입
builder.add_edge("vector_search", "merge_results")
builder.add_edge("merge_results", "rerank")
builder.add_edge("rerank",        END)
```

- `START`에서 두 노드로 엣지를 추가하면 LangGraph가 자동으로 병렬 실행
- `merge_results`: `_id` 기준 중복 제거, 동일 문서는 높은 score 채택
- `rerank`: `seen_ids`에 있는 청크 제외 후 Cohere Rerank API 리랭킹 (미설정 시 score 내림차순 fallback)

### 메모리 구조

LangGraph에서 상태를 대화 간 유지하려면 **체크포인터(checkpointer)** 를 컴파일 시 주입해야 한다. 메인·서브 에이전트는 역할이 다르므로 메모리 전략도 다르게 설계했다.

| | 메인 에이전트 | 서브 에이전트 |
|---|---|---|
| 체크포인터 | `MemorySaver` ([`agent_service.py:66`](../../app/services/agent_service.py)) | 없음 — `_sec_search_graph.invoke()` 직접 호출 |
| 저장 단위 | thread_id별 전체 대화 이력 | 저장하지 않음 (매 호출 독립 실행) |
| 역할 | "방금 물어본 종목", "다시 알려줘" 같은 멀티턴 맥락 유지 | 순수 검색 파이프라인 — 대화 맥락 불필요 |
| cross-invocation 상태 | 체크포인터로 자동 관리 | `_seen_ids_cache` (메모리 딕셔너리)로 직접 관리 |

```
메인 에이전트 (MemorySaver)
  "방금 물어본 종목" → thread 이력에서 AAPL 추출 → search_sec_filing(AAPL, query)
                                                              ↓
                                              서브 에이전트 (stateless)
                                              매 호출마다 새 SecSearchState 생성
                                              seen_ids만 _seen_ids_cache에서 읽어 중복 제거
```

멀티턴 맥락은 메인 에이전트가, 검색 중복 제거는 서브 에이전트 외부 캐시가 각각 담당한다. 서브 에이전트에 체크포인터를 붙이면 thread_id 관리 복잡도가 높아지고, 단순 검색 파이프라인에 불필요한 상태가 쌓인다.

---

## 3. Opik Evaluation

### 평가 레이어 분리 — 왜?

2주차에는 메인 에이전트만 평가했다. SEC 서브 에이전트가 추가되면서 두 가지 문제가 생겼다.

1. **검색 품질 불투명**: 메인 에이전트 점수가 낮아도 원인이 LLM 답변인지 검색 결과인지 구분 불가
2. **Hallucination 오탐**: judge LLM이 실시간 수치의 출처를 알 수 없어 정확한 데이터도 환각으로 판정

→ 레이어를 분리해 각 계층의 품질을 독립적으로 측정하는 구조로 확장했다.

```
Layer 1: 서브 에이전트 단독 평가 (run_eval_sec.py)
  → sec_dataset.json (10건)
  → _sec_search_graph.invoke() 직접 호출
  → SecRetrievalRelevance: 검색된 청크가 질문과 관련 있는가?

Layer 2: 메인 에이전트 엔드투엔드 평가 (run_eval.py)
  → dataset.json (21건, SEC 질문 6건 추가)
  → 메인 에이전트 astream() 호출
  → SecGroundedness: search_sec_filing 결과를 context로 캡처 → 근거 충실도 판단
```

### 메트릭 구성 (전체 5종)

| 메트릭 | 평가 레이어 | 판단 기준 | 점수 방향 | 기반 |
|---|---|---|---|---|
| `StockHallucination` | L2 (메인) | 조회하지 않은 수치를 지어냈는지 | **낮을수록 좋음** | `Hallucination` |
| `StockAnswerRelevance` | L1·L2 (메인) | 질문과 답변의 관련성 | 높을수록 좋음 | `AnswerRelevance` |
| `StockTaskCompletion` | L2 (메인) | expected_output 기준 작업 완수 | 높을수록 좋음 | `GEval` |
| **`SecRetrievalRelevance`** | L1·L2 (서브) | 검색 청크가 질문과 관련 있는가 | 높을수록 좋음 | `GEval` |
| **`SecGroundedness`** | L2 (메인) | 답변이 검색 결과에 근거하는가 | **낮을수록 좋음** | `Hallucination` + context |

### SecGroundedness — Hallucination 오탐 해결

기존 `StockHallucination`은 context 없이 수치를 판단해 실시간 데이터도 환각으로 오판했다. `SecGroundedness`는 tool output을 직접 context로 주입해 이 문제를 해결했다.

```python
# run_eval.py — astream 루프에서 search_sec_filing tool output 수집
if step == "tools":
    for msg in event.get("messages", []):
        if getattr(msg, "name", None) == "search_sec_filing":
            tool_outputs.append(msg.content)

# evaluation_task 반환값에 context 추가
return {"input": q, "output": output, "context": tool_outputs, ...}
```

```python
# sec_groundedness.py
def score(self, output: str, **kwargs) -> ScoreResult | None:
    context = kwargs.get("context", [])
    if not context:
        return None   # search_sec_filing 미호출 항목 → 채점 skip
    return self._inner.score(input=..., output=output, context=context)
```

`context=None` 반환으로 Opik이 해당 항목을 채점에서 제외 → 실시간 데이터 항목에 대한 오탐 방지.

### 평가 결과

**서브 에이전트 단독 (run_eval_sec.py, L1 4건)**

| 메트릭 | 점수 |
|---|---|
| `sec_retrieval_relevance` | 0.45 |

**메인 에이전트 엔드투엔드 (run_eval.py)**

| 메트릭 | L1 5건 | **L2 21건** | 2주차 결과 (참고) |
|---|---|---|---|
| `hallucination_metric` | 0.12 | **0.360** | 0.305 |
| `answer_relevance_metric` | 0.644 | **0.884** | 0.804 |
| `stock_task_completion` | — | **0.724** | — |
| `sec_groundedness` | — | skip (SEC 미호출 항목 16건) | — |

L2 개선 원인: Cohere Rerank API 적용(리랭킹 정상화) + SEC 데이터셋 6건 추가 + 프롬프트 개선("도구 조회 없이 수치 직접 생성 금지") + 뉴스 키워드 필터링.

---

## 4. 주요 트러블슈팅

### 1. SEC 10-K 섹션 추출이 목차(TOC)에서 조기 종료

**증상**: `item1` 1자, `item7` 88자 — 실제 본문이 아닌 목차 항목만 추출됨.

**원인**: SEC 10-K 문서는 앞부분에 목차가 있고 뒷부분에 본문이 있다. `re.search()`가 목차의 첫 번째 매칭에서 멈춰버렸다.

**해결**: `re.findall()`로 모든 매칭을 찾고 **마지막 매칭**(= 실제 본문)을 사용.

```python
matches = list(re.finditer(start_pat, text, re.IGNORECASE))
m_start = matches[-1]  # 마지막 = 실제 본문
```

→ item7: 18,099자 / 9개 청크로 정상 추출.

---

### 2. 섹션 stop 패턴이 교차 참조 텍스트에 매칭

**증상**: `item7` 추출 결과 227자 — 본문 첫 문장에서 잘림.

**원인**: 본문 첫 단락에 `"Part II, Item 8 of this Form 10-K"` 교차 참조 포함 → stop 패턴 `r"Item\s+(?:7A|8)\b"`이 실제 섹션 헤더가 아닌 교차 참조에 매칭.

**해결**: 마침표(`.`) 추가 — 실제 헤더는 `Item 8.` 형식이고 교차 참조는 마침표 없이 `Item 8`으로만 쓰임.

```python
_SECTION_CONFIG = {
    "item1":  (r"Item\s+1\.\s+Business",        r"Item\s+1A\."),
    "item1a": (r"Item\s+1A\.\s+Risk\s+Factors", r"Item\s+(?:1B|2)\."),
    "item7":  (r"Item\s+7\.\s+",                r"Item\s+(?:7A|8)\."),
}
```

---

### 3. GEval이 `expected_output`을 무시해 0점 반환

**증상**: `expected_output`을 전달했음에도 GEval이 0점 반환.

**원인**: `GEval.score(self, output: str, **ignored_kwargs)` — `input`, `expected_output`은 내부적으로 버려진다.

```python
def score(self, output: str, **ignored_kwargs): ...  # 추가 인자 무시
```

**해결**: 페이로드 패킹 — 세 필드를 하나의 문자열로 묶어 `output`에 전달.

```python
payload = f"QUESTION: {input}\nEXPECTED_OUTPUT: {expected_output}\nOUTPUT: {output}"
self._inner.score(output=payload)
```

→ 0점 항목 0건, task_completion 0.55 → 0.61.

---

### 4. JSON 인젝션 버그 (SSE 스트림 파괴)

**증상**: tool 결과에 `"` 또는 `}` 포함 시 SSE JSON이 깨짐.

**원인**: `message.content`를 f-string에 직접 삽입.

```python
# 수정 전 — JSON 이스케이프 없음
yield f'{{"step": "tools", "content": {message.content}}}'

# 수정 후
yield f'{{"step": "tools", "content": {json.dumps(message.content, ensure_ascii=False)}}}'
```

---

### 5. yfinance 뉴스 오탐 — 단어 경계 미적용

**증상**: `"EV" in "seven"` → True, `"EV" in "ever"` → True 오탐 발생.

**해결**: `re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)` — 단어 경계(`\b`) 기반 매칭으로 전환.

---

## 5. 미해결 이슈

### `get_stock_history` 30일 데이터 OpenAI 400 오류

30행 OHLCV 데이터가 LangChain 메시지 히스토리에 들어간 후 다음 OpenAI 호출 시 `"We could not parse the JSON body"` 400 오류 발생. 5일 이하는 정상 동작. 데이터 크기 또는 특수문자 인코딩 문제로 추정.

**개선 방향**: 메시지 히스토리에 삽입 전 도구 결과 크기 제한 또는 압축 처리.

---

### Hallucination 메트릭이 실시간 데이터 에이전트에 근본적으로 부적합

`Hallucination`은 RAG 시스템에서 "제공된 context에 없는 내용을 지어냈는가"를 판단하도록 설계되어 있다. 실시간 도구 호출 에이전트에서는 judge LLM이 수치의 출처를 알 수 없어 정확한 데이터도 환각으로 오판한다.

| 질문 | judge 판단 | 실제 |
|---|---|---|
| NVDA 현재 주가 | "출처·타임스탬프 없어 검증 불가" → 0.65 | yfinance 실시간 데이터 |
| NVDA 7일 히스토리 | "2026년 3월 $180대는 말이 안 됨" → 0.85 | ES 저장 정확 데이터 |

SEC 공시 데이터는 `SecGroundedness`로 해결했으나, 실시간 도구(yfinance)는 tool output을 context로 캡처하는 구조 개선이 추가로 필요하다.

---

### ES rerank AuthorizationException (403) → Cohere Rerank API로 해결

ES Inference API 리랭킹 호출 시 403 오류. 교육용 클러스터의 Basic 라이선스에서는 Inference API가 제한됨.

**해결**: Cohere Rerank API(`rerank-v3.5`)로 교체. `COHERE_API_KEY` 환경변수 설정으로 활성화되며, 미설정 시 기존과 동일하게 score 내림차순 정렬로 fallback.

---

## 6. 배운 점

### LangGraph StateGraph 직접 설계 vs `create_agent()`

`create_agent()`는 LLM이 흐름을 결정한다. LLM이 "어떤 도구를 쓸지"를 잘 결정하지만, 병렬 실행이나 고정 순서 파이프라인에는 적합하지 않다. StateGraph로 직접 설계하면 개발자가 노드·엣지를 정의하므로 흐름이 예측 가능하고 병렬 실행이 가능하다. **두 방식을 적재적소에 조합하는 것**이 핵심이다.

### subagent-as-tool — 복잡한 파이프라인의 단일 도구 추상화

BM25 + kNN + 리랭킹의 복잡한 파이프라인 전체를 `@tool` 하나로 감쌌다. 메인 에이전트는 내부 구조를 모른다. 이 패턴은 각 계층의 책임을 분리하고, 서브 에이전트를 독립적으로 테스트·평가할 수 있게 해준다.

### 평가 레이어 분리 — 검색 품질 vs 엔드투엔드 품질

에이전트가 복잡해질수록 단일 E2E 점수만으로는 어느 계층에서 문제가 발생했는지 알 수 없다. "서브 에이전트가 엉뚱한 청크를 반환한 것인가, 아니면 LLM이 정확한 청크를 받아도 잘못 요약한 것인가"를 구분하려면 레이어별 평가가 필요하다.

### context 캡처로 Hallucination 오탐 해결 — SecGroundedness 설계

`Hallucination` 메트릭의 한계를 파악하고, tool output을 context로 직접 주입하는 방식으로 우회했다. `context=None` 반환으로 비해당 항목을 채점에서 제외하는 설계도 중요한 포인트였다. **메트릭 자체의 한계를 이해하고 데이터 파이프라인으로 보완하는 사고방식**을 익혔다.

### 평가 → 원인 분석 → 프롬프트 수정 → 재평가 사이클

2주차 발표에서 정리한 핵심: 점수만 보는 것이 아니라 낮은 점수의 원인을 추적하고 수정하는 반복 개선이 실질적인 품질 향상을 만든다. 이번에는 그 사이클을 서브 에이전트 레이어까지 확장했다.
