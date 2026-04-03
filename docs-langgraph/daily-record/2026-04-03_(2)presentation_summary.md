# 주식 분석 AI Agent 개발 — 최종 발표 (요약)

---

## 1. 프로젝트 개요

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

| 계층 | 기술 |
|---|---|
| API 서버 | FastAPI + SSE 스트리밍 |
| 메인 에이전트 | LangChain `create_agent` (ReAct) |
| 서브 에이전트 | LangGraph `StateGraph` |
| LLM | OpenAI GPT-4o |
| 실시간 데이터 | yfinance |
| 공시 RAG | Elasticsearch (BM25 + kNN + Cohere rerank) |
| 평가 | Opik |

### 에이전트 도구(Tool) 목록

| Tool | 데이터 소스 | 설명 |
|---|---|---|
| `get_stock_price` | yfinance | 현재가 + 등락률 |
| `get_company_info` | yfinance | 시가총액·PER·업종 |
| `get_recent_news` | yfinance | 최근 뉴스 최대 3건 (키워드 필터링) |
| `get_stock_history` | Elasticsearch | OHLCV 히스토리컬 데이터 |
| `search_sec_filing` | Elasticsearch (RAG) | SEC 10-K 기반 정성 정보 — 서브 에이전트 |

### 에이전트 페르소나 ([`app/agents/prompts.py`](../../app/agents/prompts.py))

| 영역 | 핵심 규칙 |
|---|---|
| 역할·범위 | 주식·금융 외 질문 거절 / 미지원 기능(배당·전망·추천) 거절 + 가능 기능 안내 |
| 도구 사용 | 도구 조회 없이 수치 직접 생성 금지 |
| 응답 규칙 | 한국어 답변 / 묻지 않은 설명 금지 / search_sec_filing 결과에 회계연도 명시 |

> 응답 규칙은 Opik 평가 결과를 반영해 추가된 항목들 — hallucination·task_completion 점수 개선 목적.

### 아키텍처 흐름

![alt text](image.png)

---

## 2. LangGraph 구현 방식

### 메인 vs 서브 에이전트

| 항목 | 메인 에이전트 | 서브 에이전트 |
|---|---|---|
| 흐름 제어 | LLM이 동적으로 도구 선택 (ReAct) | 개발자가 노드·엣지를 정적 정의 |
| 병렬 실행 | 불가 | BM25 + kNN fan-out |
| 체크포인터 | `MemorySaver` (thread_id 기반 대화 이력) | 없음 — stateless |
| cross-invocation 상태 | 체크포인터로 자동 관리 | `_seen_ids_cache`로 직접 관리 |

### subagent-as-tool 패턴

StateGraph 전체를 `@tool`로 감싸 메인 에이전트에 일반 도구처럼 등록. `InjectedToolArg`로 `config`(thread_id 포함)를 LLM 스키마 노출 없이 주입받아 `_seen_ids_cache` 키로 사용.

### 메모리 구조

| | 메인 에이전트 | 서브 에이전트 |
|---|---|---|
| 체크포인터 | `MemorySaver` — thread_id별 전체 대화 이력 | 없음 — 매 호출 독립 실행 |
| 역할 | 멀티턴 맥락 유지 ("방금 물어본 종목") | 불필요 — 순수 검색 파이프라인 |
| cross-invocation 상태 | 체크포인터로 자동 관리 | `_seen_ids_cache` 딕셔너리로 직접 관리 |

서브 에이전트에 체크포인터를 붙이면 불필요한 상태가 쌓이고 thread_id 관리 복잡도가 높아짐 → 의도적으로 stateless 설계.

### 데이터 준비 — RAG 파이프라인 ([`scripts/ingest_10k.py`](../../scripts/ingest_10k.py))

**SEC EDGAR**: SEC(미국 증권거래위원회)가 운영하는 공시 문서 전자 수집 시스템. 미국 상장기업 10-K 의무 제출, API 키 없이 무료 다운로드 가능.

```
EDGAR 다운로드 → HTML 파싱 → 섹션 추출(Item1/1A/7)
  → tiktoken 청킹(512 tokens, overlap 50)
  → text-embedding-3-small 임베딩
  → ES bulk upsert (chunk_id로 멱등성 보장)
```

tiktoken 방식 선택 이유: LLM context window는 토큰 단위 제한 → 청크 크기를 토큰으로 직접 제어하는 것이 예측 가능.

### 검색 방식 — BM25 vs kNN

| | BM25 | kNN |
|---|---|---|
| 원리 | 키워드 빈도 기반 | 의미 유사도 기반 |
| 강점 | 고유명사 ("CUDA", "Blackwell") | 패러프레이징·동의어 |
| 약점 | 동의어에 취약 | 희귀 용어·신조어에 취약 |

```python
_TOP_K = 20            # BM25 / kNN 각각 최대 20개 검색
_NUM_CANDIDATES = 100  # kNN 후보 탐색 범위 (최소 _TOP_K 이상)
_FINAL_TOP_N = 5       # 리랭킹 후 LLM에 전달할 최종 청크 수
```

### 리랭킹

BM25와 kNN의 score는 척도가 달라 단순 비교 불가. 병합 후 Cohere Rerank API(cross-encoder 기반)로 질문과 각 청크 간 실제 관련도 재산출 → Top 5 추출. 미설정 시 score 내림차순 fallback.

### `_id` 중복 제거 vs `seen_ids` 제외

| | `_id` 중복 제거 (`merge_results`) | `seen_ids` 제외 (`rerank`) |
|---|---|---|
| 타이밍 | 단일 호출 내부 | 호출 간 (cross-invocation) |
| 목적 | BM25·kNN이 동일 청크를 동시 반환 시 1개만 남김 | 이전 호출에서 이미 반환한 청크 재등장 방지 |

---

## 3. Opik Evaluation

### 평가 레이어 구조

```
Layer 1 (run_eval_sec.py)  서브 에이전트 단독 — 검색 품질
Layer 2 (run_eval.py)      메인 에이전트 E2E  — 답변 품질
```

레이어 분리 이유: E2E 점수만으로는 낮은 점수가 검색 문제인지 LLM 문제인지 구분 불가.

### 메트릭 구성 (5종)

| 메트릭 | 레이어 | 기반 | 점수 방향 |
|---|---|---|---|
| `StockHallucination` | L2 메인 | `Hallucination` | **낮을수록 좋음** |
| `StockAnswerRelevance` | L1·L2 메인 | `AnswerRelevance` | 높을수록 좋음 |
| `StockTaskCompletion` | L2 메인 | `GEval` + 페이로드 패킹 | 높을수록 좋음 |
| `SecRetrievalRelevance` | L1·L2 서브 | `GEval` | 높을수록 좋음 |
| `SecGroundedness` | L2 메인 | `Hallucination` + context | **낮을수록 좋음** |

### SecGroundedness — Hallucination 오탐 해결

기존 `StockHallucination`은 context 없이 수치를 판단해 실시간 도구 결과도 환각으로 오판했다. `SecGroundedness`는 `search_sec_filing` tool output을 context로 직접 캡처·주입. context 없는 항목(실시간 도구)은 `None` 반환으로 채점 skip.

### 평가 결과

**서브 에이전트 단독 (L1 4건)**

| 메트릭 | 점수 |
|---|---|
| `sec_retrieval_relevance` | 0.45 |

**메인 에이전트 엔드투엔드**

| 메트릭 | L1 5건 | L2 21건 | 2주차 (참고) |
|---|---|---|---|
| `hallucination_metric` | 0.12 | **0.360** | 0.305 |
| `answer_relevance_metric` | 0.644 | **0.884** | 0.804 |
| `stock_task_completion` | — | **0.724** | — |
| `sec_groundedness` | — | skip (SEC 미호출 16건) | — |

L2 개선 원인: Cohere Rerank 적용(리랭킹 정상화) + SEC 데이터셋 6건 추가 + 프롬프트 개선 + 뉴스 키워드 필터링.

---

## 4. 주요 트러블슈팅

### yfinance 뉴스 무관 기사 혼입

**근본 원인**: yfinance `stock.news`는 티커를 직접 매핑하지 않고 **섹터·키워드 기반**으로 뉴스를 집계한다. TSLA 조회 시 Lucid(경쟁사)·Anthropic 기사가 혼입되는 것은 라이브러리 내부 문제로, 코드 레벨에서 완전히 제거할 수 없다.

**개선 과정**

| 단계 | 변경 내용 | 잔존 문제 |
|---|---|---|
| 초기 | 3건 조회 → LLM 필터 | 반환 건수 불안정 |
| 1차 | 30건 조회 → 도구 레벨 키워드 필터 | `"EV" in "seven"` 오탐 |
| 2차 | `\b` 단어 경계 regex 적용 | `_is_recent` 날짜 필터 항상 통과 |
| 3차 | `content.pubDate` 필드로 수정 | 아래 문제 참고 |

**`content.pubDate` 수정 후에도 남아있는 문제**

```python
def _is_recent(article: dict, cutoff: datetime) -> bool:
    pub_str = article.get("content", {}).get("pubDate")
    if not pub_str:
        return True  # ← pubDate 없으면 날짜 무관하게 통과
```

- `pubDate`가 없는 기사(구형 API 응답 구조, 일부 제공사)는 날짜 필터를 **무조건 통과**한다 — 30일이 넘은 오래된 기사도 혼입될 수 있다
- yfinance는 공식 API가 아니라 Yahoo Finance 비공식 스크래핑 기반이므로 **응답 구조가 라이브러리 버전마다 바뀔 수 있다** — `content.pubDate` 경로도 언제든 깨질 수 있고, 깨지면 또 전량 통과
- 키워드 필터 후 관련 기사가 0건이면 "없음" 반환 (fallback 없음) — 등록되지 않은 종목이나 키워드 범위 밖 기사는 아예 반환되지 않아 **사용자 입장에서 뉴스가 없는 것처럼 보일 수 있다**

---

## 5. 미해결 이슈

### Hallucination 메트릭이 실시간 데이터 에이전트에 근본적으로 부적합

`Hallucination` 메트릭은 원래 RAG 시스템에서 "제공된 context에 없는 내용을 지어냈는가"를 판단하도록 설계되어 있다. judge LLM 입장에서는 에이전트가 어떤 도구를 호출했는지, 그 결과가 무엇인지 알 수 없다. 때문에 도구로 정확하게 조회한 수치도 "근거 없는 환각"으로 오판한다.

| 질문 | judge 판단 | 실제 |
|---|---|---|
| NVDA 현재 주가 | "출처·타임스탬프 없어 검증 불가" → 높은 점수 | yfinance 실시간 조회 정확 데이터 |
| NVDA 7일 히스토리 | "2026년 3월 $180대는 말이 안 됨" → 높은 점수 | ES 저장 정확 데이터 |

SEC 공시 데이터는 `SecGroundedness`로 해결했지만(tool output을 context로 캡처), **yfinance 실시간 도구도 동일하게 context 캡처 구조를 적용해야** 근본 해결이 가능하다. 현재는 미구현 상태로, L2 hallucination 점수(0.360)에는 이 오탐이 포함되어 있어 실제 환각보다 수치가 높게 나타난다.

---

## 6. 배운 점

- **StateGraph vs `create_agent()`**: 고정 파이프라인은 StateGraph, 범용 추론은 ReAct — 두 방식을 적재적소에 조합하는 것이 핵심
- **subagent-as-tool**: 복잡한 파이프라인을 `@tool` 하나로 추상화 → 메인 에이전트와 독립적으로 테스트·평가 가능
- **평가 레이어 분리**: 검색 품질(서브)과 답변 품질(E2E)을 분리해야 개선 방향을 정확히 파악할 수 있다
- **메트릭의 한계를 데이터로 보완**: Hallucination 오탐을 context 캡처(SecGroundedness)로 해결 — 메트릭을 맹신하지 않고 설계로 보완하는 사고방식
- **평가 → 원인 분석 → 프롬프트 수정 → 재평가** 사이클을 서브 에이전트 레이어까지 확장
