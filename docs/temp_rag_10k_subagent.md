# [임시] 기업 공시 문서 기반 RAG 서브 에이전트

> 상태: 아이디어 검토 중 (미착수)

---

## 개요

현재 에이전트가 답하지 못하는 **기업 공시·사업보고서 기반 질문**을 처리하기 위한 RAG 파이프라인 및 서브 에이전트 설계.

```
"AAPL의 주요 사업 리스크가 뭐야?"
"MSFT가 공시에서 밝힌 AI 전략은?"
"TSLA 사업보고서에서 매출 비중이 가장 큰 부문은?"
```

---

## 데이터 소스

- **대상 문서**: SEC EDGAR 10-K (연간 사업보고서)
- **대상 종목**: AAPL, MSFT, TSLA, NVDA (기존 에이전트와 동일)
- **추출 섹션**:
  - Item 1. Business — 사업 개요
  - Item 1A. Risk Factors — 리스크 요인
  - Item 7. MD&A — 경영진의 사업 성과 분석

---

## 파이프라인 설계

### 1단계: 문서 파싱

```
SEC EDGAR PDF 다운로드 → 텍스트 추출 (pdfplumber 또는 pypdf)
→ 섹션 단위 분리 (Item 1, 1A, 7 헤더 기준)
```

### 2단계: 텍스트 청킹

- 전략: 섹션 단위 1차 분리 후 고정 크기(512 tokens) + 오버랩(50 tokens) 청킹
- 메타데이터 보존: `ticker`, `section`, `page_num`, `fiscal_year`

### 3단계: 임베딩

```python
# OpenAI text-embedding-3-small
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### 4단계: ES 적재

```
인덱스명: {ES_INDEX_PREFIX}-10k-docs
필드:
  - ticker: keyword
  - section: keyword
  - fiscal_year: keyword
  - text: text
  - embedding: dense_vector (dims=1536)
  - ingested_at: date
```

현재 OHLCV 인덱스(term+range 쿼리)와 달리 **kNN 벡터 검색** 사용 — 같은 ES에서 두 방식 비교 가능.

### 5단계: 검색 + 리랭킹

```
질문 → 임베딩 → kNN 검색 (상위 20개)
  → cross-encoder 리랭킹 (상위 3~5개 선별)
    → LLM 컨텍스트로 전달
```

리랭킹 라이브러리 후보: `sentence-transformers` cross-encoder 또는 Cohere Rerank API

---

## 에이전트 연동

### 신규 도구

```python
@tool
def search_sec_filing(ticker: str, query: str) -> str:
    """
    기업 공시(10-K)에서 질문과 관련된 내용을 검색합니다.
    사업 구조, 리스크 요인, 경영 성과 분석 등 정성적 정보 조회에 사용합니다.
    """
```

### 메인 에이전트 라우팅

```
사용자 질문
  ├─ 주가/재무 수치 → 기존 yfinance 도구
  ├─ 과거 주가 → 기존 ES OHLCV 도구
  └─ 공시·전략·리스크 → 신규 search_sec_filing 도구
```

---

## LangGraph 서브 에이전트 구조 (선택 적용)

단순 도구로 구현해도 되지만, LangGraph 학습 목적으로 서브 에이전트 그래프로 구성 시:

```
입력 노드 (질문 + ticker)
  → 쿼리 생성 노드 (LLM이 검색에 최적화된 쿼리로 변환)
  → kNN 검색 노드
  → 리랭킹 노드
  → 답변 생성 노드
```

- 학습 포인트: 고정 흐름 그래프(ReAct가 아닌 선형/조건 분기), 노드 간 state 설계

---

## 현재 프로젝트와의 차이점 (학습 관점)

| 항목 | 현재 (OHLCV) | 신규 (10-K RAG) |
|---|---|---|
| 데이터 형태 | 구조화된 수치 | 비구조화 장문 텍스트 |
| ES 검색 방식 | term + range 쿼리 | kNN 벡터 검색 |
| 임베딩 | 없음 | 필요 |
| 리랭킹 | 없음 | cross-encoder |
| 적재 시점 | 앱 시작 시 (yfinance → ES) | 사전 오프라인 파이프라인 |

---

## 예상 추가 패키지

```
uv add pdfplumber sentence-transformers langchain-openai
```
