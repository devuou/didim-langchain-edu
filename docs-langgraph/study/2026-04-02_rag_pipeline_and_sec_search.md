# RAG 파이프라인 & SEC 검색 서브 에이전트 학습 정리

> 학습 날짜: 2026-04-02 <br>
> 대상 파일: `scripts/ingest_10k.py`, `app/agents/sec_search_agent.py`, `app/agents/tools/_rag_common.py`

---

## 1. Context Window 초과란?

LLM은 한 번에 처리할 수 있는 **토큰 수의 상한선**이 있으며, 이를 **context window**라고 한다.

```
GPT-4o:     최대 128,000 토큰
Claude 3.5: 최대 200,000 토큰
```

10-K Item 1A (Risk Factors)는 원문이 수십만 자에 달해, 그대로 LLM에 전달하면 API가 오류를 반환한다.

**해결 방법: 청킹 + 검색**
```
원문 전체 (~300,000 토큰)
  → 512토큰 단위로 청킹 후 ES에 저장
  → 질문과 관련성 높은 상위 5개 청크만 꺼냄 (~2,500 토큰)
  → LLM에 전달
```

---

## 2. 임베딩(Embedding)이란?

텍스트를 숫자 벡터로 변환하는 것이다. 의미가 비슷한 문장은 비슷한 벡터를 갖는다.

```
"Apple의 주요 사업 리스크는..."      → [0.023, -0.481, 0.109, ...]  # 1536개 숫자
"AAPL faces supply chain risks..."  → [0.019, -0.476, 0.112, ...]  # 유사한 벡터
```

이 성질 덕분에 **키워드가 달라도 의미가 같으면 검색되는 kNN 벡터 검색**이 가능하다.
BM25(키워드 매칭)가 놓치는 동의어, 유사 표현을 벡터 검색이 보완한다.

### 현재 프로젝트에서 사용하는 임베딩 모델

**`text-embedding-3-small`** (OpenAI, 1536차원)

적재와 검색 두 곳에서 동일 모델을 사용한다.

| 위치 | 함수 | 역할 |
|---|---|---|
| `scripts/ingest_10k.py` | `embed_texts()` | 청크 텍스트 → 벡터, ES에 저장 |
| `app/agents/tools/_rag_common.py` | `embed_query()` | 사용자 질문 → 벡터, kNN 검색에 사용 |

**두 곳이 반드시 같은 모델이어야 한다.** 다른 모델을 쓰면 벡터 공간이 달라져 kNN 검색 결과가 엉터리가 된다.

### embed_texts()의 배치 처리

API를 청크마다 1회씩 호출하는 대신, 100개씩 묶어서 한 번에 요청한다.

```
240개 청크를 1개씩 → 240회 API 호출
240개 청크를 100개씩 → 3회 API 호출 (훨씬 빠름, 비용 절감)
```

### 지수 백오프(Exponential Backoff) 재시도

API에는 분당 요청 수 제한(Rate Limit)이 있어 초과 시 429 오류가 반환된다.
단순 즉시 재시도는 서버 부하를 가중시켜 역효과가 난다.

지수 백오프는 재시도마다 **대기 시간을 늘려** 서버 회복 시간을 준다.

```
1차 실패 → 5초 대기 → 재시도
2차 실패 → 10초 대기 → 재시도
3차 실패 → 15초 대기 → 재시도
4차 실패 → 포기 (예외 발생)
```

분산 시스템에서 표준적으로 사용하는 패턴이다.

---

## 3. 서브 에이전트 내 LLM 사용 여부

**현재 구현에서는 LLM을 사용하지 않는다.** 4개 노드 모두 데이터 처리만 수행한다.

```
bm25_search   → ES API 호출 (키워드 검색)
vector_search → OpenAI Embeddings API + ES API 호출 (의미 검색)
merge_results → Python dict 연산 (중복 제거)
rerank        → CrossEncoder 모델 (소형 ML 모델, LLM 아님)
```

LLM은 서브 에이전트가 반환한 텍스트를 **받아서 읽는** 메인 에이전트에서만 사용된다.

LLM을 서브 에이전트에 추가하려면 `summarize` 노드를 추가하고 `rerank → summarize → END`로 연결하면 된다. 다만 메인 에이전트 LLM이 이미 검색 결과를 읽고 답변하므로 중복이 될 수 있다.

---

## 4. BM25 match 쿼리

### text 필드와 역색인

ES에서 `{"type": "text"}` 필드는 저장 시 **역색인(inverted index)**을 자동 생성한다.

```
저장된 단어 → 어느 문서에 있는지 목록
"supply" → [문서3, 문서7, 문서12, ...]
"chain"  → [문서3, 문서9, ...]
```

`match` 쿼리는 이 역색인을 통해 검색어가 포함된 문서를 빠르게 찾는다.

### BM25 점수 계산

- **TF (Term Frequency)**: 문서 안에서 검색어가 자주 등장할수록 높은 점수
- **IDF (Inverse Document Frequency)**: 전체 문서 중 희귀한 단어일수록 높은 점수 ("the" 같은 흔한 단어는 낮은 점수)

### must vs filter

```python
"bool": {
    "must":   [{"match": {"text": query}}],   # BM25 점수 계산 포함
    "filter": [{"term": {"ticker": ticker}}]  # 점수와 무관한 필터링만 (더 빠름)
}
```

`filter`는 점수 계산을 하지 않으므로 캐싱이 가능해 `must`보다 빠르다.

---

## 5. ES Query DSL — 약속된 규격

`bm25_search()`와 `vector_search()`의 body 형식이 다른 건 자유 설계가 아니라 **ES Query DSL이라는 공식 규격**을 따르기 때문이다.

```python
# BM25: query 키 사용
body = {"size": 20, "query": {"bool": {"must": [...], "filter": [...]}}}

# kNN: knn 키 사용 (완전히 다른 검색 엔진)
body = {"knn": {"field": "embedding", "query_vector": [...], "k": 20, ...}}
```

두 구조가 다른 이유는 BM25와 kNN이 ES 내부에서 완전히 다른 검색 엔진을 사용하기 때문이다.

---

## 6. num_candidates vs k (kNN 검색)

```
도서관 비유:
  num_candidates=100: 사서가 "관련 있을 것 같은 책 100권"을 창고에서 꺼냄
  k=20:               그 100권 중 가장 유사한 20권만 반환
```

kNN은 정확한 최근접 이웃을 찾으려면 전체 벡터를 다 비교해야 하는데, 수십만 개를 모두 비교하면 너무 느리다. ES는 근사 알고리즘(HNSW)으로 "그럴듯한 후보 `num_candidates`개"를 먼저 추린 후, 그 안에서 정확하게 상위 `k`개를 고른다.

```
num_candidates 크게 → 정확도 높아짐, 속도 느려짐
num_candidates 작게 → 속도 빠름, 좋은 결과를 놓칠 수 있음
```

---

## 7. merge_results / _merge_results_fn 분리 이유

**테스트 편의성** 때문이다.

LangGraph 노드 함수는 `SecSearchState` TypedDict를 받는다. 테스트 시 TypedDict의 모든 필드를 채우는 것보다 일반 `dict`로 직접 호출하는 게 간단하다.

```python
# 핵심 로직: 일반 dict 받음 → 테스트에서 바로 호출 가능
def _merge_results_fn(state: dict) -> dict: ...

# LangGraph 노드: TypedDict 받음 → 그래프에 등록되는 인터페이스
def merge_results(state: SecSearchState) -> dict:
    return _merge_results_fn(state)

# 테스트
result = _merge_results_fn({"bm25_hits": [...], "vector_hits": [...]})
```

`rerank` / `_rerank_fn`도 같은 이유로 분리되어 있다.

---

## 8. 리랭킹(Reranking)이란?

BM25와 kNN을 병합하면 **점수 체계가 다른 문서들이 섞인다**.

```
BM25 결과:  문서A score=12.3  (TF-IDF 기반)
kNN 결과:   문서C score=0.91  (코사인 유사도 기반)
```

`12.3 > 0.91`이라고 문서A가 더 관련 있는 게 아니다. 단위가 다르다.

리랭커는 모든 후보를 **동일 기준으로 재점수화**한다.

```
리랭커 입력: [(질문, 문서A 텍스트), (질문, 문서C 텍스트), ...]
리랭커 출력: [0.94, 0.87, ...]  ← 모두 같은 기준의 관련도 점수
```

### 리랭커 종류 비교

| 도구 | 방식 | 특징 |
|---|---|---|
| **ES 호스팅 rerank (현재)** | ES Inference API | 별도 패키지 불필요, ES 서버에 모델 배포 필요 |
| CrossEncoder (로컬) | 로컬 ML 모델 | 무료, `sentence-transformers` 설치 필요 |
| Cohere Rerank API | 클라우드 API | 유료, 빠름, API 키 필요 |
| BGE-Reranker | 로컬 ML 모델 | 다국어 지원 강함 |

### ES 호스팅 리랭커 동작 방식

```
1. eland로 HuggingFace 모델을 ES에 업로드 (또는 Elastic 기본 제공 모델 사용)
2. ES에 inference endpoint 생성 → inference_id 부여 (예: .rerank-v1-elasticsearch)
3. ES Inference API 호출:
   POST /_inference/rerank/{inference_id}
   { "query": "...", "input": ["text1", "text2", ...] }
4. ES가 각 (query, text) 쌍의 관련도 점수를 계산하여 내림차순으로 반환
```

### 현재 프로젝트 구성

교육용 ES 서버(`GET _inference` 조회 결과)에 이미 배포된 inference endpoint:

| inference_id | task_type | model |
|---|---|---|
| `.rerank-v1-elasticsearch` | rerank | `.rerank-v1` (Elastic 자체 제공) |
| `.multilingual-e5-small-elasticsearch` | text_embedding | multilingual-e5-small |
| `.elser-2-elasticsearch` | sparse_embedding | ELSER v2 |

`.rerank-v1`은 HuggingFace cross-encoder가 아닌 Elastic이 자체 제공하는 rerank 모델이지만, 동일한 Inference API 인터페이스로 사용한다.

---

## 9. ES 호스팅 리랭커 — 코드 구조

`ES_RERANKER_INFERENCE_ID`를 `.env`에 설정하면 리랭킹이 활성화된다.

```python
# _rag_common.py
def rerank_hits(query: str, hits: list[dict]) -> list[dict] | None:
    inference_id = settings.ES_RERANKER_INFERENCE_ID
    if not inference_id:
        return None  # 미설정 → fallback

    resp = es.inference.inference(
        task_type="rerank",
        inference_id=inference_id,
        body={"query": query, "input": [h["_source"]["text"] for h in hits]},
    )
    ranked = resp.get("rerank", [])
    return [hits[item["index"]] for item in ranked]  # index = 원본 hits 배열 인덱스
```

```python
# sec_search_agent.py — rerank 노드
reranked = rerank_hits(query, hits)
if reranked is None:
    hits = sorted(hits, key=lambda h: h["_score"], reverse=True)  # fallback
else:
    hits = reranked
```
