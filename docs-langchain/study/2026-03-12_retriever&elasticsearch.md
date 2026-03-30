# Retriever & ElasticsearchRetriever 학습 정리

**참고 문서**: `docs/ref/retriever_integration.md`, `docs/ref/retriever_elasticsearch.md`

---

## 1. Retriever란?

**비구조화 쿼리(자연어)를 받아 관련 문서 목록을 반환하는 인터페이스**입니다.

```
입력: str (자연어 질문)
출력: list[Document]
```

### Vector Store와의 차이

| | Vector Store | Retriever |
|---|---|---|
| 역할 | 문서 저장 + 검색 | 검색만 |
| 범위 | 벡터 DB에 한정 | Wikipedia, Kendra 등 외부 소스도 포함 |
| 관계 | Vector Store → Retriever로 변환 가능 | 더 넓은 개념 |

> Vector Store는 모두 Retriever로 변환 가능 (`.as_retriever()`)

---

## 2. Retriever 종류

### 직접 구축형 (내 데이터 인덱싱 후 검색)

| Retriever | 특징 |
|---|---|
| `ElasticsearchRetriever` | 셀프호스팅 + 클라우드 모두 지원. Query DSL 완전 활용 가능 |
| `AzureAISearchRetriever` | Azure 클라우드 전용 |
| `AmazonKnowledgeBasesRetriever` | AWS Bedrock 기반 |

### 외부 인덱스형 (인터넷/외부 데이터 검색)

| Retriever | 데이터 소스 |
|---|---|
| `WikipediaRetriever` | 위키피디아 문서 |
| `ArxivRetriever` | 학술 논문 |
| `TavilySearchAPIRetriever` | 인터넷 검색 |

---

## 3. ElasticsearchRetriever

### 개요

- **Elasticsearch**: 분산형 REST 검색 엔진. 키워드 검색, 벡터 검색, 하이브리드 검색, 복합 필터링 모두 지원
- **ElasticsearchRetriever**: Elasticsearch의 [Query DSL](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html) 전체를 LangChain에서 자유롭게 사용할 수 있게 해주는 범용 래퍼

> `ElasticsearchStore`, `ElasticsearchEmbeddings` 등 다른 클래스로 해결되지 않을 때 사용

### 설치

```bash
pip install langchain-elasticsearch
```

### 연결 방법

```python
from elasticsearch import Elasticsearch

# 로컬 실행 (Docker 등)
es_client = Elasticsearch(hosts=["http://localhost:9200"])

# Elastic Cloud (managed 서비스)
es_client = Elasticsearch(cloud_id="...", api_key="...")
```

---

## 4. ElasticsearchRetriever 생성 패턴

`from_es_params()` 클래스 메서드를 사용해 생성합니다.

```python
retriever = ElasticsearchRetriever.from_es_params(
    index_name="인덱스명",
    body_func=쿼리함수,       # str → dict (ES Query DSL 반환 함수)
    content_field="text",     # Document.page_content로 사용할 필드명
    url="http://localhost:9200",
)
```

핵심은 **`body_func`** — 검색어(str)를 받아 ES Query DSL 딕셔너리를 반환하는 함수입니다.
이 함수만 바꾸면 검색 방식을 자유롭게 교체할 수 있습니다.

---

## 5. 검색 방식별 예시

### 5-1. Vector Search (벡터 유사도 검색)

의미가 비슷한 문서를 찾을 때 사용. 임베딩 모델이 필요합니다.

```python
def vector_query(search_query: str) -> dict:
    vector = embeddings.embed_query(search_query)
    return {
        "knn": {
            "field": "embedding_field",
            "query_vector": vector,
            "k": 5,              # 반환할 문서 수
            "num_candidates": 10 # 후보 탐색 수
        }
    }
```

### 5-2. BM25 (키워드 검색)

전통적인 단어 매칭 방식. 입력 단어가 문서에 얼마나 자주 등장하는지 기반으로 점수 계산.

```python
def bm25_query(search_query: str) -> dict:
    return {
        "query": {
            "match": {
                "text": search_query
            }
        }
    }
```

### 5-3. Hybrid Search (하이브리드 검색)

벡터 검색 + BM25를 **RRF(Reciprocal Rank Fusion)** 알고리즘으로 결합.
의미적 유사도와 키워드 매칭을 동시에 고려하므로 가장 강력합니다.

```python
def hybrid_query(search_query: str) -> dict:
    vector = embeddings.embed_query(search_query)
    return {
        "retriever": {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": {"match": {"text": search_query}}}},
                    {"knn": {"field": "embedding_field", "query_vector": vector, "k": 5, "num_candidates": 10}}
                ]
            }
        }
    }
```

### 5-4. Fuzzy Matching (오타 허용 검색)

`fuzziness: "AUTO"` 설정으로 오타가 있어도 유사한 단어를 찾습니다.

```python
def fuzzy_query(search_query: str) -> dict:
    return {
        "query": {
            "match": {
                "text": {
                    "query": search_query,
                    "fuzziness": "AUTO"  # 문자 길이에 따라 허용 오차 자동 조절
                }
            }
        }
    }
```

> 예: `"fox"` 검색 시 `"foo"`, `"for"` 등도 매칭됨

### 5-5. Complex Filtering (복합 필터)

`bool` 쿼리로 여러 조건을 조합합니다.

```python
def filter_query(search_query: str) -> dict:
    return {
        "query": {
            "bool": {
                "must": [                                    # 반드시 충족
                    {"range": {"num_chars": {"gte": 5}}}
                ],
                "must_not": [                               # 반드시 제외
                    {"prefix": {"text": "bla"}}
                ],
                "should": [                                 # 충족하면 점수 가산
                    {"match": {"text": search_query}}
                ]
            }
        }
    }
```

| 절 | 의미 |
|---|---|
| `must` | AND 조건. 반드시 충족해야 결과에 포함 |
| `must_not` | NOT 조건. 충족하면 결과에서 제외 |
| `should` | OR 조건. 충족하면 점수 가산 (결과 순위에 영향) |
| `filter` | AND 조건이지만 점수 계산 제외 (성능 유리) |

---

## 6. Custom Document Mapper

ES 검색 결과(hit)를 LangChain `Document`로 변환하는 방식을 커스터마이징할 수 있습니다.

```python
def my_mapper(hit: dict) -> Document:
    return Document(
        page_content=hit["_source"]["text"],
        metadata={
            "score": hit["_score"],
            "id": hit["_id"],
        }
    )

retriever = ElasticsearchRetriever.from_es_params(
    ...
    document_mapper=my_mapper,  # 기본 mapper 대신 사용
)
```

기본적으로 `content_field`에 지정한 필드가 `page_content`가 되고, 나머지는 `metadata`로 들어갑니다.
특정 필드만 추출하거나 `page_content`를 가공하고 싶을 때 사용합니다.

---

## 7. 검색 실행

```python
# 단건 조회
results: list[Document] = retriever.invoke("검색어")

# 각 Document 구조
# Document(
#     page_content="...",
#     metadata={"_index": "...", "_id": "...", "_score": 0.97, "_source": {...}}
# )
```

---

## 8. 검색 방식 선택 가이드

| 상황 | 권장 방식 |
|---|---|
| 정확한 단어 매칭이 중요할 때 | BM25 |
| 의미 기반 유사 문서 검색 | Vector Search |
| 정확도를 최대로 높이고 싶을 때 | Hybrid (BM25 + Vector) |
| 사용자 입력에 오타가 예상될 때 | Fuzzy Matching |
| 특정 조건으로 범위를 좁혀야 할 때 | Complex Filtering |
