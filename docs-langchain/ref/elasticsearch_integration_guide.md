# yfinance → Elasticsearch → LangChain Agent 연동 가이드

## 전체 아키텍처

```
yfinance (데이터 수집)
    → Elasticsearch Ingest Pipeline (변환/정규화)
        → Elasticsearch Index (저장)
            → ElasticsearchRetriever (LangChain)
                → Agent Tools
```

---

## 1. Elasticsearch 인덱스 & Ingest Pipeline 구성

### 인덱스 매핑 생성

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(
    "http://localhost:9200",
    api_key="YOUR_API_KEY"  # 또는 basic_auth=("elastic", "password")
)

# 주가 데이터 인덱스 매핑
es.indices.create(
    index="stock-data",
    body={
        "mappings": {
            "properties": {
                "ticker":      {"type": "keyword"},
                "date":        {"type": "date"},
                "open":        {"type": "float"},
                "high":        {"type": "float"},
                "low":         {"type": "float"},
                "close":       {"type": "float"},
                "volume":      {"type": "long"},
                "ingested_at": {"type": "date"}
            }
        }
    }
)
```

### Ingest Pipeline 생성

인덱싱 전 데이터를 자동으로 전처리합니다.
Elasticsearch Ingest Pipeline은 processor들을 순차 실행하며, `set`, `date`, `rename` 등 다양한 processor를 지원합니다.

```json
PUT _ingest/pipeline/stock-data-pipeline
{
  "description": "yfinance 주가 데이터 전처리 파이프라인",
  "processors": [
    {
      "set": {
        "description": "수집 시각 기록",
        "field": "ingested_at",
        "value": "{{{_ingest.timestamp}}}"
      }
    },
    {
      "lowercase": {
        "description": "ticker 소문자 통일 방지 - 필요시 제거",
        "field": "ticker",
        "ignore_failure": true
      }
    },
    {
      "set": {
        "description": "티커를 대문자로 다시 설정",
        "field": "ticker",
        "value": "{{ticker}}"
      }
    }
  ],
  "on_failure": [
    {
      "set": {
        "field": "error_information",
        "value": "Pipeline failed: {{_ingest.on_failure_message}}"
      }
    }
  ]
}
```

---

## 2. yfinance 데이터 수집 및 Elasticsearch 저장

```python
import yfinance as yf
from elasticsearch import Elasticsearch, helpers
from datetime import datetime

es = Elasticsearch("http://localhost:9200", api_key="YOUR_API_KEY")

def ingest_stock_data(ticker: str, period: str = "1y"):
    """yfinance에서 데이터를 받아 ES에 bulk 저장"""
    df = yf.Ticker(ticker).history(period=period)
    df.reset_index(inplace=True)

    actions = []
    for _, row in df.iterrows():
        doc = {
            "_index": "stock-data",
            "_source": {
                "ticker":  ticker,
                "date":    row["Date"].strftime("%Y-%m-%d"),
                "open":    round(float(row["Open"]), 4),
                "high":    round(float(row["High"]), 4),
                "low":     round(float(row["Low"]), 4),
                "close":   round(float(row["Close"]), 4),
                "volume":  int(row["Volume"]),
            }
        }
        actions.append(doc)

    # pipeline 파라미터로 Ingest Pipeline 적용
    helpers.bulk(es, actions, pipeline="stock-data-pipeline")
    print(f"{ticker}: {len(actions)}건 저장 완료")

# 원하는 종목 미리 수집
for ticker in ["AAPL", "MSFT", "TSLA", "NVDA"]:
    ingest_stock_data(ticker)
```

---

## 3. yfinance → Elasticsearch 파이프라인 트리거링 방식

### 방식 1: 주기적 스케줄링 (가장 일반적)

Agent가 질문을 받기 전, 별도 프로세스가 주기적으로 ES를 최신 상태로 채웁니다.

```
[스케줄러] → ingest_stock_data() 호출 → [yfinance] → [ES]
                                                          ↑
[Agent] ← 질문 도착 ← [사용자]                    미리 적재된 데이터
    ↓
[ElasticsearchRetriever]
```

cron, APScheduler, Airflow 등을 사용합니다.
예를 들어 장 마감 후 매일 오후 5시에 당일 데이터를 적재하는 식입니다.

### 방식 2: Agent Tool 내부에서 On-demand 적재

Agent가 질문을 받을 때 Tool이 직접 yfinance를 조회하고, ES에 없으면 그때 가져와 캐싱합니다.

```python
@tool
def get_stock_price(ticker: str) -> str:
    # 1. ES에 데이터가 있는지 먼저 확인
    result = es_client.search(
        index="stock-data",
        body={"query": {"term": {"ticker": ticker}}, "size": 1}
    )

    if result["hits"]["total"]["value"] == 0:
        # 2. 없으면 yfinance에서 가져와 ES에 저장
        ingest_stock_data(ticker)

    # 3. ES에서 조회
    docs = retriever.invoke({"ticker": ticker})
    ...
```

요청이 들어올 때 처음 한 번만 yfinance를 호출하고, 이후엔 ES 캐시를 사용합니다.
단, 데이터 신선도 관리가 별도로 필요합니다.

### 방식 3: 하이브리드 (TTL 기반 캐시)

ES에 `ingested_at` 필드를 두고, 마지막 적재 시각이 N시간을 초과했으면 yfinance를 다시 호출합니다.

```python
@tool
def get_stock_price(ticker: str) -> str:
    # 마지막 적재 시각 확인
    result = es_client.search(
        index="stock-data",
        body={
            "query": {"term": {"ticker": ticker}},
            "sort": [{"ingested_at": {"order": "desc"}}],
            "size": 1
        }
    )

    needs_refresh = True
    if result["hits"]["total"]["value"] > 0:
        last_ingested = result["hits"]["hits"][0]["_source"]["ingested_at"]
        elapsed = datetime.now() - datetime.fromisoformat(last_ingested)
        needs_refresh = elapsed.total_seconds() > 3600  # 1시간 TTL

    if needs_refresh:
        ingest_stock_data(ticker, period="5d")  # 최근 5일치만 갱신

    docs = retriever.invoke({"ticker": ticker})
    ...
```

### 트리거링 방식 선택 기준

| 상황 | 추천 방식 |
|---|---|
| 주가처럼 갱신 주기가 명확한 경우 | 방식 1 (스케줄링) |
| 조회할 종목이 불특정 다수인 경우 | 방식 2 (On-demand) |
| 자주 쓰이는 종목은 캐시, 나머지는 실시간이 필요한 경우 | 방식 3 (하이브리드) |

> 주가 데이터는 갱신 주기가 장 마감 기준으로 명확하기 때문에 **방식 1(스케줄링)** 이 가장 깔끔합니다.

---

## 4. LangChain ElasticsearchRetriever 기반 Tool 구성

```python
from langchain_elasticsearch import ElasticsearchRetriever
from langchain.tools import tool
from elasticsearch import Elasticsearch

es_client = Elasticsearch("http://localhost:9200", api_key="YOUR_API_KEY")

def build_stock_query(ticker: str, days: int = 30):
    """특정 종목의 최근 N일 데이터 쿼리"""
    return {
        "size": days,
        "query": {
            "term": {"ticker": ticker.upper()}
        },
        "sort": [{"date": {"order": "desc"}}]
    }

retriever = ElasticsearchRetriever(
    index_name="stock-data",
    body_func=lambda q: build_stock_query(q["ticker"], q.get("days", 30)),
    content_field="close",   # Document.page_content에 담길 필드
    es_client=es_client
)

@tool
def get_stock_price(ticker: str) -> str:
    """주어진 종목의 최근 30일 종가 데이터를 Elasticsearch에서 조회합니다."""
    docs = retriever.invoke({"ticker": ticker, "days": 30})
    if not docs:
        return f"{ticker} 데이터를 찾을 수 없습니다."

    results = []
    for doc in docs:
        meta = doc.metadata
        results.append(f"{meta.get('date')}: 종가={meta.get('close')}, 거래량={meta.get('volume')}")

    return "\n".join(results)

@tool
def get_stock_high_low(ticker: str) -> str:
    """주어진 종목의 최근 30일 고가/저가를 Elasticsearch에서 조회합니다."""
    docs = retriever.invoke({"ticker": ticker, "days": 30})
    if not docs:
        return f"{ticker} 데이터를 찾을 수 없습니다."

    highs = [doc.metadata.get("high", 0) for doc in docs]
    lows  = [doc.metadata.get("low", 0)  for doc in docs]
    return f"{ticker} - 30일 최고가: {max(highs):.2f}, 최저가: {min(lows):.2f}"
```

---

## 5. Agent에 Tool 등록

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
tools = [get_stock_price, get_stock_high_low]

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 주식 분석 어시스턴트입니다. Elasticsearch에 저장된 주가 데이터를 활용합니다."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
agent_executor.invoke({"input": "AAPL의 최근 30일 최고가와 최저가를 알려줘"})
```

---

## 6. 추가로 알아야 할 사항

### 6-1. 검색 전략 선택

주가 데이터처럼 숫자/날짜 필터 중심 데이터는 BM25 또는 `ElasticsearchRetriever`의 `bool` 쿼리 DSL이 가장 적합합니다.
벡터 전략은 임베딩 생성 비용이 발생하므로 필요할 때만 사용합니다.

| 전략 | 적합한 경우 |
|---|---|
| **BM25 (키워드)** | 종목명, 날짜처럼 정확한 값을 필터링할 때 |
| **Dense Vector (의미검색)** | "기술주 중 변동성이 큰 종목" 같은 자연어 질의 |
| **Hybrid (BM25 + Vector)** | 두 경우를 모두 커버하고 싶을 때 |
| **Exact Score** | 정확도가 중요하고 데이터가 적을 때 |

### 6-2. ElasticsearchRetriever 선택 이유

`ElasticsearchRetriever`는 **Query DSL을 직접 작성**할 수 있어 범용적이며, 이미 있는 인덱스에 붙이기 좋습니다.
주가 데이터처럼 스키마가 고정된 경우에 적합하며, 임베딩 모델 없이도 사용 가능합니다.

- Query DSL을 직접 제어할 수 있어 range, term, bool 쿼리 등 세밀한 필터링 가능
- 이미 구성된 인덱스에 그대로 붙여 사용 가능
- 임베딩 모델 불필요 → 비용 및 지연 없음
- `document_mapper`로 결과 형태를 자유롭게 커스터마이징 가능

### 6-3. `document_mapper`로 컨텍스트 품질 높이기

LLM이 읽기 좋은 형태로 Document를 가공하면 응답 품질이 올라갑니다.

```python
def stock_document_mapper(hit: dict) -> Document:
    src = hit["_source"]
    content = (
        f"[{src['ticker']}] {src['date']} | "
        f"시가:{src['open']} 고가:{src['high']} 저가:{src['low']} 종가:{src['close']} "
        f"거래량:{src['volume']:,}"
    )
    return Document(
        page_content=content,
        metadata={"ticker": src["ticker"], "date": src["date"]}
    )

retriever = ElasticsearchRetriever.from_es_params(
    index_name="stock-data",
    body_func=build_stock_query,
    document_mapper=stock_document_mapper,  # 커스텀 매퍼 적용
    url="http://localhost:9200",
)
```

### 6-4. 인증 및 연결 방식

운영 환경에서는 반드시 인증을 설정해야 합니다.

```python
# API Key 방식 (권장)
es_client = Elasticsearch(
    "http://localhost:9200",
    api_key="your_api_key_here"
)

# 계정/비밀번호 방식
es_client = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "your_password")
)

# ElasticsearchRetriever에서도 동일하게
retriever = ElasticsearchRetriever.from_es_params(
    index_name="stock-data",
    body_func=build_stock_query,
    content_field="close",
    url="http://localhost:9200",
    es_user="elastic",
    es_password="your_password",
)
```

### 6-5. `ElasticsearchCache` - LLM 응답 캐싱

ES를 데이터 저장소로만 쓰는 게 아니라, **LLM 응답 자체를 캐시**하는 용도로도 사용할 수 있습니다.
동일한 질문이 반복될 때 LLM 호출 없이 ES에서 직접 꺼내 줍니다.

```python
from langchain_elasticsearch import ElasticsearchCache
from langchain_core.globals import set_llm_cache

set_llm_cache(
    ElasticsearchCache(
        es_url="http://localhost:9200",
        index_name="llm-cache",
        es_user="elastic",
        es_password="your_password",
    )
)
# 이후 llm.invoke()는 자동으로 캐시를 먼저 확인
```

### 6-6. Bulk 인덱싱 타임아웃 대응

데이터가 많거나 네트워크가 느릴 때 bulk insert에서 타임아웃이 발생할 수 있습니다.

```python
helpers.bulk(
    es,
    actions,
    pipeline="stock-data-pipeline",
    chunk_size=200,             # 기본값 500 → 줄이기
    max_chunk_bytes=50_000_000  # 기본값 100MB → 줄이기
)
```

### 6-7. 인덱스 매핑 주의사항

매핑은 처음 한 번만 설정할 수 있고, 필드 타입을 나중에 바꾸기가 어렵습니다.
특히 벡터 전략으로 전환할 경우 `dense_vector` 필드의 `dims`와 `similarity` 알고리즘을 나중에 변경하려면 인덱스를 새로 만들어야 합니다.
처음 설계 시 필드 타입을 신중하게 결정해 두는 것이 중요합니다.

---

## 우선순위 체크리스트

| 시점 | 항목 |
|---|---|
| **운영 전 필수** | 인증 설정 (API Key 또는 계정/비밀번호) |
| **운영 전 필수** | 인덱스 매핑 확정 (나중에 변경 어려움) |
| **바로 적용 권장** | `document_mapper` 커스터마이징 |
| **바로 적용 권장** | Ingest Pipeline을 통한 데이터 정규화 |
| **규모 커지면 고려** | `ElasticsearchCache`로 LLM 비용 절감 |
| **규모 커지면 고려** | Bulk 파라미터 튜닝 |
