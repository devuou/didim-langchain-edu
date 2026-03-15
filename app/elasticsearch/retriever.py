from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchRetriever

from app.core.config import settings
from app.elasticsearch.client import es_client

INDEX_NAME = f"{settings.ES_INDEX_PREFIX}-stock-data"


def _build_query(query: str) -> dict:
    """
    "TICKER_days" 형식 문자열을 받아 ES Query DSL을 반환합니다.
    예: "AAPL_30" → AAPL 최근 30건을 날짜 내림차순으로 조회
    """
    parts = query.split("_")
    ticker = parts[0].upper()
    days = int(parts[1]) if len(parts) > 1 else 30

    return {
        "size": days,
        "query": {
            "bool": {
                "must": [
                    {"term": {"ticker": ticker}}
                ]
            }
        },
        "sort": [{"date": {"order": "desc"}}],
    }


def _stock_document_mapper(hit: dict) -> Document:
    """ES hit을 LLM이 읽기 좋은 Document로 변환합니다."""
    src = hit["_source"]
    content = (
        f"[{src['ticker']}] {src['date']} | "
        f"시가:{src['open']:.2f} 고가:{src['high']:.2f} "
        f"저가:{src['low']:.2f} 종가:{src['close']:.2f} "
        f"거래량:{src['volume']:,}"
    )
    return Document(
        page_content=content,
        metadata={"ticker": src["ticker"], "date": src["date"]},
    )


stock_retriever = ElasticsearchRetriever(
    index_name=INDEX_NAME,
    body_func=_build_query,
    document_mapper=_stock_document_mapper,
    client=es_client,
)
