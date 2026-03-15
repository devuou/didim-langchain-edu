from datetime import datetime, timezone

import yfinance as yf
from elasticsearch import helpers

from app.core.config import settings
from app.elasticsearch.client import es_client
from app.utils.logger import custom_logger

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]
INDEX_NAME = f"{settings.ES_INDEX_PREFIX}-stock-data"


def _ensure_index():
    """인덱스가 없으면 생성합니다."""
    if es_client.indices.exists(index=INDEX_NAME):
        return
    es_client.indices.create(
        index=INDEX_NAME,
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
                    "ingested_at": {"type": "date"},
                }
            }
        },
    )
    custom_logger.info(f"ES 인덱스 생성: {INDEX_NAME}")


def _build_actions(ticker: str) -> list[dict]:
    """yfinance에서 1년치 OHLCV 데이터를 가져와 ES bulk 액션 리스트로 변환합니다."""
    df = yf.Ticker(ticker).history(period="1y")
    df.reset_index(inplace=True)

    now = datetime.now(timezone.utc).isoformat()
    actions = []
    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d")
        actions.append({
            "_index": INDEX_NAME,
            "_id": f"{ticker}_{date_str}",  # upsert 키: 재시작 시 중복 방지
            "_source": {
                "ticker":      ticker,
                "date":        date_str,
                "open":        round(float(row["Open"]), 4),
                "high":        round(float(row["High"]), 4),
                "low":         round(float(row["Low"]), 4),
                "close":       round(float(row["Close"]), 4),
                "volume":      int(row["Volume"]),
                "ingested_at": now,
            },
        })
    return actions


def ingest_all():
    """고정 종목 전체를 ES에 적재합니다. 앱 시작 시 1회 호출."""
    _ensure_index()
    for ticker in TICKERS:
        try:
            actions = _build_actions(ticker)
            helpers.bulk(es_client, actions)
            custom_logger.info(f"ES 적재 완료: {ticker} {len(actions)}건")
        except Exception as e:
            custom_logger.error(f"ES 적재 실패: {ticker} - {e}")
