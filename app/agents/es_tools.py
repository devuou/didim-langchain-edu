from langchain_core.tools import tool

from app.elasticsearch.retriever import stock_retriever

SUPPORTED_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


@tool
def get_stock_history(ticker: str, days: int = 30) -> str:
    """주어진 종목의 OHLCV 히스토리컬 데이터를 Elasticsearch에서 조회합니다.

    최근 N일간의 시가, 고가, 저가, 종가, 거래량 데이터를 반환합니다.
    주가 추세, 특정 기간 최고가/최저가, 거래량 분석 등에 활용합니다.
    지원 종목: AAPL, MSFT, TSLA, NVDA

    Args:
        ticker: 주식 티커 심볼 (예: AAPL, MSFT, TSLA, NVDA)
        days: 조회할 최근 일수 (기본값: 30, 최대: 252)
    """
    try:
        ticker = ticker.upper()
        if ticker not in SUPPORTED_TICKERS:
            return f"'{ticker}'는 지원하지 않는 종목입니다. 지원 종목: {', '.join(SUPPORTED_TICKERS)}"

        days = min(max(days, 1), 252)
        docs = stock_retriever.invoke(f"{ticker}_{days}")

        if not docs:
            return f"{ticker} 데이터를 찾을 수 없습니다."

        lines = [doc.page_content for doc in docs]
        return f"{ticker} 최근 {days}일 OHLCV 데이터 (최신순):\n" + "\n".join(lines)
    except Exception as e:
        return f"'{ticker}' 히스토리 조회 중 오류가 발생했습니다: {str(e)}"
