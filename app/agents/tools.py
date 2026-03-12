import yfinance as yf
from langchain_core.tools import tool


@tool
def get_stock_price(ticker: str) -> str:
    """현재 주가와 전일 대비 등락률을 조회합니다.

    사용자가 특정 주식의 현재 가격, 주가, 등락률을 물어볼 때 사용합니다.

    Args:
        ticker: 주식 티커 심볼 (예: AAPL, TSLA, NVDA, 005930.KS)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

        if current_price is None or prev_close is None:
            return f"'{ticker}' 티커의 주가 데이터를 가져올 수 없습니다. 올바른 티커 심볼인지 확인해주세요."

        change_pct = ((current_price - prev_close) / prev_close) * 100
        sign = "+" if change_pct >= 0 else ""

        return f"{ticker.upper()} 현재가: ${current_price:.2f} | 등락률: {sign}{change_pct:.2f}%"
    except Exception as e:
        return f"'{ticker}' 주가 조회 중 오류가 발생했습니다: {str(e)}"


@tool
def get_company_info(ticker: str) -> str:
    """기업의 기본 재무 정보(시가총액, PER, 업종)를 조회합니다.

    사용자가 기업 규모, 밸류에이션, 업종 등 기업 기본 정보를 물어볼 때 사용합니다.

    Args:
        ticker: 주식 티커 심볼 (예: AAPL, TSLA, NVDA, 005930.KS)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        market_cap = info.get("marketCap")
        if market_cap:
            trillion = market_cap / 1_000_000_000_000
            billion = market_cap / 100_000_000
            market_cap_str = f"{trillion:.2f}조 달러" if trillion >= 1 else f"{billion:.2f}억 달러"
        else:
            market_cap_str = "N/A"

        per = info.get("trailingPE")
        per_str = f"{per:.2f}" if per else "N/A"

        sector = info.get("sector") or "N/A"

        return f"{ticker.upper()} | 시가총액: {market_cap_str} | PER: {per_str} | 업종: {sector}"
    except Exception as e:
        return f"'{ticker}' 기업 정보 조회 중 오류가 발생했습니다: {str(e)}"


@tool
def get_recent_news(ticker: str) -> str:
    """해당 주식의 최근 뉴스 최대 3건을 조회합니다.

    사용자가 특정 주식 관련 최신 뉴스, 소식, 이슈를 물어볼 때 사용합니다.

    Args:
        ticker: 주식 티커 심볼 (예: AAPL, TSLA, NVDA, 005930.KS)
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            return "관련 뉴스를 찾을 수 없습니다."

        result = []
        for i, article in enumerate(news[:3], 1):
            content = article.get("content", {})
            title = content.get("title") or article.get("title", "제목 없음")
            link = (
                content.get("canonicalUrl", {}).get("url")
                or article.get("link", "링크 없음")
            )
            result.append(f"{i}. {title}\n   {link}")

        return "\n\n".join(result)
    except Exception as e:
        return f"'{ticker}' 뉴스 조회 중 오류가 발생했습니다: {str(e)}"
