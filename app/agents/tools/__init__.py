import re
from datetime import datetime, timezone, timedelta

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
            market = info.get("market", "")
            if market and market != "us_market":
                return f"'{ticker}'의 주가 데이터를 가져올 수 없습니다. 해외 주식(미국 외)은 일부 데이터가 제공되지 않을 수 있습니다. 미국 주식 티커(예: AAPL, TSLA)를 이용해주세요."
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

        # 미국 외 시장은 일부 데이터 미제공 안내
        market = info.get("market", "")
        if market and market != "us_market":
            return f"'{ticker}'는 미국 외 시장 종목으로, 일부 재무 데이터가 제공되지 않을 수 있습니다. 미국 주식 티커(예: AAPL, TSLA)를 이용해주세요."

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


# 티커별 관련 키워드 매핑 (뉴스 필터링용)
_TICKER_KEYWORDS: dict[str, list[str]] = {
    "AAPL": [
        "Apple", "AAPL", "iPhone", "iPad", "Mac", "Tim Cook",
        "Apple Intelligence", "App Store", "Vision Pro", "MacBook",
        "AirPods", "iOS", "macOS", "Apple Watch", "Siri",
    ],
    "TSLA": [
        "Tesla", "TSLA", "Elon Musk", "EV", "electric vehicle",
        "Cybertruck", "Model S", "Model 3", "Model X", "Model Y",
        "FSD", "Full Self-Driving", "Autopilot", "Robotaxi", "Gigafactory",
    ],
    "NVDA": [
        "Nvidia", "NVIDIA", "NVDA", "GPU", "Jensen Huang",
        "GeForce", "RTX", "CUDA", "H100", "Blackwell", "Hopper",
    ],
    "MSFT": [
        "Microsoft", "MSFT", "Windows", "Azure", "Satya Nadella",
        "Xbox", "GitHub", "LinkedIn", "Bing",
    ],
}
_NEWS_FETCH_LIMIT = 30   # yfinance에서 최대 가져올 뉴스 수
_NEWS_RETURN_LIMIT = 3   # 최종적으로 반환할 뉴스 수
_NEWS_MAX_AGE_DAYS = 30  # 이 기간(일) 이내 뉴스만 포함


@tool
def get_recent_news(ticker: str) -> str:
    """해당 주식과 관련된 최근 뉴스 최대 3건을 조회합니다.

    사용자가 특정 주식 관련 최신 뉴스, 소식, 이슈를 물어볼 때 사용합니다.
    yfinance에서 최대 10건을 가져온 후 티커/회사명이 직접 언급된 뉴스만 필터링합니다.

    Args:
        ticker: 주식 티커 심볼 (예: AAPL, TSLA, NVDA, 005930.KS)
    """
    try:
        ticker_upper = ticker.upper()
        stock = yf.Ticker(ticker_upper)
        news = stock.news

        if not news:
            return "관련 뉴스를 찾을 수 없습니다."

        # 필터링 키워드: 등록된 종목은 매핑 사용, 그 외는 티커 심볼만 사용
        keywords = _TICKER_KEYWORDS.get(ticker_upper, [ticker_upper])

        def _is_relevant(article: dict) -> bool:
            content = article.get("content", {})
            title = (content.get("title") or article.get("title", "")).lower()
            summary = (content.get("summary") or article.get("summary", "")).lower()
            text = title + " " + summary
            # 단어 경계(\b) 기반 매칭 — "EV"가 "seven", "ever" 등에 오탐되는 것을 방지
            return any(re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE) for kw in keywords)

        def _is_recent(article: dict, cutoff: datetime) -> bool:
            # yfinance 신형 구조: content.pubDate (ISO 8601 문자열)
            pub_str = article.get("content", {}).get("pubDate")
            if not pub_str:
                return True  # 날짜 정보 없으면 통과
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                return pub_dt >= cutoff
            except ValueError:
                return True

        cutoff = datetime.now(timezone.utc) - timedelta(days=_NEWS_MAX_AGE_DAYS)

        # 최대 _NEWS_FETCH_LIMIT 건에서 관련 뉴스 필터링 (키워드 + 날짜)
        relevant = [
            a for a in news[:_NEWS_FETCH_LIMIT]
            if _is_relevant(a) and _is_recent(a, cutoff)
        ]

        # 관련 뉴스가 1건 이상이면 필터링 결과 사용, 0건이면 "없음" 반환
        if not relevant:
            return f"현재 {ticker_upper}와 직접 관련된 최근 {_NEWS_MAX_AGE_DAYS}일 이내 뉴스를 찾을 수 없습니다."
        candidates = relevant

        result = []
        for i, article in enumerate(candidates[:_NEWS_RETURN_LIMIT], 1):
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
