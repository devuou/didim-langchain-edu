from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.es_tools import get_stock_history
from app.agents.prompts import system_prompt
from app.agents.sec_search_agent import search_sec_filing
from app.agents.tools import get_company_info, get_recent_news, get_stock_price


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """에이전트의 최종 응답 스키마.

    에이전트는 항상 이 포맷으로 응답해야 한다 (시스템 프롬프트의 ChatResponse 호출 규칙 참고).
    ToolStrategy(ChatResponse)가 LLM에 이 스키마를 강제한다.
    """

    message_id: str          # UUID 형식의 메시지 식별자
    content: str             # 사용자 질문에 대한 최종 답변
    metadata: dict[str, Any] # 추가 메타데이터 (현재 빈 dict)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_stock_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None,):
    """
    ChatOpenAI 모델과 checkpointer를 받아 주식 분석 에이전트를 생성합니다.

    에이전트가 사용할 수 있는 도구:
    - get_stock_price: 실시간 주가/등락률 (yfinance)
    - get_company_info: 시가총액·PER·업종 등 기업 정보 (yfinance)
    - get_recent_news: 최신 뉴스 (yfinance)
    - get_stock_history: 과거 OHLCV 데이터 (Elasticsearch)
    - search_sec_filing: 10-K 공시 정성 정보 (BM25+kNN 서브 에이전트)

    Args:
        model: 초기화된 ChatOpenAI 인스턴스
        checkpointer: 대화 이력을 저장할 checkpointer (기본값: None, caller가 주입)

    Returns:
        create_agent()로 생성된 LangChain 에이전트
    """

    agent = create_agent(
        model=model,
        # 도구 목록: 실시간 데이터 3종 + 히스토리컬 데이터 1종 + SEC 공시 RAG 1종
        tools=[get_stock_price, get_company_info, get_recent_news, get_stock_history, search_sec_filing],
        system_prompt=system_prompt,
        # ToolStrategy: 에이전트가 반드시 ChatResponse 도구를 호출하여 응답하도록 강제
        response_format=ToolStrategy(ChatResponse),
        checkpointer=checkpointer,
    )
    return agent
