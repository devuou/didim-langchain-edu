from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.agents.prompts import system_prompt
from app.agents.tools import get_company_info, get_recent_news, get_stock_price


# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """에이전트의 최종 응답 스키마."""

    message_id: str
    content: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_stock_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None,):
    """
    ChatOpenAI 모델과 checkpointer를 받아 주식 분석 에이전트를 생성합니다.

    Args:
        model: 초기화된 ChatOpenAI 인스턴스
        checkpointer: 대화 이력을 저장할 checkpointer (기본값: MemorySaver)

    Returns:
        create_agent()로 생성된 LangChain 에이전트
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_stock_price, get_company_info, get_recent_news],
        system_prompt=system_prompt,
        response_format=ToolStrategy(ChatResponse),
        checkpointer=checkpointer,
    )
    return agent
