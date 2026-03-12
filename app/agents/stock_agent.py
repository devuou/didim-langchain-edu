import uuid

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.prompts import system_prompt
from app.agents.tools import get_company_info, get_recent_news, get_stock_price
from app.core.config import settings


@tool
def ChatResponse(message_id: str, content: str, metadata: dict = {}) -> str:
    """모든 정보 수집이 완료된 후 최종 답변을 사용자에게 전달할 때 사용합니다.
    반드시 분석의 마지막 단계에서만 호출합니다.

    Args:
        message_id: 응답의 고유 식별자 (UUID 형식)
        content: 사용자에게 전달할 최종 답변
        metadata: 추가 메타데이터 (기본값: 빈 딕셔너리)
    """
    return content


class Agent:
    def __init__(self):
        llm = ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY)

        stock_tools = [get_stock_price, get_company_info, get_recent_news]
        all_tools = stock_tools + [ChatResponse]

        self.llm_with_tools = llm.bind_tools(all_tools)
        self.tool_node = ToolNode(stock_tools)
        self.graph = self._build_graph()

    def _build_graph(self):
        def model(state: MessagesState):
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.llm_with_tools.invoke(messages)
            # LLM이 tool 호출 없이 텍스트로 직접 답변한 경우 ChatResponse로 감싸서 반환
            if not response.tool_calls and response.content:
                response = AIMessage(
                    content="",
                    tool_calls=[{
                        "id": str(uuid.uuid4()),
                        "name": "ChatResponse",
                        "args": {
                            "message_id": str(uuid.uuid4()),
                            "content": response.content,
                            "metadata": {},
                        },
                        "type": "tool_call",
                    }]
                )
            return {"messages": [response]}

        def should_continue(state: MessagesState):
            last_message = state["messages"][-1]
            if not last_message.tool_calls:
                return END
            if last_message.tool_calls[0]["name"] == "ChatResponse":
                return "execute_chat_response"
            return "tools"

        def execute_chat_response(state: MessagesState):
            """ChatResponse tool_call에 대응하는 ToolMessage를 추가해 대화 히스토리를 유효하게 유지"""
            last_message = state["messages"][-1]
            call = last_message.tool_calls[0]
            return {"messages": [ToolMessage(
                content=call["args"].get("content", ""),
                tool_call_id=call["id"],
                name="ChatResponse",
            )]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", model)
        graph.add_node("tools", self.tool_node)
        graph.add_node("execute_chat_response", execute_chat_response)
        graph.add_edge(START, "model")
        graph.add_conditional_edges("model", should_continue)
        graph.add_edge("tools", "model")
        graph.add_edge("execute_chat_response", END)

        return graph.compile(checkpointer=MemorySaver())

    async def astream(self, input_data: dict, config: dict = None, stream_mode: str = "updates"):
        async for chunk in self.graph.astream(input_data, config=config, stream_mode=stream_mode):
            yield chunk
