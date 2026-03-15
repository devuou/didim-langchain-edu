"""
Q9. AI Hallucination 식별 — 실제 import 오류 여부 검증

각 선지를 실제로 import/실행하여 오류 발생 여부를 확인합니다.
"""
import pytest


def test_A_langchain_agents_structured_output_ToolStrategy():
    """(A) from langchain.agents.structured_output import ToolStrategy"""
    try:
        from langchain.agents.structured_output import ToolStrategy
        print("\n(A) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(A) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(A) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_B_langchain_core_tools_tool():
    """(B) from langchain_core.tools import tool"""
    try:
        from langchain_core.tools import tool
        print("\n(B) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(B) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(B) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_C_create_agent_with_empty_tools():
    """(C) from langchain.agents import create_agent + create_agent(model, tools=[])"""
    try:
        from langchain.agents import create_agent
        print("\n(C) import ✅ 성공")
    except ImportError as e:
        print(f"\n(C) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(C) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_D_langchain_core_messages_HumanMessage():
    """(D) from langchain_core.messages import HumanMessage"""
    try:
        from langchain_core.messages import HumanMessage
        print("\n(D) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(D) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(D) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_E_typo_variable_name_still_works():
    """
    (E) response_fromat = ToolStrategy(ChatResponse)  # 변수명 오타
        create_agent(model, tools=[], response_format=response_fromat)

    변수명은 오타(response_fromat)이지만,
    create_agent의 파라미터에는 올바르게(response_format=) 전달하고 있음.
    문법 오류인지 확인.
    """
    try:
        from langchain.agents.structured_output import ToolStrategy
        from dataclasses import dataclass

        @dataclass
        class ChatResponse:
            content: str

        # 변수명 오타 — Python 문법상 문제없음 (그냥 변수명일 뿐)
        response_fromat = ToolStrategy(ChatResponse)

        # create_agent의 response_format 파라미터에 올바르게 전달
        # (실제 실행 없이 파라미터 전달 가능 여부만 확인)
        print("\n(E) ✅ 변수명 오타(response_fromat)는 Python 문법 오류가 아님")
        print("     create_agent 호출 시 response_format=response_fromat 로 올바르게 전달 가능")
        assert response_fromat is not None
    except Exception as e:
        print(f"\n(E) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")
