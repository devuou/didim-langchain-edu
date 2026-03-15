"""
Q14. import 경로 교정 — 실제 import 오류 여부 검증

각 선지를 실제로 import하여 잘못된 경로(존재하지 않거나 deprecated)인지 확인합니다.
"""
import pytest


def test_1_langchain_agents_structured_output_ToolStrategy():
    """(1) from langchain.agents.structured_output import ToolStrategy"""
    try:
        from langchain.agents.structured_output import ToolStrategy
        print("\n(1) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(1) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(1) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_2_langchain_tools_tool():
    """(2) from langchain.tools import tool"""
    try:
        from langchain.tools import tool
        print("\n(2) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(2) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(2) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_3_langchain_core_messages_HumanMessage():
    """(3) from langchain_core.messages import HumanMessage"""
    try:
        from langchain_core.messages import HumanMessage
        print("\n(3) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(3) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(3) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_4_langchain_agents_create_agent():
    """(4) from langchain.agents import create_agent"""
    try:
        from langchain.agents import create_agent
        print("\n(4) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(4) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(4) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")


def test_5_langchain_core_tools_tool():
    """(5) from langchain_core.tools import tool as core_tool"""
    try:
        from langchain_core.tools import tool as core_tool
        print("\n(5) ✅ 정상 import 성공")
    except ImportError as e:
        print(f"\n(5) ❌ ImportError: {e}")
        pytest.fail(f"ImportError 발생: {e}")
    except Exception as e:
        print(f"\n(5) ❌ {type(e).__name__}: {e}")
        pytest.fail(f"오류 발생: {e}")
