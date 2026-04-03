import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_sec_eval_task_calls_graph_with_ticker_and_query(monkeypatch):
    """sec_eval_task가 ticker, query를 그래프에 전달하는지 확인"""
    from unittest.mock import MagicMock
    import evaluation.run_eval_sec as module

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"result": "Apple의 주요 사업은 iPhone입니다."}
    monkeypatch.setattr(module, "_graph", mock_graph)

    result = module.sec_eval_task({
        "ticker": "AAPL",
        "query": "애플의 주요 사업은?",
        "expected_output": "iPhone 등 주요 부문",
    })

    mock_graph.invoke.assert_called_once_with({"ticker": "AAPL", "query": "애플의 주요 사업은?"})
    assert result["output"] == "Apple의 주요 사업은 iPhone입니다."
    assert result["input"] == "애플의 주요 사업은?"
    assert result["expected_output"] == "iPhone 등 주요 부문"


def test_sec_eval_task_returns_required_keys(monkeypatch):
    """반환 dict에 input, output, expected_output 키가 있는지 확인"""
    from unittest.mock import MagicMock
    import evaluation.run_eval_sec as module

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"result": "관련 공시 정보를 찾을 수 없습니다."}
    monkeypatch.setattr(module, "_graph", mock_graph)

    result = module.sec_eval_task({"ticker": "AMZN", "query": "아마존 리스크", "expected_output": "찾을 수 없습니다."})

    assert "input" in result
    assert "output" in result
    assert "expected_output" in result


def test_level_config_l1_is_4_samples():
    """L1 설정이 4건인지 확인"""
    import evaluation.run_eval_sec as module
    assert module.LEVEL_CONFIG["L1"]["nb_samples"] == 4


def test_level_config_l2_is_10_samples():
    """L2 설정이 10건인지 확인"""
    import evaluation.run_eval_sec as module
    assert module.LEVEL_CONFIG["L2"]["nb_samples"] == 10
