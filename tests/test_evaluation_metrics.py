import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── SecRetrievalRelevance ────────────────────────────────────────────────────

def test_sec_retrieval_relevance_payload_format(monkeypatch):
    """score()가 QUESTION/EXPECTED/RETRIEVED 형식으로 GEval에 전달하는지 확인"""
    from evaluation.metrics.sec_retrieval_relevance import SecRetrievalRelevance
    from opik.evaluation.metrics.score_result import ScoreResult

    metric = SecRetrievalRelevance()
    captured = {}

    def mock_geval_score(output):
        captured["payload"] = output
        return ScoreResult(value=0.8, name="sec_retrieval_relevance")

    monkeypatch.setattr(metric._inner, "score", mock_geval_score)

    metric.score(
        output="Apple의 주요 사업은 iPhone입니다.",
        input="애플 주요 사업은?",
        expected_output="iPhone, Mac 등 주요 부문",
    )

    assert "QUESTION: 애플 주요 사업은?" in captured["payload"]
    assert "EXPECTED: iPhone, Mac 등 주요 부문" in captured["payload"]
    assert "RETRIEVED: Apple의 주요 사업은 iPhone입니다." in captured["payload"]


def test_sec_retrieval_relevance_returns_score_result(monkeypatch):
    """score()가 ScoreResult를 반환하는지 확인"""
    from evaluation.metrics.sec_retrieval_relevance import SecRetrievalRelevance
    from opik.evaluation.metrics.score_result import ScoreResult

    metric = SecRetrievalRelevance()
    monkeypatch.setattr(
        metric._inner, "score",
        lambda output: ScoreResult(value=0.9, name="sec_retrieval_relevance")
    )

    result = metric.score(output="검색 결과", input="질문", expected_output="기대값")

    assert isinstance(result, ScoreResult)
    assert result.value == 0.9


# ─── SecGroundedness ──────────────────────────────────────────────────────────

def test_sec_groundedness_returns_none_when_no_context():
    """context 없으면 None 반환 (해당 항목 skip)"""
    from evaluation.metrics.sec_groundedness import SecGroundedness

    metric = SecGroundedness()
    result = metric.score(output="AAPL 주가는 $150입니다.", input="AAPL 주가는?", context=[])
    assert result is None


def test_sec_groundedness_returns_none_when_context_missing_from_kwargs():
    """context 키 자체가 없어도 None 반환"""
    from evaluation.metrics.sec_groundedness import SecGroundedness

    metric = SecGroundedness()
    result = metric.score(output="답변", input="질문")
    assert result is None


def test_sec_groundedness_scores_when_context_present(monkeypatch):
    """context가 있으면 Hallucination 메트릭을 호출하고 ScoreResult 반환"""
    from evaluation.metrics.sec_groundedness import SecGroundedness
    from opik.evaluation.metrics.score_result import ScoreResult

    metric = SecGroundedness()
    mock_result = ScoreResult(value=0.1, name="sec_groundedness")
    monkeypatch.setattr(metric._inner, "score", lambda **kwargs: mock_result)

    result = metric.score(
        output="애플의 주요 리스크는 공급망 문제입니다.",
        input="AAPL 리스크는?",
        context=["[1] 섹션: item1a | 티커: AAPL\n공급망 리스크 관련 내용..."],
    )

    assert result is not None
    assert result.value == 0.1


def test_sec_groundedness_passes_correct_args_to_inner(monkeypatch):
    """inner Hallucination에 input, output, context가 올바르게 전달되는지 확인"""
    from evaluation.metrics.sec_groundedness import SecGroundedness
    from opik.evaluation.metrics.score_result import ScoreResult

    metric = SecGroundedness()
    captured = {}

    def mock_score(**kwargs):
        captured.update(kwargs)
        return ScoreResult(value=0.0, name="sec_groundedness")

    monkeypatch.setattr(metric._inner, "score", mock_score)

    context_chunks = ["chunk1 내용", "chunk2 내용"]
    metric.score(output="최종 답변", input="사용자 질문", context=context_chunks)

    assert captured["input"] == "사용자 질문"
    assert captured["output"] == "최종 답변"
    assert captured["context"] == context_chunks
