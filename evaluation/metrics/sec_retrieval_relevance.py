from opik.evaluation.metrics import BaseMetric, GEval
from opik.evaluation.metrics import score_result


class SecRetrievalRelevance(BaseMetric):
    """
    SEC 공시 서브 에이전트의 검색 관련성을 평가합니다.
    반환된 10-K 청크가 질문과 관련 있는지 GEval로 판단합니다.

    지원 종목 외(빈 결과)는 expected_output과 비교하여 정상 처리합니다.
    점수: 0.0~1.0 (높을수록 좋음)
    """

    def __init__(self, name: str = "sec_retrieval_relevance"):
        self.name = name
        self._inner = GEval(
            task_introduction=(
                "SEC 10-K 공시 문서 검색 시스템의 결과를 평가합니다. "
                "이 시스템은 AAPL, MSFT, TSLA, NVDA의 사업구조(Item 1), "
                "리스크 요인(Item 1A), 경영 성과(Item 7) 정보를 검색합니다."
            ),
            evaluation_criteria=(
                "RETRIEVED가 QUESTION에 대한 관련 정보를 포함하는지 판단하세요. "
                "EXPECTED에 '찾을 수 없습니다'가 포함된 경우, RETRIEVED도 동일한 "
                "안내 메시지를 반환해야 높은 점수를 줍니다. "
                "그 외의 경우, RETRIEVED 내용이 QUESTION과 관련된 10-K 정보를 "
                "포함하면 높은 점수, 무관하거나 비어있으면 낮은 점수를 줍니다."
            ),
            name=name,
        )

    def score(self, output: str, **kwargs) -> score_result.ScoreResult:
        input_ = kwargs.get("input", "")
        expected_output = kwargs.get("expected_output", "")
        payload = f"QUESTION: {input_}\nEXPECTED: {expected_output}\nRETRIEVED: {output}"
        return self._inner.score(output=payload)
