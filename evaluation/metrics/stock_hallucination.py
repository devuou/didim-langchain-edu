from opik.evaluation.metrics import BaseMetric, Hallucination
from opik.evaluation.metrics import score_result


class StockHallucination(BaseMetric):
    """
    주식 에이전트 응답의 환각(hallucination) 여부를 평가합니다.
    없는 주가·수치를 지어냈는지 탐지합니다.
    """

    def __init__(self, name: str = "stock_hallucination"):
        self.name = name
        self._inner = Hallucination()

    def score(self, output: str, **kwargs) -> score_result.ScoreResult:
        input = kwargs.get("input", "")
        return self._inner.score(input=input, output=output)
