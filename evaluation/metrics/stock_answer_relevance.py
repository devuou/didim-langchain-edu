from opik.evaluation.metrics import BaseMetric, AnswerRelevance
from opik.evaluation.metrics import score_result


class StockAnswerRelevance(BaseMetric):
    """
    주식 질문에 대한 에이전트 응답의 관련성을 평가합니다.
    주식 질문에 관련 있는 답변인지 확인합니다.
    """

    def __init__(self, name: str = "stock_answer_relevance"):
        self.name = name
        self._inner = AnswerRelevance(require_context=False)

    def score(self, output: str, **kwargs) -> score_result.ScoreResult:
        input = kwargs.get("input", "")
        return self._inner.score(input=input, output=output)
