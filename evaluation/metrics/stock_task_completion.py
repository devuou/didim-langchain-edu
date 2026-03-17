from opik.evaluation.metrics import BaseMetric, GEval
from opik.evaluation.metrics import score_result


class StockTaskCompletion(BaseMetric):
    """
    주식 에이전트가 요청한 작업을 적절히 처리했는지 평가합니다.
    GEval을 사용하며 expected_output을 기준으로 판단합니다.

    지원 기능 내 질문: 실제 데이터(수치, 뉴스 등)를 포함한 답변이 정답
    지원 기능 외 질문: 정중한 거절 + 제공 가능 기능 안내가 정답

    GEval은 output 하나만 받으므로, score() 내부에서
    QUESTION / EXPECTED_OUTPUT / OUTPUT 을 하나의 문자열로 패킹하여 전달합니다.
    """

    def __init__(self, name: str = "stock_task_completion"):
        self.name = name
        self._inner = GEval(
            task_introduction=(
                "주식 분석 에이전트의 답변을 평가합니다. "
                "이 에이전트는 현재 주가/등락률, 시가총액/PER/업종, 최근 뉴스, "
                "히스토리컬 OHLCV 데이터만 제공하며 그 외 질문은 정중히 거절합니다."
            ),
            evaluation_criteria=(
                "OUTPUT이 EXPECTED_OUTPUT에 명시된 핵심 정보를 충족하는지 판단하세요. "
                "지원 기능 내 질문은 관련 수치나 정보가 포함되어야 높은 점수를 줍니다. "
                "지원 기능 외 질문(배당, 전망, 투자 추천 등)에 대해 "
                "정중하게 거절하고 제공 가능한 기능을 안내한 경우 높은 점수를 줍니다. "
                "범위 밖 질문에 억지로 답변하거나 잘못된 정보를 제공한 경우 낮은 점수를 줍니다."
            ),
            name=name,
        )

    def score(self, output: str, **kwargs) -> score_result.ScoreResult:
        input = kwargs.get("input", "")
        expected_output = kwargs.get("expected_output", "")

        payload = f"QUESTION: {input}\nEXPECTED_OUTPUT: {expected_output}\nOUTPUT: {output}"
        return self._inner.score(output=payload)
