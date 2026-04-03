from __future__ import annotations

from opik.evaluation.metrics import BaseMetric, Hallucination
from opik.evaluation.metrics import score_result


class SecGroundedness(BaseMetric):
    """
    메인 에이전트의 SEC 공시 기반 답변이 검색된 청크에 근거하는지 평가합니다.

    search_sec_filing 도구가 호출된 항목에만 적용됩니다.
    context(검색된 청크)가 없으면 None을 반환하여 해당 항목을 skip합니다.

    점수 방향: 0.0 = 근거 있음(좋음), 1.0 = 근거 없음(나쁨)
    """

    def __init__(self, name: str = "sec_groundedness"):
        self.name = name
        self._inner = Hallucination()

    def score(self, output: str, **kwargs) -> score_result.ScoreResult:
        context = kwargs.get("context", [])
        if not context:
            return score_result.ScoreResult(
                name=self.name,
                value=0.0,
                reason="search_sec_filing not called — skipped",
                scoring_failed=True,
            )
        input_ = kwargs.get("input", "")
        return self._inner.score(input=input_, output=output, context=context)
