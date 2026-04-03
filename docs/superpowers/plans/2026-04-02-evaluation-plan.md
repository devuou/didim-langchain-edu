# 평가 시스템 확장 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SEC 공시 검색 서브 에이전트를 포함한 두 레이어 평가 시스템을 Opik 기반으로 구축한다.

**Architecture:** 서브 에이전트 단독 평가(`run_eval_sec.py`)와 메인 에이전트 엔드투엔드 평가(`run_eval.py` 수정)를 분리한다. 메인 에이전트 평가는 `search_sec_filing` tool output을 context로 캡처하여 Hallucination 메트릭이 SEC 답변에 대해 제대로 동작하도록 한다.

**Tech Stack:** Python, Opik (`evaluate`, `GEval`, `Hallucination`, `BaseMetric`), LangGraph `StateGraph`, pytest

---

## 파일 맵

| 파일 | 작업 |
|---|---|
| `evaluation/data/sec_dataset.json` | 신규 생성 — 서브 에이전트 전용 10건 |
| `evaluation/metrics/sec_retrieval_relevance.py` | 신규 생성 — GEval 기반 검색 관련성 메트릭 |
| `evaluation/metrics/sec_groundedness.py` | 신규 생성 — Hallucination+context 기반 근거 충실도 메트릭 |
| `evaluation/run_eval_sec.py` | 신규 생성 — 서브 에이전트 단독 평가 진입점 |
| `evaluation/data/dataset.json` | 수정 — SEC 질문 6건 추가 (15→21건) |
| `evaluation/run_eval.py` | 수정 — context 캡처, SecGroundedness 레벨 편입 |
| `tests/test_evaluation_metrics.py` | 신규 생성 — 새 메트릭 2종 단위 테스트 |
| `tests/test_run_eval_sec.py` | 신규 생성 — sec_eval_task 단위 테스트 |

---

## Task 1: sec_dataset.json 작성

**Files:**
- Create: `evaluation/data/sec_dataset.json`

- [ ] **Step 1: 파일 생성**

`evaluation/data/sec_dataset.json`:
```json
[
  {
    "ticker": "AAPL",
    "query": "애플의 주요 제품과 서비스 부문은?",
    "section": "item1",
    "expected_output": "iPhone, Mac, iPad, Wearables, Services 등 주요 사업 부문이 언급된 검색 결과"
  },
  {
    "ticker": "MSFT",
    "query": "마이크로소프트가 공시한 주요 사업 리스크는?",
    "section": "item1a",
    "expected_output": "경쟁, 규제, 사이버 보안 등 리스크 요인이 포함된 검색 결과"
  },
  {
    "ticker": "AAPL",
    "query": "애플 최근 사업연도 매출 성과는?",
    "section": "item7",
    "expected_output": "매출, 수익 등 재무 성과에 대한 경영진 분석 내용이 포함된 검색 결과"
  },
  {
    "ticker": "AMZN",
    "query": "아마존 리스크 알려줘",
    "section": "unsupported",
    "expected_output": "관련 공시 정보를 찾을 수 없습니다."
  },
  {
    "ticker": "NVDA",
    "query": "NVDA의 핵심 사업 영역은?",
    "section": "item1",
    "expected_output": "데이터센터, 게이밍 GPU, AI 칩 등 핵심 사업이 포함된 검색 결과"
  },
  {
    "ticker": "TSLA",
    "query": "테슬라의 공급망 리스크는?",
    "section": "item1a",
    "expected_output": "배터리, 반도체, 원자재 공급망 관련 리스크가 포함된 검색 결과"
  },
  {
    "ticker": "NVDA",
    "query": "NVDA의 AI 사업 성과는?",
    "section": "item7",
    "expected_output": "AI, 데이터센터 관련 매출 성장 내용이 포함된 검색 결과"
  },
  {
    "ticker": "GOOGL",
    "query": "구글 사업구조 알려줘",
    "section": "unsupported",
    "expected_output": "관련 공시 정보를 찾을 수 없습니다."
  },
  {
    "ticker": "MSFT",
    "query": "마이크로소프트 클라우드 전략과 관련 리스크는?",
    "section": "item1",
    "expected_output": "Azure 클라우드 전략과 관련 경쟁, 규제 리스크가 모두 포함된 검색 결과"
  },
  {
    "ticker": "TSLA",
    "query": "테슬라 전기차 시장 전략과 경쟁 리스크는?",
    "section": "item1",
    "expected_output": "EV 시장 전략과 경쟁사 관련 리스크가 모두 포함된 검색 결과"
  }
]
```

> **참고:** 첫 4건(AAPL item1, MSFT item1a, AAPL item7, AMZN unsupported)이 L1 샘플. 섹션 유형별 1건씩 배치.

- [ ] **Step 2: 파일이 올바르게 로드되는지 확인**

```bash
cd /Users/n-hryu/Dev/workspace/AIAgent/agent
uv run python -c "import json; items = json.load(open('evaluation/data/sec_dataset.json')); print(len(items), 'items'); print([i['ticker'] for i in items])"
```

Expected:
```
10 items
['AAPL', 'MSFT', 'AAPL', 'AMZN', 'NVDA', 'TSLA', 'NVDA', 'GOOGL', 'MSFT', 'TSLA']
```

- [ ] **Step 3: Commit**

```bash
git add evaluation/data/sec_dataset.json
git commit -m "feat(eval): add sec_dataset.json for sub-agent evaluation (10 items)"
```

---

## Task 2: SecRetrievalRelevance 메트릭

**Files:**
- Create: `evaluation/metrics/sec_retrieval_relevance.py`
- Create: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_evaluation_metrics.py`:
```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
uv run pytest tests/test_evaluation_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'evaluation.metrics.sec_retrieval_relevance'`

- [ ] **Step 3: 메트릭 구현**

`evaluation/metrics/sec_retrieval_relevance.py`:
```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
uv run pytest tests/test_evaluation_metrics.py::test_sec_retrieval_relevance_payload_format tests/test_evaluation_metrics.py::test_sec_retrieval_relevance_returns_score_result -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics/sec_retrieval_relevance.py tests/test_evaluation_metrics.py
git commit -m "feat(eval): add SecRetrievalRelevance metric with GEval"
```

---

## Task 3: SecGroundedness 메트릭

**Files:**
- Create: `evaluation/metrics/sec_groundedness.py`
- Modify: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: 실패하는 테스트 추가**

`tests/test_evaluation_metrics.py` 끝에 추가:
```python

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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
uv run pytest tests/test_evaluation_metrics.py -k "groundedness" -v
```

Expected: `ModuleNotFoundError: No module named 'evaluation.metrics.sec_groundedness'`

- [ ] **Step 3: 메트릭 구현**

`evaluation/metrics/sec_groundedness.py`:
```python
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

    def score(self, output: str, **kwargs) -> score_result.ScoreResult | None:
        context = kwargs.get("context", [])
        if not context:
            return None
        input_ = kwargs.get("input", "")
        return self._inner.score(input=input_, output=output, context=context)
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
uv run pytest tests/test_evaluation_metrics.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics/sec_groundedness.py tests/test_evaluation_metrics.py
git commit -m "feat(eval): add SecGroundedness metric (Hallucination with context)"
```

---

## Task 4: run_eval_sec.py 구현

**Files:**
- Create: `evaluation/run_eval_sec.py`
- Create: `tests/test_run_eval_sec.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_run_eval_sec.py`:
```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
uv run pytest tests/test_run_eval_sec.py -v
```

Expected: `ModuleNotFoundError: No module named 'evaluation.run_eval_sec'`

- [ ] **Step 3: run_eval_sec.py 구현**

`evaluation/run_eval_sec.py`:
```python
"""
SEC 공시 검색 서브 에이전트 Opik 평가 실행 스크립트.

사용법:
    python evaluation/run_eval_sec.py --level L1   # 빠른 검증 (4건)
    python evaluation/run_eval_sec.py --level L2   # 전체 평가 (10건)
    python evaluation/run_eval_sec.py --level L1 --nb-samples 2
    python evaluation/run_eval_sec.py --level L2 --experiment-name my-sec-eval
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

_OPIK_ENV_MAP = {
    "OPIK_URL_OVERRIDE": os.environ.get("OPIK__URL_OVERRIDE"),
    "OPIK_PROJECT_NAME": os.environ.get("OPIK__PROJECT"),
    "OPIK_WORKSPACE":    os.environ.get("OPIK__WORKSPACE"),
}
for _k, _v in _OPIK_ENV_MAP.items():
    if _v:
        os.environ.setdefault(_k, _v)

import opik
from opik.evaluation import evaluate

from evaluation.metrics.sec_retrieval_relevance import SecRetrievalRelevance


# ---------------------------------------------------------------------------
# 레벨별 평가 설정
# ---------------------------------------------------------------------------

LEVEL_CONFIG = {
    "L1": {"nb_samples": 4},   # 섹션 유형별 1건 (item1, item1a, item7, unsupported)
    "L2": {"nb_samples": 10},  # 전체
}

_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "sec_dataset.json")


def _load_dataset_items() -> list[dict]:
    import json
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


DATASET_ITEMS = _load_dataset_items()


# ---------------------------------------------------------------------------
# 서브 에이전트 초기화
# ---------------------------------------------------------------------------

def _build_graph():
    from app.agents.sec_search_agent import _sec_search_graph
    return _sec_search_graph


_graph = _build_graph()


# ---------------------------------------------------------------------------
# 평가 task
# ---------------------------------------------------------------------------

def sec_eval_task(dataset_item: dict) -> dict:
    """Opik evaluate()가 각 데이터셋 항목에 대해 호출하는 task 함수."""
    result = _graph.invoke({
        "ticker": dataset_item["ticker"],
        "query": dataset_item["query"],
    })
    return {
        "input": dataset_item["query"],
        "output": result["result"],
        "expected_output": dataset_item.get("expected_output", ""),
    }


# ---------------------------------------------------------------------------
# 데이터셋 생성 또는 조회
# ---------------------------------------------------------------------------

def get_or_create_dataset(client: opik.Opik, name: str) -> opik.Dataset:
    try:
        dataset = client.get_dataset(name=name)
    except Exception:
        dataset = client.create_dataset(
            name=name,
            description="SEC 공시 검색 서브 에이전트 평가 데이터셋",
        )
        dataset.insert(DATASET_ITEMS)
    return dataset


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["L1", "L2"], default="L1")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--dataset-name", default="hryu-sec-dataset")
    parser.add_argument("--nb-samples", type=int, default=None)
    args = parser.parse_args()

    config = LEVEL_CONFIG[args.level]
    nb_samples = args.nb_samples or config["nb_samples"]
    experiment_name = args.experiment_name or f"hryu-sec-agent-{args.level}-eval"

    client = opik.Opik()
    dataset = get_or_create_dataset(client, name=args.dataset_name)

    evaluation = evaluate(
        experiment_name=experiment_name,
        dataset=dataset,
        task=sec_eval_task,
        scoring_metrics=[SecRetrievalRelevance()],
        nb_samples=nb_samples,
        experiment_config={
            "level": args.level,
            "dataset": args.dataset_name,
            "nb_samples": nb_samples,
        },
        project_name=os.getenv("OPIK_PROJECT_NAME", "stock-agent-evaluation"),
    )

    scores = evaluation.aggregate_evaluation_scores()
    print(f"\n=== SEC 서브 에이전트 평가 결과 ({args.level}, {nb_samples}건) ===")
    for metric_name, stats in scores.aggregated_scores.items():
        print(f"  {metric_name}: {stats}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
uv run pytest tests/test_run_eval_sec.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add evaluation/run_eval_sec.py tests/test_run_eval_sec.py
git commit -m "feat(eval): add run_eval_sec.py for sub-agent standalone evaluation"
```

---

## Task 5: dataset.json SEC 질문 6건 추가

**Files:**
- Modify: `evaluation/data/dataset.json`

- [ ] **Step 1: 기존 파일 끝에 6건 추가**

`evaluation/data/dataset.json`의 마지막 `]` 앞에 아래 6건을 추가:
```json
  ,
  {
    "input": "AAPL 10-K에서 밝힌 주요 사업 리스크는?",
    "expected_output": "공급망, 경쟁, 규제 등 리스크 요인 내용이 포함된 답변. 10-K 기준임을 명시."
  },
  {
    "input": "MSFT 공시에서 밝힌 클라우드(Azure) 관련 내용은?",
    "expected_output": "Azure 클라우드 사업 전략, 성장, 또는 관련 리스크 내용이 포함된 답변. 10-K 기준임을 명시."
  },
  {
    "input": "TSLA 10-K에서 경영진이 분석한 재무 성과는?",
    "expected_output": "테슬라 매출, 수익 등 MD&A 섹션의 경영 성과 분석 내용이 포함된 답변. 10-K 기준임을 명시."
  },
  {
    "input": "NVDA 현재 주가랑 공시에서 밝힌 AI 전략 알려줘",
    "expected_output": "NVDA 현재 주가(수치 포함)와 10-K에 기재된 AI/데이터센터 전략 내용이 모두 포함된 답변"
  },
  {
    "input": "AAPL 시가총액이랑 10-K에서 밝힌 주요 리스크 함께 알려줘",
    "expected_output": "AAPL 시가총액(수치 포함)과 10-K 리스크 요인 내용이 모두 포함된 답변"
  },
  {
    "input": "삼성전자 10-K 공시 알려줘",
    "expected_output": "지원하지 않는 종목(AAPL, MSFT, TSLA, NVDA만 지원)임을 안내하는 답변"
  }
```

- [ ] **Step 2: 총 21건인지 확인**

```bash
uv run python -c "import json; items = json.load(open('evaluation/data/dataset.json')); print(len(items), 'items')"
```

Expected: `21 items`

- [ ] **Step 3: Commit**

```bash
git add evaluation/data/dataset.json
git commit -m "feat(eval): add 6 SEC-related items to dataset.json (15→21)"
```

---

## Task 6: run_eval.py 수정 — context 캡처 + SecGroundedness 편입

**Files:**
- Modify: `evaluation/run_eval.py`

- [ ] **Step 1: import 추가 및 METRIC_MAP 수정**

`evaluation/run_eval.py` 상단 import 블록에 추가:
```python
from evaluation.metrics.sec_groundedness import SecGroundedness
```

`METRIC_MAP` dict에 `"groundedness"` 추가:
```python
METRIC_MAP = {
    "hallucination": StockHallucination,
    "relevance": StockAnswerRelevance,
    "task_completion": StockTaskCompletion,
    "groundedness": SecGroundedness,
}
```

`LEVEL_CONFIG`의 L2 nb_samples와 metrics 수정:
```python
LEVEL_CONFIG = {
    "L1": {
        "nb_samples": 5,
        "metrics": ["hallucination", "relevance"],
    },
    "L2": {
        "nb_samples": 21,
        "metrics": ["hallucination", "relevance", "task_completion", "groundedness"],
    },
}
```

- [ ] **Step 2: run_stock_agent 반환값을 tuple로 변경**

기존:
```python
@track
def run_stock_agent(question: str, thread_id: str) -> str:
    """주식 에이전트를 동기적으로 실행하고 최종 답변을 반환합니다."""

    async def _stream() -> str:
        final = ""
        from langchain_core.messages import HumanMessage

        async for chunk in _agent.astream(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            for step, event in chunk.items():
                if step != "model":
                    continue
                messages = event.get("messages", [])
                if not messages:
                    continue
                tool_calls = messages[0].tool_calls
                for tool in tool_calls:
                    if tool.get("name") == "ChatResponse":
                        final = tool.get("args", {}).get("content", "")
        return final

    return asyncio.run(_stream())
```

변경 후:
```python
@track
def run_stock_agent(question: str, thread_id: str) -> tuple[str, list[str]]:
    """주식 에이전트를 동기적으로 실행하고 (최종 답변, SEC 도구 출력 목록)을 반환합니다.

    tool_outputs: search_sec_filing 호출 시 반환된 텍스트 목록.
    SecGroundedness 메트릭의 context로 사용된다.
    """

    async def _stream() -> tuple[str, list[str]]:
        final = ""
        tool_outputs: list[str] = []
        from langchain_core.messages import HumanMessage

        async for chunk in _agent.astream(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            for step, event in chunk.items():
                if step == "tools":
                    messages = event.get("messages", [])
                    for msg in messages:
                        if getattr(msg, "name", None) == "search_sec_filing":
                            tool_outputs.append(msg.content)
                if step != "model":
                    continue
                messages = event.get("messages", [])
                if not messages:
                    continue
                tool_calls = messages[0].tool_calls
                for tool in tool_calls:
                    if tool.get("name") == "ChatResponse":
                        final = tool.get("args", {}).get("content", "")
        return final, tool_outputs

    return asyncio.run(_stream())
```

- [ ] **Step 3: evaluation_task 수정**

기존:
```python
def evaluation_task(dataset_item: dict) -> dict:
    """Opik evaluate()가 각 데이터셋 항목에 대해 호출하는 task 함수."""
    question = dataset_item["input"]
    thread_id = f"eval-{dataset_item.get('id', uuid.uuid4())}"
    output = run_stock_agent(question, thread_id)
    return {
        "input": question,
        "output": output,
    }
```

변경 후:
```python
def evaluation_task(dataset_item: dict) -> dict:
    """Opik evaluate()가 각 데이터셋 항목에 대해 호출하는 task 함수."""
    question = dataset_item["input"]
    thread_id = f"eval-{dataset_item.get('id', uuid.uuid4())}"
    output, tool_outputs = run_stock_agent(question, thread_id)
    return {
        "input": question,
        "output": output,
        "expected_output": dataset_item.get("expected_output", ""),
        "context": tool_outputs,  # SecGroundedness가 사용. 비어있으면 해당 항목 skip.
    }
```

- [ ] **Step 4: 전체 테스트 실행 — 기존 테스트 회귀 확인**

```bash
uv run pytest tests/ -v --ignore=tests/test_main.py
```

Expected: 기존 테스트 포함 전체 통과. `test_evaluation_metrics.py`와 `test_run_eval_sec.py`도 포함.

- [ ] **Step 5: Commit**

```bash
git add evaluation/run_eval.py
git commit -m "feat(eval): capture search_sec_filing context and add SecGroundedness to L2"
```

---

## 최종 동작 확인

- [ ] **전체 테스트 통과 확인**

```bash
uv run pytest tests/ -v
```

Expected: 모든 테스트 통과.

- [ ] **파일 구조 확인**

```bash
ls evaluation/metrics/
ls evaluation/data/
ls evaluation/run_eval*.py 2>/dev/null || ls evaluation/
```

Expected:
```
evaluation/metrics/: sec_groundedness.py  sec_retrieval_relevance.py  stock_answer_relevance.py  stock_hallucination.py  stock_task_completion.py
evaluation/data/:    dataset.json  sec_dataset.json
evaluation/:         run_eval.py  run_eval_sec.py  ...
```

- [ ] **실행 가능 여부 확인 (dry-run — 실제 API 호출 없음)**

```bash
uv run python -c "
from evaluation.metrics.sec_retrieval_relevance import SecRetrievalRelevance
from evaluation.metrics.sec_groundedness import SecGroundedness
from evaluation.run_eval_sec import LEVEL_CONFIG, DATASET_ITEMS
print('SecRetrievalRelevance:', SecRetrievalRelevance().name)
print('SecGroundedness:', SecGroundedness().name)
print('sec_dataset items:', len(DATASET_ITEMS))
print('L1 samples:', LEVEL_CONFIG['L1']['nb_samples'])
print('L2 samples:', LEVEL_CONFIG['L2']['nb_samples'])
"
```

Expected:
```
SecRetrievalRelevance: sec_retrieval_relevance
SecGroundedness: sec_groundedness
sec_dataset items: 10
L1 samples: 4
L2 samples: 10
```
