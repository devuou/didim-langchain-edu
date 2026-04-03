"""
주식 에이전트 Opik 평가 실행 스크립트.

사용법:
    python evaluation/run_eval.py --level L1                              # 빠른 검증 (5건)
    python evaluation/run_eval.py --level L2                              # 전체 평가 (10건)
    python evaluation/run_eval.py --level L1 --nb-samples 3              # 샘플 수 오버라이드
    python evaluation/run_eval.py --level L2 --experiment-name my-eval   # 실험 이름 지정
"""

from __future__ import annotations

import argparse
import asyncio
import os
import uuid

# ── .env 로드 및 Opik 환경변수 설정 ─────────────────────────────────────────
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
from opik import track
from opik.evaluation import evaluate

from evaluation.metrics.stock_hallucination import StockHallucination
from evaluation.metrics.stock_answer_relevance import StockAnswerRelevance
from evaluation.metrics.stock_task_completion import StockTaskCompletion
from evaluation.metrics.sec_groundedness import SecGroundedness


# ---------------------------------------------------------------------------
# 레벨별 평가 설정
# ---------------------------------------------------------------------------

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

METRIC_MAP = {
    "hallucination": StockHallucination,
    "relevance": StockAnswerRelevance,
    "task_completion": StockTaskCompletion,
    "groundedness": SecGroundedness,
}

# 평가용 데이터셋 — evaluation/data/dataset.json 에서 로드
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.json")

def _load_dataset_items() -> list[dict]:
    import json
    with open(_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)

DATASET_ITEMS = _load_dataset_items()


# ---------------------------------------------------------------------------
# 에이전트 초기화 (모듈 레벨)
# ---------------------------------------------------------------------------

def _build_agent():
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver
    from pydantic import SecretStr
    from app.core.config import settings
    from app.agents.stock_agent import create_stock_agent

    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
    )
    return create_stock_agent(model=model, checkpointer=MemorySaver())


_agent = _build_agent()


# ---------------------------------------------------------------------------
# 에이전트 실행 (동기 래퍼)
# ---------------------------------------------------------------------------

@track  # Opik 데코레이터로 이 함수가 호출될 때마다 Opik 서버에 자동으로 기록
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

    return asyncio.run(_stream())  # 비동기 에이전트를 동기 함수로 감싼 래퍼


# ---------------------------------------------------------------------------
# Opik evaluation task (Opik evaluate()의 진입점 -> run_stock_agent()를 호출함)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 데이터셋 생성 또는 조회
# ---------------------------------------------------------------------------

def get_or_create_dataset(client: opik.Opik, name: str) -> opik.Dataset:
    """데이터셋이 없으면 생성하고 항목을 삽입합니다."""
    try:
        dataset = client.get_dataset(name=name)
    except Exception:
        dataset = client.create_dataset(name=name, description="주식 에이전트를 평가하기 위한 기본 DataSet")
        dataset.insert(DATASET_ITEMS)
    return dataset


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level",
        choices=["L1", "L2"],
        default="L1",
        help="평가 레벨 (L1: 5건, L2: 10건)",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--dataset-name", default="hryu-stock-dataset")
    parser.add_argument(
        "--nb-samples",
        type=int,
        default=None,
        help="샘플 수 오버라이드 (미지정 시 레벨 기본값)",
    )
    args = parser.parse_args()

    config = LEVEL_CONFIG[args.level]
    nb_samples = args.nb_samples or config["nb_samples"]
    experiment_name = args.experiment_name or f"hryu-stock-agent-{args.level}-eval"

    client = opik.Opik()
    dataset = get_or_create_dataset(client, name=args.dataset_name)

    metrics = [METRIC_MAP[m]() for m in config["metrics"]]

    evaluation = evaluate(
        experiment_name=experiment_name,
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=metrics,
        nb_samples=nb_samples,
        experiment_config={
            "level": args.level,
            "model": os.getenv("OPENAI_MODEL", "unknown"),
            "dataset": args.dataset_name,
            "nb_samples": nb_samples,
        },
        project_name=os.getenv("OPIK_PROJECT_NAME", "stock-agent-evaluation"),
    )

    scores = evaluation.aggregate_evaluation_scores()
    print(f"\n=== 평가 결과 ({args.level}, {nb_samples}건) ===")
    for metric_name, stats in scores.aggregated_scores.items():
        print(f"  {metric_name}: {stats}")


if __name__ == "__main__":
    main()
