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
