"""SEC 10-K 공시 문서 검색 서브 에이전트.

LangGraph StateGraph로 BM25 + kNN 벡터 검색을 병렬(fan-out) 실행하고
결과를 병합 → 리랭킹 후 메인 에이전트에 반환한다.
agent-sample의 search_agent.py 패턴을 따른다.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from app.agents.tools._rag_common import (
    get_es_client,
    get_reranker as _get_reranker,
    embed_query,
    format_hits,
)
from app.core.config import settings


# ─── State ───────────────────────────────────────────────────────────────────

class SecSearchState(TypedDict):
    query: str               # 검색 쿼리
    ticker: str              # 대상 종목 (AAPL, MSFT, TSLA, NVDA)
    bm25_hits: list[dict]    # BM25 키워드 검색 결과
    vector_hits: list[dict]  # kNN 벡터 검색 결과
    merged_hits: list[dict]  # 병합 + 중복 제거된 결과
    result: str              # 최종 포맷팅 문자열 (LLM 컨텍스트용)


_INDEX_NAME = f"{settings.ES_INDEX_PREFIX}-10k-docs"
_TOP_K = 20
_FINAL_TOP_N = 5


# ─── 노드 함수 ────────────────────────────────────────────────────────────────

def bm25_search(state: SecSearchState) -> dict:
    """ES text 필드에 BM25 match 쿼리를 실행한다."""
    es = get_es_client()
    body = {
        "size": _TOP_K,
        "query": {
            "bool": {
                "must": [{"match": {"text": state["query"]}}],
                "filter": [{"term": {"ticker": state["ticker"]}}],
            }
        },
    }
    resp = es.search(index=_INDEX_NAME, body=body)
    return {"bm25_hits": resp["hits"]["hits"]}


def vector_search(state: SecSearchState) -> dict:
    """질문을 임베딩하여 kNN 벡터 검색을 실행한다."""
    es = get_es_client()
    query_vector = embed_query(state["query"])
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": _TOP_K,
            "num_candidates": 100,
            "filter": [{"term": {"ticker": state["ticker"]}}],
        }
    }
    resp = es.search(index=_INDEX_NAME, body=body)
    return {"vector_hits": resp["hits"]["hits"]}


def _merge_results_fn(state: dict) -> dict:
    """BM25 + 벡터 검색 결과를 병합하고 _id 기준으로 중복을 제거한다."""
    seen: dict[str, dict] = {}
    for hit in state.get("bm25_hits", []) + state.get("vector_hits", []):
        hit_id = hit["_id"]
        if hit_id not in seen or hit["_score"] > seen[hit_id]["_score"]:
            seen[hit_id] = hit
    return {"merged_hits": list(seen.values())}


def merge_results(state: SecSearchState) -> dict:
    return _merge_results_fn(state)


def _rerank_fn(state: dict) -> dict:
    """cross-encoder로 리랭킹한다. 리랭커 없으면 score 내림차순 정렬 후 반환한다."""
    hits = state.get("merged_hits", [])
    query = state.get("query", "")
    reranker = _get_reranker()

    if reranker is None:
        hits = sorted(hits, key=lambda h: h["_score"], reverse=True)
    else:
        pairs = [[query, h["_source"]["text"]] for h in hits]
        scores = reranker.predict(pairs)
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]

    top_hits = hits[:_FINAL_TOP_N]
    return {"merged_hits": top_hits, "result": format_hits(top_hits)}


def rerank(state: SecSearchState) -> dict:
    return _rerank_fn(state)


# ─── 그래프 조립 ──────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool


def _build_graph():
    builder = StateGraph(SecSearchState)

    builder.add_node("bm25_search", bm25_search)
    builder.add_node("vector_search", vector_search)
    builder.add_node("merge_results", merge_results)
    builder.add_node("rerank", rerank)

    # fan-out: START에서 두 노드를 병렬 실행
    builder.add_edge(START, "bm25_search")
    builder.add_edge(START, "vector_search")

    # fan-in: 두 노드 완료 후 merge 실행
    builder.add_edge("bm25_search", "merge_results")
    builder.add_edge("vector_search", "merge_results")

    builder.add_edge("merge_results", "rerank")
    builder.add_edge("rerank", END)

    return builder.compile()


_sec_search_graph = _build_graph()


# ─── @tool 래핑 ───────────────────────────────────────────────────────────────

@tool
def search_sec_filing(ticker: str, query: str) -> str:
    """기업 공시(10-K 사업보고서)에서 질문과 관련된 내용을 검색합니다.
    사업 구조, 리스크 요인, 경영 성과 분석(MD&A) 등 정성적 정보 조회에 사용합니다.
    지원 종목: AAPL, MSFT, TSLA, NVDA (이외 종목 불가)
    보유 데이터: 최신 연간 보고서(10-K) 기준
    """
    result = _sec_search_graph.invoke({"ticker": ticker.upper(), "query": query})
    return result["result"]
