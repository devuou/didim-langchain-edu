"""SEC 10-K 공시 문서 검색 서브 에이전트.

LangGraph StateGraph로 BM25 + kNN 벡터 검색을 병렬(fan-out) 실행하고
결과를 병합 → 리랭킹 후 메인 에이전트에 반환한다.
agent-sample의 search_agent.py 패턴을 따른다.

그래프 흐름:
    START
      ├──→ bm25_search     (ES match 쿼리, ticker 필터)
      └──→ vector_search   (ES kNN, 임베딩 생성 후 검색)
                ↓ (fan-in: 두 노드 완료 후 진입)
           merge_results   (_id 기준 중복 제거, 높은 score 우선)
                ↓
              rerank       (ES Inference API 리랭킹, 실패 시 score 내림차순 fallback)
                ↓
               END
"""

from __future__ import annotations

from typing_extensions import TypedDict

from app.agents.tools._rag_common import (
    get_es_client,
    embed_query,
    rerank_hits,
    format_hits,
)
from app.core.config import settings


# ─── State ───────────────────────────────────────────────────────────────────

class SecSearchState(TypedDict):
    """서브 에이전트 그래프 전체에서 공유하는 상태 스키마.

    각 노드는 상태에서 필요한 키만 읽고, 반환하는 dict의 키만 상태에 업데이트한다.
    """
    query: str               # 검색 쿼리
    ticker: str              # 대상 종목 (AAPL, MSFT, TSLA, NVDA)
    bm25_hits: list[dict]    # BM25 키워드 검색 결과
    vector_hits: list[dict]  # kNN 벡터 검색 결과
    merged_hits: list[dict]  # 병합 + 중복 제거된 결과
    result: str              # 최종 포맷팅 문자열 (LLM 컨텍스트용)


# ES 인덱스 이름: ingest_10k.py의 build_index_name()과 동일 규칙으로 생성
_INDEX_NAME = f"{settings.ES_INDEX_PREFIX}-10k-docs"
_TOP_K = 20       # BM25 / kNN 각각 최대 20개 검색
_FINAL_TOP_N = 5  # 리랭킹 후 LLM에 전달할 최종 청크 수


# ─── 노드 함수 ────────────────────────────────────────────────────────────────

def bm25_search(state: SecSearchState) -> dict:
    """ES text 필드에 BM25 match 쿼리를 실행한다.

    - must: query와 text 필드 간 BM25 유사도 계산
    - filter: ticker로 종목 한정 (점수에 영향 없이 필터링)
    """
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
    """질문을 임베딩하여 kNN 벡터 검색을 실행한다.

    - query_vector: 질문을 text-embedding-3-small으로 변환한 1536차원 벡터
    - k: 반환할 최근접 이웃 수
    - num_candidates: kNN 후보 탐색 범위 (클수록 정확하지만 느림)
    - filter: ticker로 종목 한정
    """
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
    """BM25 + 벡터 검색 결과를 병합하고 _id 기준으로 중복을 제거한다.

    동일 청크(_id)가 두 검색에서 모두 나올 경우 높은 score를 가진 것을 채택한다.
    """
    seen: dict[str, dict] = {}
    for hit in state.get("bm25_hits", []) + state.get("vector_hits", []):
        hit_id = hit["_id"]
        if hit_id not in seen or hit["_score"] > seen[hit_id]["_score"]:
            seen[hit_id] = hit
    return {"merged_hits": list(seen.values())}


def merge_results(state: SecSearchState) -> dict:
    return _merge_results_fn(state)


def _rerank_fn(state: dict) -> dict:
    """ES Inference API로 리랭킹한다. 실패 시 score 내림차순 정렬로 fallback한다.

    ES_RERANKER_INFERENCE_ID가 설정된 경우 ES 호스팅 rerank 모델을 호출한다.
    설정이 없거나 호출에 실패하면 기존 ES score 기준으로 내림차순 정렬한다.
    """
    hits = state.get("merged_hits", [])
    query = state.get("query", "")

    reranked = rerank_hits(query, hits)
    if reranked is None:
        # fallback: ES score 기준 내림차순 정렬
        hits = sorted(hits, key=lambda h: h["_score"], reverse=True)
    else:
        hits = reranked

    # 상위 N개만 LLM에 전달 (컨텍스트 길이 제한)
    top_hits = hits[:_FINAL_TOP_N]
    return {"merged_hits": top_hits, "result": format_hits(top_hits)}


def rerank(state: SecSearchState) -> dict:
    return _rerank_fn(state)


# ─── 그래프 조립 ──────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool


def _build_graph():
    """StateGraph를 조립하고 컴파일한 그래프를 반환한다.

    fan-out: START → bm25_search, START → vector_search (두 노드 병렬 실행)
    fan-in:  bm25_search → merge_results, vector_search → merge_results
             (두 노드가 모두 완료된 후 merge_results 진입)
    """
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


# 모듈 로드 시 그래프를 1회 컴파일 — 이후 search_sec_filing 호출마다 재사용
_sec_search_graph = _build_graph()


# ─── @tool 래핑 ───────────────────────────────────────────────────────────────

@tool
def search_sec_filing(ticker: str, query: str) -> str:
    """기업 공시(10-K 사업보고서)에서 질문과 관련된 내용을 검색합니다.
    사업 구조, 리스크 요인, 경영 성과 분석(MD&A) 등 정성적 정보 조회에 사용합니다.
    지원 종목: AAPL, MSFT, TSLA, NVDA (이외 종목 불가)
    보유 데이터: 최신 연간 보고서(10-K) 기준

    subagent-as-tool 패턴:
    StateGraph 전체를 @tool로 래핑하면 메인 에이전트 입장에서 일반 도구와 동일하게 사용된다.
    """
    result = _sec_search_graph.invoke({"ticker": ticker.upper(), "query": query})
    return result["result"]
