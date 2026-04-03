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

from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

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
    seen_ids: list[str]      # 이미 반환된 chunk ID 목록 (cross-invocation 중복 제거용)


# ES 인덱스 이름: ingest_10k.py의 build_index_name()과 동일 규칙으로 생성
_INDEX_NAME = f"{settings.ES_INDEX_PREFIX}-10k-docs"
_TOP_K = 20            # BM25 / kNN 각각 최대 20개 검색
_NUM_CANDIDATES = 100  # kNN 후보 탐색 범위 (클수록 정확하지만 느림, 최소 _TOP_K 이상)
_FINAL_TOP_N = 5       # 리랭킹 후 LLM에 전달할 최종 청크 수


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
    - 임베딩 또는 ES 호출 실패 시 빈 hits를 반환하여 BM25 결과만으로 계속 진행한다.
    """
    try:
        es = get_es_client()
        query_vector = embed_query(state["query"])
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": _TOP_K,
                "num_candidates": _NUM_CANDIDATES,
                "filter": [{"term": {"ticker": state["ticker"]}}],
            }
        }
        resp = es.search(index=_INDEX_NAME, body=body)
        return {"vector_hits": resp["hits"]["hits"]}
    except Exception:
        return {"vector_hits": []}


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

    seen_ids에 포함된 청크를 먼저 제외한 뒤 리랭킹한다.
    모든 청크가 seen_ids에 포함된 경우(전량 소진) 필터 없이 전체 후보를 사용한다.
    선택된 청크 ID는 seen_ids에 누적되어 반환된다.
    """
    hits = state.get("merged_hits", [])
    seen_ids = set(state.get("seen_ids", []))
    query = state.get("query", "")

    # 이미 반환된 청크 제외 — 모두 소진된 경우 전체 사용(fallback)
    unseen = [h for h in hits if h["_id"] not in seen_ids]
    candidates = unseen if unseen else hits

    reranked = rerank_hits(query, candidates)
    if reranked is None:
        candidates = sorted(candidates, key=lambda h: h["_score"], reverse=True)
    else:
        candidates = reranked

    top_hits = candidates[:_FINAL_TOP_N]
    new_seen_ids = list(seen_ids) + [h["_id"] for h in top_hits]
    return {"merged_hits": top_hits, "result": format_hits(top_hits), "seen_ids": new_seen_ids}


def rerank(state: SecSearchState) -> dict:
    return _rerank_fn(state)


# ─── 그래프 조립 ──────────────────────────────────────────────────────────────

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

# thread_id + ticker 조합별로 반환된 chunk ID를 누적 저장
# key: "{thread_id}:{ticker_upper}", value: seen chunk ID 목록
_seen_ids_cache: dict[str, list[str]] = {}


# ─── @tool 래핑 ───────────────────────────────────────────────────────────────

@tool
def search_sec_filing(
    ticker: str,
    query: str,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    """기업 공시(10-K 사업보고서)에서 질문과 관련된 내용을 검색합니다.
    사업 구조, 리스크 요인, 경영 성과 분석(MD&A) 등 정성적 정보 조회에 사용합니다.
    지원 종목: AAPL, MSFT, TSLA, NVDA (이외 종목 불가)
    보유 데이터: 최신 연간 보고서(10-K) 기준

    subagent-as-tool 패턴:
    StateGraph 전체를 @tool로 래핑하면 메인 에이전트 입장에서 일반 도구와 동일하게 사용된다.
    thread_id 기반 seen_ids 캐시로 cross-invocation 중복 청크를 제거한다.
    """
    ticker_upper = ticker.upper()

    # thread_id 추출 — None일 경우 캐시 비활성화
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    cache_key = f"{thread_id}:{ticker_upper}" if thread_id else None

    seen_ids = _seen_ids_cache.get(cache_key, []) if cache_key else []

    result = _sec_search_graph.invoke({
        "ticker": ticker_upper,
        "query": query,
        "seen_ids": seen_ids,
    })

    # 반환된 seen_ids를 캐시에 저장 (다음 호출에서 사용)
    if cache_key:
        _seen_ids_cache[cache_key] = result.get("seen_ids", seen_ids)

    return result["result"]
