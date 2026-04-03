# Sub-agent seen_ids Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** "추가로 더 없어?" 요청 시 서브 에이전트가 이전에 반환한 청크를 제외하고 다음 순위 청크를 반환하도록 한다.

**Architecture:** 메인 에이전트의 `thread_id`를 `RunnableConfig`로 툴에 주입하고, 모듈 수준 캐시(`_seen_ids_cache`)에 `{thread_id}:{ticker}` 키로 반환된 chunk ID를 축적한다. `rerank` 노드에서 이전에 반환된 청크를 필터링한 후 상위 N개를 선택하고, 선택된 ID를 state에 반영한다. 툴 호출 완료 후 캐시를 업데이트한다.

**Tech Stack:** LangGraph `StateGraph`, `langchain_core.tools.InjectedToolArg`, `langchain_core.runnables.RunnableConfig`, Python `TypedDict`

---

## 수정 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `app/agents/sec_search_agent.py` | `SecSearchState`에 `seen_ids` 추가, `_seen_ids_cache` 추가, `_rerank_fn` 필터링 로직 추가, `search_sec_filing` 시그니처 및 캐시 연동 |
| `tests/test_sec_search.py` | `seen_ids` 필터링 테스트 2개 추가, 기존 state/graph 테스트 업데이트 |

---

## Task 1: `SecSearchState`에 `seen_ids` 필드 추가

**Files:**
- Modify: `app/agents/sec_search_agent.py`
- Test: `tests/test_sec_search.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_sec_search.py`의 `test_sec_search_state_accepts_all_fields` 를 아래와 같이 수정:

```python
def test_sec_search_state_accepts_all_fields():
    """SecSearchState TypedDict가 모든 필드(seen_ids 포함)를 허용하는지 확인"""
    from app.agents.sec_search_agent import SecSearchState
    state: SecSearchState = {
        "query": "사업 리스크",
        "ticker": "AAPL",
        "bm25_hits": [],
        "vector_hits": [],
        "merged_hits": [],
        "result": "",
        "seen_ids": ["AAPL_item1_0", "AAPL_item1a_0"],
    }
    assert state["ticker"] == "AAPL"
    assert state["seen_ids"] == ["AAPL_item1_0", "AAPL_item1a_0"]
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
uv run pytest tests/test_sec_search.py::test_sec_search_state_accepts_all_fields -v
```

Expected: `FAILED` — `TypedDict got an unexpected key 'seen_ids'` 또는 타입 오류

- [ ] **Step 3: `SecSearchState`에 `seen_ids` 필드 추가**

`app/agents/sec_search_agent.py`의 `SecSearchState` 클래스를 아래와 같이 수정:

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_sec_search_state_accepts_all_fields -v
```

Expected: `PASSED`

- [ ] **Step 5: 전체 테스트 통과 확인**

```bash
uv run pytest tests/test_sec_search.py -v
```

Expected: 전체 통과 (기존 테스트 영향 없음)

- [ ] **Step 6: 커밋**

```bash
git add app/agents/sec_search_agent.py tests/test_sec_search.py
git commit -m "feat: add seen_ids field to SecSearchState for cross-invocation dedup"
```

---

## Task 2: `_rerank_fn`에 seen_ids 필터링 로직 추가

**Files:**
- Modify: `app/agents/sec_search_agent.py`
- Test: `tests/test_sec_search.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_sec_search.py`에 아래 두 테스트를 추가:

```python
def test_rerank_filters_seen_ids(monkeypatch):
    """seen_ids에 포함된 청크는 결과에서 제외됨"""
    import app.agents.sec_search_agent as module
    monkeypatch.setattr(module, "rerank_hits", lambda query, hits: None)
    from app.agents.sec_search_agent import _rerank_fn
    state = {
        "merged_hits": [
            {"_id": "chunk_a", "_source": {"text": "already returned", "section": "item1", "ticker": "AAPL", "fiscal_year": "2024"}, "_score": 2.0},
            {"_id": "chunk_b", "_source": {"text": "new content", "section": "item1a", "ticker": "AAPL", "fiscal_year": "2024"}, "_score": 1.5},
        ],
        "query": "test",
        "seen_ids": ["chunk_a"],
    }
    result = _rerank_fn(state)
    ids = [h["_id"] for h in result["merged_hits"]]
    assert "chunk_a" not in ids
    assert "chunk_b" in ids


def test_rerank_falls_back_when_all_seen(monkeypatch):
    """모든 청크가 seen_ids에 포함된 경우 필터 없이 전체 반환"""
    import app.agents.sec_search_agent as module
    monkeypatch.setattr(module, "rerank_hits", lambda query, hits: None)
    from app.agents.sec_search_agent import _rerank_fn
    state = {
        "merged_hits": [
            {"_id": "chunk_a", "_source": {"text": "only chunk", "section": "item1", "ticker": "AAPL", "fiscal_year": "2024"}, "_score": 1.0},
        ],
        "query": "test",
        "seen_ids": ["chunk_a"],
    }
    result = _rerank_fn(state)
    # fallback: 전체 hits 반환 (중복이라도 빈 결과보다 낫다)
    assert len(result["merged_hits"]) == 1
    assert result["merged_hits"][0]["_id"] == "chunk_a"
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
uv run pytest tests/test_sec_search.py::test_rerank_filters_seen_ids tests/test_sec_search.py::test_rerank_falls_back_when_all_seen -v
```

Expected: `FAILED` — 현재 `_rerank_fn`에 seen_ids 필터 없음

- [ ] **Step 3: `_rerank_fn` seen_ids 필터링 구현**

`app/agents/sec_search_agent.py`의 `_rerank_fn`을 아래와 같이 수정:

```python
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
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_rerank_filters_seen_ids tests/test_sec_search.py::test_rerank_falls_back_when_all_seen -v
```

Expected: 두 테스트 모두 `PASSED`

- [ ] **Step 5: 전체 테스트 통과 확인**

```bash
uv run pytest tests/test_sec_search.py -v
```

Expected: 전체 통과

- [ ] **Step 6: 커밋**

```bash
git add app/agents/sec_search_agent.py tests/test_sec_search.py
git commit -m "feat: filter seen_ids in rerank node, fallback to full set when all seen"
```

---

## Task 3: `search_sec_filing` 툴에 thread_id 기반 캐시 연동

**Files:**
- Modify: `app/agents/sec_search_agent.py`
- Test: `tests/test_sec_search.py`

- [ ] **Step 1: 기존 그래프 invoke 테스트 업데이트**

`tests/test_sec_search.py`의 `test_search_sec_filing_invokes_graph`를 아래와 같이 수정:

```python
def test_search_sec_filing_invokes_graph(monkeypatch):
    """search_sec_filing 호출 시 내부 그래프가 seen_ids와 함께 invoke되는지 확인"""
    from unittest.mock import MagicMock
    import app.agents.sec_search_agent as module

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "result": "Apple faces competition risks.",
        "seen_ids": ["AAPL_item1_0"],
    }
    monkeypatch.setattr(module, "_sec_search_graph", mock_graph)
    # 캐시 초기화
    module._seen_ids_cache.clear()

    result = module.search_sec_filing.invoke({"ticker": "AAPL", "query": "사업 리스크"})
    assert "Apple faces" in result
    mock_graph.invoke.assert_called_once_with(
        {"ticker": "AAPL", "query": "사업 리스크", "seen_ids": []}
    )
```

- [ ] **Step 2: 캐시 누적 테스트 추가**

```python
def test_search_sec_filing_accumulates_seen_ids(monkeypatch):
    """두 번째 호출 시 첫 번째에서 반환된 seen_ids가 전달됨"""
    from unittest.mock import MagicMock
    import app.agents.sec_search_agent as module

    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = [
        {"result": "first result", "seen_ids": ["chunk_a", "chunk_b"]},
        {"result": "second result", "seen_ids": ["chunk_a", "chunk_b", "chunk_c"]},
    ]
    monkeypatch.setattr(module, "_sec_search_graph", mock_graph)
    module._seen_ids_cache.clear()

    # 첫 번째 호출
    module.search_sec_filing.invoke({"ticker": "AAPL", "query": "리스크"})
    # 두 번째 호출 — 첫 번째의 seen_ids가 전달돼야 함
    module.search_sec_filing.invoke({"ticker": "AAPL", "query": "리스크"})

    second_call_args = mock_graph.invoke.call_args_list[1][0][0]
    assert second_call_args["seen_ids"] == ["chunk_a", "chunk_b"]
```

- [ ] **Step 3: 테스트 실행 — 실패 확인**

```bash
uv run pytest tests/test_sec_search.py::test_search_sec_filing_invokes_graph tests/test_sec_search.py::test_search_sec_filing_accumulates_seen_ids -v
```

Expected: `FAILED`

- [ ] **Step 4: `_seen_ids_cache` 및 `search_sec_filing` 캐시 연동 구현**

`app/agents/sec_search_agent.py`에서 import 블록 아래에 캐시 변수를 추가하고 `search_sec_filing`을 수정:

```python
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing import Annotated
```

캐시 변수 (모듈 레벨, `_sec_search_graph` 선언 위에 추가):

```python
# thread_id + ticker 조합별로 반환된 chunk ID를 누적 저장
# key: "{thread_id}:{ticker_upper}", value: seen chunk ID 목록
_seen_ids_cache: dict[str, list[str]] = {}
```

`search_sec_filing` 툴 전체를 아래와 같이 교체:

```python
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

    # thread_id 추출 — config 없으면 캐시 비활성화
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
```

- [ ] **Step 5: 테스트 실행 — 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_search_sec_filing_invokes_graph tests/test_sec_search.py::test_search_sec_filing_accumulates_seen_ids -v
```

Expected: 두 테스트 모두 `PASSED`

- [ ] **Step 6: 전체 테스트 통과 확인**

```bash
uv run pytest tests/test_sec_search.py -v
```

Expected: 전체 통과 (17개 → 20개)

- [ ] **Step 7: 커밋**

```bash
git add app/agents/sec_search_agent.py tests/test_sec_search.py
git commit -m "feat: thread_id cache in search_sec_filing for cross-invocation seen_ids"
```

---

## 최종 동작 확인

- [ ] **서버 기동 후 수동 테스트**

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

1차 호출:
```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "test-dedup-001", "message": "AAPL 리스크 요인 알려줘"}'
```

2차 호출 (추가 요청):
```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "test-dedup-001", "message": "추가로 더 없어?"}'
```

Expected: 2차 호출에서 1차와 다른 청크가 반환됨
