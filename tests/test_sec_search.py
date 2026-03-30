import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Task 2: 청킹 + 섹션 추출 ────────────────────────────────────────────────

def test_chunk_text_single_chunk_when_short():
    """짧은 텍스트는 청크 1개"""
    from scripts.ingest_10k import chunk_text
    chunks = chunk_text("hello world", max_tokens=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_text_splits_long_text():
    """긴 텍스트는 여러 청크로 분리"""
    from scripts.ingest_10k import chunk_text
    # 200개 단어 = 약 200 토큰
    text = " ".join(["word"] * 200)
    chunks = chunk_text(text, max_tokens=50, overlap=10)
    assert len(chunks) > 1


def test_chunk_text_has_overlap():
    """인접 청크 사이에 오버랩이 존재"""
    from scripts.ingest_10k import chunk_text
    text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunk_text(text, max_tokens=30, overlap=10)
    # chunk[0] 끝 10 토큰이 chunk[1] 시작에 포함돼야 함
    tokens_0 = chunks[0].split()
    tokens_1 = chunks[1].split()
    assert tokens_0[-5] in tokens_1[:15]


def test_extract_sections_finds_item1():
    """Item 1 섹션을 올바르게 추출"""
    from scripts.ingest_10k import extract_sections
    fake_text = (
        "PART I\n"
        "Item 1. Business\n"
        "Apple Inc. designs consumer electronics.\n"
        "Item 1A. Risk Factors\n"
        "Our business faces many risks.\n"
        "Item 7. MD&A\n"
        "Revenue increased 5%.\n"
        "Item 8. Financial Statements\n"
        "Other content.\n"
    )
    sections = extract_sections(fake_text)
    assert "item1" in sections
    assert "Apple Inc." in sections["item1"]
    assert "item1a" in sections
    assert "many risks" in sections["item1a"]
    assert "item7" in sections
    assert "Revenue" in sections["item7"]


def test_extract_sections_returns_empty_for_missing():
    """존재하지 않는 섹션은 빈 문자열"""
    from scripts.ingest_10k import extract_sections
    fake_text = "No relevant sections here."
    sections = extract_sections(fake_text)
    assert sections.get("item1", "") == ""


# ─── Task 3: ES 인덱스 생성 ──────────────────────────────────────────────────

def test_build_index_mapping_has_required_fields():
    """인덱스 매핑에 필수 필드가 모두 포함됐는지 확인"""
    from scripts.ingest_10k import build_index_mapping
    mapping = build_index_mapping()
    props = mapping["mappings"]["properties"]
    assert "ticker" in props
    assert "section" in props
    assert "text" in props
    assert "embedding" in props
    assert props["embedding"]["type"] == "dense_vector"
    assert props["embedding"]["dims"] == 1536


def test_build_index_name_uses_prefix():
    """인덱스명이 ES_INDEX_PREFIX를 포함하는지 확인"""
    from scripts.ingest_10k import build_index_name
    name = build_index_name("myprefix")
    assert name == "myprefix-10k-docs"


# ─── Task 4: build_docs 검증 ─────────────────────────────────────────────────

def test_build_docs_generates_correct_chunk_ids():
    """생성된 문서의 chunk_id 형식이 올바른지 확인"""
    from scripts.ingest_10k import build_docs
    sections = {"item1": "Apple designs iPhones and Macs for consumers worldwide."}
    docs = list(build_docs("AAPL", "2024", sections, max_tokens=20, overlap=5))
    assert len(docs) >= 1
    assert docs[0]["chunk_id"].startswith("AAPL_item1_")
    assert docs[0]["ticker"] == "AAPL"
    assert docs[0]["fiscal_year"] == "2024"


def test_build_docs_skips_empty_section():
    """빈 섹션은 문서를 생성하지 않음"""
    from scripts.ingest_10k import build_docs
    sections = {"item1": "", "item1a": "Some risk content here."}
    docs = list(build_docs("MSFT", "2024", sections))
    for doc in docs:
        assert doc["section"] != "item1"


# ─── Task 5: _rag_common 싱글톤 ──────────────────────────────────────────────

def test_get_es_client_returns_singleton():
    """동일 인스턴스를 반환하는지 확인"""
    import app.agents.tools._rag_common as rag
    rag._es_client = None  # 초기화 리셋
    c1 = rag.get_es_client()
    c2 = rag.get_es_client()
    assert c1 is c2


def test_get_reranker_returns_none_without_cohere(monkeypatch):
    """COHERE_API_KEY 없을 때 None 반환"""
    import app.agents.tools._rag_common as rag
    rag._reranker = None
    rag._reranker_initialized = False
    monkeypatch.setattr(
        "app.agents.tools._rag_common._get_cohere_api_key",
        lambda: None,
    )
    result = rag.get_reranker()
    assert result is None


def test_format_hits_returns_string():
    """hits 목록을 문자열로 포맷팅하는지 확인"""
    from app.agents.tools._rag_common import format_hits
    hits = [
        {"_source": {"text": "Apple designs consumer electronics.", "section": "item1", "ticker": "AAPL"}, "_score": 1.5},
        {"_source": {"text": "Risk factors include competition.", "section": "item1a", "ticker": "AAPL"}, "_score": 1.2},
    ]
    result = format_hits(hits)
    assert "Apple designs" in result
    assert "Risk factors" in result
    assert isinstance(result, str)


# ─── Task 6: sec_search_agent 노드 ────────────────────────────────────────────

def test_sec_search_state_accepts_all_fields():
    """SecSearchState TypedDict가 모든 필드를 허용하는지 확인"""
    from app.agents.sec_search_agent import SecSearchState
    state: SecSearchState = {
        "query": "사업 리스크",
        "ticker": "AAPL",
        "bm25_hits": [],
        "vector_hits": [],
        "merged_hits": [],
        "result": "",
    }
    assert state["ticker"] == "AAPL"


def test_merge_results_deduplicates_by_id():
    """동일 _id는 한 번만 포함됨"""
    from app.agents.sec_search_agent import _merge_results_fn
    state = {
        "bm25_hits": [
            {"_id": "chunk1", "_source": {"text": "a", "section": "item1", "ticker": "AAPL"}, "_score": 1.5},
        ],
        "vector_hits": [
            {"_id": "chunk1", "_source": {"text": "a", "section": "item1", "ticker": "AAPL"}, "_score": 0.9},
            {"_id": "chunk2", "_source": {"text": "b", "section": "item1a", "ticker": "AAPL"}, "_score": 0.8},
        ],
        "merged_hits": [],
    }
    result = _merge_results_fn(state)
    assert len(result["merged_hits"]) == 2


def test_rerank_sorts_by_score_without_reranker(monkeypatch):
    """리랭커 없을 때 score 내림차순으로 정렬"""
    import app.agents.sec_search_agent as module
    monkeypatch.setattr(module, "_get_reranker", lambda: None)
    from app.agents.sec_search_agent import _rerank_fn
    state = {
        "merged_hits": [
            {"_id": "a", "_source": {"text": "low", "section": "item1", "ticker": "AAPL"}, "_score": 0.5},
            {"_id": "b", "_source": {"text": "high", "section": "item1", "ticker": "AAPL"}, "_score": 2.0},
        ],
        "query": "test",
    }
    result = _rerank_fn(state)
    assert result["merged_hits"][0]["_id"] == "b"
    assert "high" in result["result"]


# ─── Task 7: 그래프 + tool ────────────────────────────────────────────────────

def test_search_sec_filing_tool_is_callable():
    """search_sec_filing이 LangChain tool로 등록됐는지 확인"""
    from app.agents.sec_search_agent import search_sec_filing
    assert hasattr(search_sec_filing, "invoke")
    assert search_sec_filing.name == "search_sec_filing"


def test_search_sec_filing_invokes_graph(monkeypatch):
    """search_sec_filing 호출 시 내부 그래프가 invoke되는지 확인"""
    from unittest.mock import MagicMock
    import app.agents.sec_search_agent as module

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"result": "Apple faces competition risks."}
    monkeypatch.setattr(module, "_sec_search_graph", mock_graph)

    result = module.search_sec_filing.invoke({"ticker": "AAPL", "query": "사업 리스크"})
    assert "Apple faces" in result
    mock_graph.invoke.assert_called_once_with(
        {"ticker": "AAPL", "query": "사업 리스크"}
    )
