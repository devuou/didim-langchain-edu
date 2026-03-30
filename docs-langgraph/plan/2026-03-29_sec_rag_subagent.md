# SEC 10-K RAG 서브 에이전트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SEC 10-K 공시 문서를 Elasticsearch에 벡터 적재하고, LangGraph StateGraph 기반 병렬 검색(BM25 + kNN) + 리랭킹 서브 에이전트를 구현하여 메인 주식 에이전트에 `search_sec_filing` 도구로 통합한다.

**Architecture:** agent-sample의 `search_agent.py` 패턴을 따른다. StateGraph가 START에서 `bm25_search`와 `vector_search`를 병렬(fan-out) 실행하고, `merge_results` → `rerank` → END 순으로 진행한다. 서브 에이전트는 `@tool`로 래핑되어 메인 에이전트에서 일반 도구처럼 호출된다.

**Tech Stack:** `sec-edgar-downloader`, `beautifulsoup4`, `tiktoken`, `sentence-transformers`, `langchain-openai`(기존), `elasticsearch`(기존), LangGraph StateGraph

---

## File Structure

```
신규 파일
├── scripts/
│   └── ingest_10k.py                  # 오프라인 데이터 파이프라인 (1회 실행)
├── app/agents/
│   ├── tools/
│   │   └── _rag_common.py             # ES vector 클라이언트 + embedding 싱글톤
│   └── sec_search_agent.py            # StateGraph 서브 에이전트 + @tool
└── tests/
    └── test_sec_search.py             # 단위 테스트

수정 파일
├── pyproject.toml                     # 의존성 추가
├── app/core/config.py                 # COHERE_API_KEY 설정 추가
├── app/agents/stock_agent.py          # search_sec_filing 도구 등록
└── app/agents/prompts.py              # search_sec_filing 도구 설명 추가
```

---

## Task 1: 의존성 추가 및 설정

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/core/config.py`
- Modify: `.env` (문서만, 실제 파일은 .gitignore)

- [ ] **Step 1: 패키지 추가**

```bash
uv add sec-edgar-downloader beautifulsoup4 tiktoken sentence-transformers cohere
```

Expected: `pyproject.toml`의 `dependencies`에 5개 패키지 추가됨

- [ ] **Step 2: pyproject.toml 확인**

```bash
uv run python -c "import sec_edgar_downloader, bs4, tiktoken, sentence_transformers, cohere; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: config.py에 COHERE_API_KEY 추가**

`app/core/config.py`의 `Settings` 클래스에 추가:

```python
# Elasticsearch 설정 아래에 추가
# Cohere 설정 (리랭킹용, 선택)
COHERE_API_KEY: str | None = None
```

- [ ] **Step 4: 커밋**

```bash
git add pyproject.toml uv.lock app/core/config.py
git commit -m "feat: add sec-rag dependencies and cohere config"
```

---

## Task 2: 텍스트 청킹 + 섹션 추출 헬퍼 (TDD)

**Files:**
- Create: `scripts/ingest_10k.py` (헬퍼 함수 부분만)
- Create: `tests/test_sec_search.py`

- [ ] **Step 1: 테스트 파일 생성 및 실패 테스트 작성**

`tests/test_sec_search.py`:

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
uv run pytest tests/test_sec_search.py::test_chunk_text_single_chunk_when_short -v
```

Expected: `ERROR` (ImportError: cannot import from scripts.ingest_10k)

- [ ] **Step 3: `scripts/ingest_10k.py` 헬퍼 함수 구현**

```python
"""SEC 10-K 공시 문서를 파싱·청킹·임베딩하여 Elasticsearch에 적재하는 오프라인 파이프라인."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import tiktoken


# ─── 텍스트 청킹 ─────────────────────────────────────────────────────────────

_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> list[str]:
    """텍스트를 max_tokens 크기의 청크로 분할한다. 인접 청크 사이에 overlap 토큰을 공유한다."""
    tokens = _ENCODER.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_ENCODER.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


# ─── 섹션 추출 ───────────────────────────────────────────────────────────────

_SECTION_PATTERNS = {
    "item1":  r"Item\s+1\.\s+Business",
    "item1a": r"Item\s+1A\.\s+Risk\s+Factors",
    "item7":  r"Item\s+7\.\s+",
}

_STOP_PATTERN = r"Item\s+(?:1A|2|3|4|5|6|7A|8)\b"


def extract_sections(text: str) -> dict[str, str]:
    """10-K 텍스트에서 Item 1 / 1A / 7 섹션을 추출한다."""
    result: dict[str, str] = {}
    for key, start_pat in _SECTION_PATTERNS.items():
        m_start = re.search(start_pat, text, re.IGNORECASE)
        if not m_start:
            result[key] = ""
            continue
        body_start = m_start.end()
        # 다음 섹션 헤더가 나오면 거기서 자름
        m_stop = re.search(_STOP_PATTERN, text[body_start:], re.IGNORECASE)
        body_end = body_start + m_stop.start() if m_stop else len(text)
        result[key] = text[body_start:body_end].strip()
    return result
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_chunk_text_single_chunk_when_short \
  tests/test_sec_search.py::test_chunk_text_splits_long_text \
  tests/test_sec_search.py::test_chunk_text_has_overlap \
  tests/test_sec_search.py::test_extract_sections_finds_item1 \
  tests/test_sec_search.py::test_extract_sections_returns_empty_for_missing -v
```

Expected: 5개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add scripts/ingest_10k.py tests/test_sec_search.py
git commit -m "feat: add text chunking and section extraction helpers"
```

---

## Task 3: ES 인덱스 생성 함수 (TDD)

**Files:**
- Modify: `scripts/ingest_10k.py` (인덱스 생성 함수 추가)
- Modify: `tests/test_sec_search.py`

- [ ] **Step 1: 테스트 추가 (`tests/test_sec_search.py` 하단에 추가)**

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
uv run pytest tests/test_sec_search.py::test_build_index_mapping_has_required_fields -v
```

Expected: `ERROR` (ImportError)

- [ ] **Step 3: `scripts/ingest_10k.py`에 인덱스 함수 추가**

기존 코드 아래에 추가:

```python
# ─── ES 인덱스 ───────────────────────────────────────────────────────────────

def build_index_name(prefix: str) -> str:
    return f"{prefix}-10k-docs"


def build_index_mapping() -> dict:
    return {
        "mappings": {
            "properties": {
                "ticker":       {"type": "keyword"},
                "section":      {"type": "keyword"},
                "fiscal_year":  {"type": "keyword"},
                "text":         {"type": "text"},
                "embedding":    {"type": "dense_vector", "dims": 1536},
                "chunk_id":     {"type": "keyword"},
                "ingested_at":  {"type": "date"},
            }
        }
    }


def ensure_index(es_client, index_name: str) -> None:
    """인덱스가 없으면 생성한다."""
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=build_index_mapping())
        print(f"인덱스 생성: {index_name}")
    else:
        print(f"인덱스 이미 존재: {index_name}")
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_build_index_mapping_has_required_fields \
  tests/test_sec_search.py::test_build_index_name_uses_prefix -v
```

Expected: 2개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add scripts/ingest_10k.py tests/test_sec_search.py
git commit -m "feat: add ES index mapping and creation helper"
```

---

## Task 4: 전체 적재 스크립트 완성

**Files:**
- Modify: `scripts/ingest_10k.py` (다운로드 + 임베딩 + bulk upsert 추가)

- [ ] **Step 1: `scripts/ingest_10k.py` 완성 — 다운로드 + 임베딩 + upsert**

기존 코드 아래에 추가:

```python
# ─── 다운로드 ─────────────────────────────────────────────────────────────────

import tempfile
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup


TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def download_latest_10k_text(ticker: str, download_dir: str) -> str:
    """SEC EDGAR에서 최신 10-K를 다운로드하고 텍스트를 반환한다."""
    dl = Downloader("student-project", "student@example.com", download_dir)
    dl.get("10-K", ticker, limit=1, download_details=False)

    base = Path(download_dir) / "sec-edgar-filings" / ticker / "10-K"
    filing_dirs = sorted(base.iterdir())
    if not filing_dirs:
        raise FileNotFoundError(f"{ticker} 10-K 다운로드 실패")
    filing_dir = filing_dirs[-1]

    # primary-document.html 또는 첫 번째 .htm 파일
    htm_files = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
    if not htm_files:
        raise FileNotFoundError(f"{ticker}: .htm 파일을 찾을 수 없음: {filing_dir}")

    html = htm_files[0].read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# ─── 임베딩 ──────────────────────────────────────────────────────────────────

from openai import OpenAI as _OpenAI

_openai_client: _OpenAI | None = None


def get_openai_client() -> _OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = _OpenAI()
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 목록을 임베딩 벡터 목록으로 변환한다. 배치 처리."""
    client = get_openai_client()
    batch_size = 100
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small",
        )
        all_vectors.extend([item.embedding for item in resp.data])
    return all_vectors


# ─── Bulk Upsert ─────────────────────────────────────────────────────────────

from datetime import datetime, timezone
from elasticsearch import Elasticsearch, helpers as es_helpers


def build_docs(
    ticker: str,
    fiscal_year: str,
    sections: dict[str, str],
    max_tokens: int = 512,
    overlap: int = 50,
) -> Generator[dict, None, None]:
    """섹션별 청크를 ES 문서 딕셔너리로 변환하는 제너레이터."""
    for section_key, section_text in sections.items():
        if not section_text:
            continue
        chunks = chunk_text(section_text, max_tokens=max_tokens, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{ticker}_{section_key}_{idx:04d}"
            yield {
                "ticker": ticker,
                "section": section_key,
                "fiscal_year": fiscal_year,
                "text": chunk,
                "chunk_id": chunk_id,
            }


def ingest_ticker(
    es_client: Elasticsearch,
    index_name: str,
    ticker: str,
    fiscal_year: str,
    download_dir: str,
) -> int:
    """티커 1개를 다운로드 → 파싱 → 임베딩 → ES 적재한다. 적재된 문서 수를 반환한다."""
    print(f"\n[{ticker}] 다운로드 중...")
    text = download_latest_10k_text(ticker, download_dir)

    print(f"[{ticker}] 섹션 추출 중...")
    sections = extract_sections(text)

    docs = list(build_docs(ticker, fiscal_year, sections))
    if not docs:
        print(f"[{ticker}] 추출된 청크 없음 — 건너뜀")
        return 0

    print(f"[{ticker}] 임베딩 중... ({len(docs)}개 청크)")
    texts = [d["text"] for d in docs]
    vectors = embed_texts(texts)

    ingested_at = datetime.now(timezone.utc).isoformat()
    actions = [
        {
            "_index": index_name,
            "_id": doc["chunk_id"],
            "_source": {**doc, "embedding": vec, "ingested_at": ingested_at},
        }
        for doc, vec in zip(docs, vectors)
    ]

    success, _ = es_helpers.bulk(es_client, actions)
    print(f"[{ticker}] 적재 완료: {success}개")
    return success


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main() -> None:
    import os
    from dotenv import load_dotenv

    load_dotenv()

    es_url = os.getenv("ES_URL", "http://localhost:9200")
    es_user = os.getenv("ES_USERNAME")
    es_pass = os.getenv("ES_PASSWORD")
    prefix = os.getenv("ES_INDEX_PREFIX", "dev")
    fiscal_year = "2024"

    es_kwargs: dict = {"hosts": [es_url]}
    if es_user and es_pass:
        es_kwargs["basic_auth"] = (es_user, es_pass)
    es = Elasticsearch(**es_kwargs)

    index_name = build_index_name(prefix)
    ensure_index(es, index_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        total = 0
        for ticker in TICKERS:
            total += ingest_ticker(es, index_name, ticker, fiscal_year, tmpdir)

    print(f"\n완료: 총 {total}개 문서 적재")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 헬퍼 함수 단위 테스트 추가 (`tests/test_sec_search.py`)**

```python
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
```

- [ ] **Step 3: 테스트 실행 → 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_build_docs_generates_correct_chunk_ids \
  tests/test_sec_search.py::test_build_docs_skips_empty_section -v
```

Expected: 2개 PASSED

- [ ] **Step 4: 커밋**

```bash
git add scripts/ingest_10k.py tests/test_sec_search.py
git commit -m "feat: complete ingest_10k pipeline script"
```

---

## Task 5: RAG 클라이언트 싱글톤 (TDD)

**Files:**
- Create: `app/agents/tools/_rag_common.py`
- Modify: `tests/test_sec_search.py`

- [ ] **Step 1: 테스트 추가 (`tests/test_sec_search.py` 하단에 추가)**

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
uv run pytest tests/test_sec_search.py::test_get_es_client_returns_singleton -v
```

Expected: `ERROR` (ModuleNotFoundError)

- [ ] **Step 3: `app/agents/tools/_rag_common.py` 구현**

```python
"""RAG 파이프라인에서 사용하는 ES 벡터 클라이언트, 임베딩, 리랭커 싱글톤."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from openai import OpenAI
    from sentence_transformers import CrossEncoder

_es_client: "Elasticsearch | None" = None
_openai_client: "OpenAI | None" = None
_reranker: "CrossEncoder | None" = None
_reranker_initialized: bool = False


def _get_cohere_api_key() -> str | None:
    try:
        from app.core.config import settings
        return settings.COHERE_API_KEY
    except Exception:
        return None


def get_es_client() -> "Elasticsearch":
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch
        from app.core.config import settings
        kwargs: dict = {"hosts": [settings.ES_URL]}
        if settings.ES_USERNAME and settings.ES_PASSWORD:
            kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)
        _es_client = Elasticsearch(**kwargs)
    return _es_client


def get_openai_client() -> "OpenAI":
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        from app.core.config import settings
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def get_reranker() -> "CrossEncoder | None":
    """cross-encoder를 반환한다. COHERE_API_KEY가 없으면 None (score 정렬 fallback)."""
    global _reranker, _reranker_initialized
    if _reranker_initialized:
        return _reranker
    _reranker_initialized = True
    if not _get_cohere_api_key():
        _reranker = None
        return None
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _reranker = None
    return _reranker


def embed_query(query: str) -> list[float]:
    """쿼리 문자열을 1536차원 벡터로 변환한다."""
    client = get_openai_client()
    resp = client.embeddings.create(input=[query], model="text-embedding-3-small")
    return resp.data[0].embedding


def format_hits(hits: list[dict]) -> str:
    """ES 검색 결과(hits)를 LLM이 읽기 좋은 문자열로 변환한다."""
    if not hits:
        return "관련 공시 정보를 찾을 수 없습니다."
    lines: list[str] = []
    for i, hit in enumerate(hits, 1):
        src = hit["_source"]
        lines.append(
            f"[{i}] 섹션: {src.get('section', '')} | 티커: {src.get('ticker', '')}\n"
            f"{src.get('text', '').strip()}"
        )
    return "\n\n".join(lines)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_get_es_client_returns_singleton \
  tests/test_sec_search.py::test_get_reranker_returns_none_without_cohere \
  tests/test_sec_search.py::test_format_hits_returns_string -v
```

Expected: 3개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add app/agents/tools/_rag_common.py tests/test_sec_search.py
git commit -m "feat: add RAG client singletons (_rag_common)"
```

---

## Task 6: StateGraph 노드 구현 (TDD)

**Files:**
- Create: `app/agents/sec_search_agent.py`
- Modify: `tests/test_sec_search.py`

- [ ] **Step 1: 테스트 추가 (`tests/test_sec_search.py` 하단에 추가)**

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
uv run pytest tests/test_sec_search.py::test_sec_search_state_accepts_all_fields -v
```

Expected: `ERROR` (ImportError)

- [ ] **Step 3: `app/agents/sec_search_agent.py` 노드 구현**

```python
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
        # fallback: score 내림차순
        hits = sorted(hits, key=lambda h: h["_score"], reverse=True)
    else:
        pairs = [[query, h["_source"]["text"]] for h in hits]
        scores = reranker.predict(pairs)
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]

    top_hits = hits[:_FINAL_TOP_N]
    return {"merged_hits": top_hits, "result": format_hits(top_hits)}


def rerank(state: SecSearchState) -> dict:
    return _rerank_fn(state)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_sec_search_state_accepts_all_fields \
  tests/test_sec_search.py::test_merge_results_deduplicates_by_id \
  tests/test_sec_search.py::test_rerank_sorts_by_score_without_reranker -v
```

Expected: 3개 PASSED

- [ ] **Step 5: 커밋**

```bash
git add app/agents/sec_search_agent.py tests/test_sec_search.py
git commit -m "feat: add SecSearchState and node functions (bm25, vector, merge, rerank)"
```

---

## Task 7: 그래프 조립 + @tool 래핑 + 통합 테스트

**Files:**
- Modify: `app/agents/sec_search_agent.py` (그래프 + tool 추가)
- Modify: `tests/test_sec_search.py`

- [ ] **Step 1: 테스트 추가 (`tests/test_sec_search.py` 하단에 추가)**

```python
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

    # tool을 직접 함수로 호출
    result = module.search_sec_filing.invoke({"ticker": "AAPL", "query": "사업 리스크"})
    assert "Apple faces" in result
    mock_graph.invoke.assert_called_once_with(
        {"ticker": "AAPL", "query": "사업 리스크"}
    )
```

- [ ] **Step 2: `app/agents/sec_search_agent.py` 하단에 그래프 + tool 추가**

기존 노드 함수 아래에 추가:

```python
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
```

- [ ] **Step 3: 테스트 실행 → 통과 확인**

```bash
uv run pytest tests/test_sec_search.py::test_search_sec_filing_tool_is_callable \
  tests/test_sec_search.py::test_search_sec_filing_invokes_graph -v
```

Expected: 2개 PASSED

- [ ] **Step 4: 전체 테스트 통과 확인**

```bash
uv run pytest tests/test_sec_search.py -v
```

Expected: 전체 PASSED (실패 없음)

- [ ] **Step 5: 커밋**

```bash
git add app/agents/sec_search_agent.py tests/test_sec_search.py
git commit -m "feat: assemble StateGraph and wrap as @tool search_sec_filing"
```

---

## Task 8: 메인 에이전트 통합

**Files:**
- Modify: `app/agents/stock_agent.py`
- Modify: `app/agents/prompts.py`

- [ ] **Step 1: `app/agents/stock_agent.py`에 도구 등록**

기존 import 목록에 추가:

```python
from app.agents.sec_search_agent import search_sec_filing
```

`create_stock_agent()` 함수의 `tools` 리스트에 추가:

```python
tools=[
    get_stock_price,
    get_company_info,
    get_recent_news,
    get_stock_history,
    search_sec_filing,   # ← 신규
],
```

- [ ] **Step 2: `app/agents/prompts.py`에 도구 설명 추가**

`get_stock_history` 아래에 추가:

```python
  - search_sec_filing: 기업 공시(10-K)에서 사업 구조, 리스크, 경영 성과 등 정성적 정보가 필요할 때 사용 (지원 종목: AAPL, MSFT, TSLA, NVDA)
```

- [ ] **Step 3: import 정상 확인**

```bash
uv run python -c "from app.agents.stock_agent import create_stock_agent; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: 커밋**

```bash
git add app/agents/stock_agent.py app/agents/prompts.py
git commit -m "feat: register search_sec_filing in main stock agent"
```

---

## Task 9: 데이터 적재 실행 + 동작 검증

**전제 조건:** ES 서버 접속 가능, OpenAI API 키 설정 완료

- [ ] **Step 1: 10-K 데이터 적재 실행**

```bash
uv run python scripts/ingest_10k.py
```

Expected:
```
인덱스 생성: dev-10k-docs  (또는 "이미 존재")
[AAPL] 다운로드 중...
[AAPL] 섹션 추출 중...
[AAPL] 임베딩 중... (N개 청크)
[AAPL] 적재 완료: N개
...
완료: 총 N개 문서 적재
```

- [ ] **Step 2: Kibana에서 적재 확인**

Kibana DevTools에서 실행:
```
GET dev-10k-docs/_count
GET dev-10k-docs/_search?size=1
```

Expected: `count` > 0, `hits.hits[0]._source`에 `embedding` 필드(1536차원) 확인

- [ ] **Step 3: 서버 기동 + 동작 테스트**

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

별도 터미널에서:
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "test-rag-01", "message": "AAPL의 주요 사업 리스크가 뭐야?"}' \
  --no-buffer
```

Expected: SSE 스트림에서 `"tool_calls": ["search_sec_filing"]` 이벤트 후 공시 기반 답변

- [ ] **Step 4: Opik 트레이스 확인**

Opik 대시보드에서 최근 트레이스 확인:
- `search_sec_filing` 도구 호출 스팬 존재
- 내부 `bm25_search`, `vector_search` 병렬 스팬 확인

- [ ] **Step 5: 최종 커밋**

```bash
git add .
git commit -m "feat: SEC 10-K RAG 서브 에이전트 구현 완료"
```

---

## 자가 검토 (Self-Review)

### 스펙 커버리지

| 스펙 항목 | 구현 태스크 |
|---|---|
| SEC EDGAR 10-K 다운로드 | Task 4 |
| 섹션 추출 (Item 1 / 1A / 7) | Task 2 |
| 텍스트 청킹 (512 tokens + overlap) | Task 2 |
| 임베딩 (text-embedding-3-small) | Task 4 |
| ES 적재 (dense_vector, upsert) | Task 3, 4 |
| BM25 + kNN 병렬 검색 | Task 6, 7 |
| 리랭킹 (cross-encoder, fallback) | Task 5, 6 |
| `@tool` 래핑 | Task 7 |
| 싱글톤 클라이언트 (_rag_common) | Task 5 |
| 메인 에이전트 통합 | Task 8 |
| 데이터 적재 + 동작 검증 | Task 9 |

### 타입 일관성 확인

- `SecSearchState` 필드명: Task 6 정의 → Task 6 노드 함수에서 동일하게 사용 ✓
- `search_sec_filing` tool 파라미터: `ticker: str, query: str` → Task 8 등록 시 동일 ✓
- `_sec_search_graph` 변수명: Task 7 정의 → Task 7 테스트 monkeypatch에서 동일 ✓
