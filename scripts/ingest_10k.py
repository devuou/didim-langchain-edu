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

# 섹션별 (시작 패턴, 종료 패턴)
# 종료 패턴에 \. 을 포함해 실제 섹션 헤더(Item N.)만 매칭하고,
# 본문 내 교차 참조("see Item 8 of this Form 10-K" 등)는 제외한다.
_SECTION_CONFIG: dict[str, tuple[str, str]] = {
    "item1":  (r"Item\s+1\.\s+Business",        r"Item\s+1A\."),
    "item1a": (r"Item\s+1A\.\s+Risk\s+Factors", r"Item\s+(?:1B|2)\."),
    "item7":  (r"Item\s+7\.\s+",                r"Item\s+(?:7A|8)\."),
}


def extract_sections(text: str) -> dict[str, str]:
    """10-K 텍스트에서 Item 1 / 1A / 7 섹션을 추출한다.

    SEC 10-K 문서는 앞부분에 목차(TOC)가 있고 뒤에 본문이 나온다.
    같은 헤더가 여러 번 매칭되므로 마지막 매칭(본문)을 사용한다.
    섹션마다 다른 종료 패턴을 사용해 인접 섹션 헤더나 내부 참조로 인한
    조기 종료를 방지한다.
    """
    result: dict[str, str] = {}
    for key, (start_pat, stop_pat) in _SECTION_CONFIG.items():
        matches = list(re.finditer(start_pat, text, re.IGNORECASE))
        if not matches:
            result[key] = ""
            continue
        # 마지막 매칭 = 실제 본문 섹션 (첫 번째는 TOC)
        m_start = matches[-1]
        body_start = m_start.end()
        m_stop = re.search(stop_pat, text[body_start:], re.IGNORECASE)
        body_end = body_start + m_stop.start() if m_stop else len(text)
        result[key] = text[body_start:body_end].strip()
    return result


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


# ─── 문서 생성 ────────────────────────────────────────────────────────────────

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


# ─── 다운로드 ─────────────────────────────────────────────────────────────────

import tempfile
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup


TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def download_latest_10k_text(ticker: str, download_dir: str) -> str:
    """SEC EDGAR에서 최신 10-K를 다운로드하고 텍스트를 반환한다."""
    dl = Downloader("student-project", "student@example.com", download_dir)
    dl.get("10-K", ticker, limit=1, download_details=True)

    base = Path(download_dir) / "sec-edgar-filings" / ticker / "10-K"
    filing_dirs = sorted(base.iterdir())
    if not filing_dirs:
        raise FileNotFoundError(f"{ticker} 10-K 다운로드 실패")
    filing_dir = filing_dirs[-1]

    htm_files = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
    if not htm_files:
        raise FileNotFoundError(f"{ticker}: .htm 파일을 찾을 수 없음: {filing_dir}")

    html = htm_files[0].read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# ─── 임베딩 ──────────────────────────────────────────────────────────────────

import time
from openai import OpenAI as _OpenAI, RateLimitError, APITimeoutError, APIError

_openai_client: _OpenAI | None = None

_BATCH_SIZE = 100
_MAX_RETRIES = 3
_RETRY_DELAY = 5  # 초, 재시도마다 배수 증가


def get_openai_client() -> _OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = _OpenAI()
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 목록을 임베딩 벡터 목록으로 변환한다. 배치 처리 + 지수 백오프 재시도."""
    client = get_openai_client()
    all_vectors: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small",
                )
                all_vectors.extend([item.embedding for item in resp.data])
                print(f"  임베딩 생성: {len(all_vectors)}/{len(texts)}")
                break
            except RateLimitError:
                wait = _RETRY_DELAY * attempt
                print(f"  Rate limit 초과, {wait}초 후 재시도 ({attempt}/{_MAX_RETRIES})")
                time.sleep(wait)
            except (APITimeoutError, APIError) as e:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(f"임베딩 생성 실패 (재시도 {_MAX_RETRIES}회 초과): {e}")
                wait = _RETRY_DELAY * attempt
                print(f"  API 오류, {wait}초 후 재시도 ({attempt}/{_MAX_RETRIES}): {e}")
                time.sleep(wait)

    return all_vectors


# ─── Bulk Upsert ─────────────────────────────────────────────────────────────

from datetime import datetime, timezone
from elasticsearch import Elasticsearch, helpers as es_helpers


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
