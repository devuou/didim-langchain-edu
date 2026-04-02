"""SEC 10-K 공시 문서를 파싱·청킹·임베딩하여 Elasticsearch에 적재하는 오프라인 파이프라인.

[실행 방법]
    uv run python scripts/ingest_10k.py

[전체 흐름]
    SEC EDGAR 다운로드 → HTML 텍스트 추출 → 섹션 분리 → 토큰 청킹
    → OpenAI 임베딩 → Elasticsearch bulk upsert
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

import tiktoken


# ─── 텍스트 청킹 ─────────────────────────────────────────────────────────────
# 청킹 이유: LLM은 한 번에 처리할 수 있는 토큰 수의 상한선이 있는데 이를 context window라고 한다.
# 청킹 전 원문을 LLM에게 그대로 던지면 이를 초과하므로 청킹해두었다가 관련성이 높은 일부 청크만 전달한다.
# tiktoken: OpenAI 모델과 동일한 토크나이저(cl100k_base)를 사용해
# LLM이 실제로 몇 토큰으로 인식하는지 기준으로 청크 크기를 제어한다.
# 문자 수(characters) 기준이 아닌 토큰 수 기준이므로 context window 초과를 방지할 수 있다.

_ENCODER = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> list[str]:
    """텍스트를 max_tokens 크기의 청크로 분할한다. 인접 청크 사이에 overlap 토큰을 공유한다.

    overlap을 두는 이유: 청크 경계에서 잘린 문장의 문맥이 다음 청크 앞부분에도
    포함되도록 해 검색 시 문맥 손실을 최소화한다.

    예) max_tokens=512, overlap=50 이면
        청크0: 토큰[0:512]
        청크1: 토큰[462:974]  ← 앞 청크와 50토큰 겹침
        청크2: 토큰[924:1436]
    """
    tokens = _ENCODER.encode(text)          # 텍스트 → 토큰 ID 목록
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_ENCODER.decode(chunk_tokens))  # 토큰 ID → 텍스트로 복원
        if end == len(tokens):
            break
        start += max_tokens - overlap       # 다음 청크 시작 = 현재 끝 - overlap
    return chunks


# ─── 섹션 추출 ───────────────────────────────────────────────────────────────
# SEC 10-K에서 검색에 유용한 3개 섹션만 추출한다.
#   - item1  (Business):     사업 구조, 제품/서비스 설명
#   - item1a (Risk Factors): 투자·사업 리스크 (가장 분량이 많음)
#   - item7  (MD&A):         경영진의 재무 성과 분석
#
# [주의] 10-K HTML에는 앞부분 목차(TOC)와 뒷부분 본문에 동일 헤더가 2번 등장한다.
#   목차 예: "Item 1. Business ........... 1"  ← 페이지 번호가 붙음
#   본문 예: "Item 1.    Business\nCompany Background\n..."
# 따라서 re.finditer()로 모든 매칭을 찾고 마지막 매칭(=본문)을 사용한다.
#
# [주의] 종료 패턴에 "." (마침표)를 포함하는 이유:
#   본문 안에 "see Item 8 of this Form 10-K" 같은 교차 참조가 있어
#   단순 r"Item\s+8\b" 패턴이 실제 섹션 헤더가 아닌 교차 참조에 매칭되어
#   섹션이 조기 종료되는 문제를 방지하기 위함이다.
#   실제 섹션 헤더: "Item 8.  Financial Statements"  ← 마침표 있음
#   교차 참조:      "see Item 8 of this Form 10-K"   ← 마침표 없음

# 섹션별 (시작 패턴, 종료 패턴) 쌍
_SECTION_CONFIG: dict[str, tuple[str, str]] = {
    "item1":  (r"Item\s+1\.\s+Business",        r"Item\s+1A\."),
    "item1a": (r"Item\s+1A\.\s+Risk\s+Factors", r"Item\s+(?:1B|2)\."),
    "item7":  (r"Item\s+7\.\s+",                r"Item\s+(?:7A|8)\."),
}


def extract_sections(text: str) -> dict[str, str]:
    """10-K 텍스트에서 Item 1 / 1A / 7 섹션을 추출한다.

    반환값: {"item1": "...", "item1a": "...", "item7": "..."}
    섹션을 찾지 못하면 해당 key의 value는 빈 문자열.
    """
    result: dict[str, str] = {}
    for key, (start_pat, stop_pat) in _SECTION_CONFIG.items():
        # 시작 패턴과 일치하는 위치를 모두 찾는다 (목차 + 본문)
        matches = list(re.finditer(start_pat, text, re.IGNORECASE))
        if not matches:
            result[key] = ""
            continue

        # 마지막 매칭 = 실제 본문 섹션 (첫 번째는 TOC)
        m_start = matches[-1]
        body_start = m_start.end()          # 섹션 헤더 직후부터 본문 시작

        # 다음 섹션 헤더가 나오는 위치까지를 본문 끝으로 삼는다
        m_stop = re.search(stop_pat, text[body_start:], re.IGNORECASE)
        body_end = body_start + m_stop.start() if m_stop else len(text)

        result[key] = text[body_start:body_end].strip()
    return result


# ─── ES 인덱스 ───────────────────────────────────────────────────────────────

def build_index_name(prefix: str) -> str:
    """ES 인덱스명을 생성한다. 예) prefix="dev" → "dev-10k-docs" """
    return f"{prefix}-10k-docs"


def build_index_mapping() -> dict:
    """ES 인덱스 매핑(스키마)을 반환한다.

    text 필드:      BM25 키워드 검색용 (역색인)
    embedding 필드: kNN 벡터 검색용 (1536차원 dense_vector)
    나머지 필드:    필터링·메타데이터용 keyword
    """
    return {
        "mappings": {
            "properties": {
                "ticker":       {"type": "keyword"},   # 종목 코드 (AAPL, MSFT 등)
                "section":      {"type": "keyword"},   # 섹션 구분 (item1, item1a, item7)
                "fiscal_year":  {"type": "keyword"},   # 회계연도 (메타데이터)
                "text":         {"type": "text"},      # 청크 원문 — BM25 검색 대상
                "embedding":    {"type": "dense_vector", "dims": 1536},  # kNN 검색 대상
                "chunk_id":     {"type": "keyword"},   # 문서 고유 ID (upsert 키)
                "ingested_at":  {"type": "date"},      # 적재 시각
            }
        }
    }


def ensure_index(es_client, index_name: str) -> None:
    """인덱스가 없으면 생성한다. 이미 있으면 그대로 유지한다 (재실행 안전)."""
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
    """섹션 텍스트를 청킹하고 ES에 저장할 문서 dict를 yield한다.

    chunk_id 형식: "{ticker}_{section}_{순번:04d}"
    예) "AAPL_item1a_0003"
    → ES에서 이 값을 _id로 사용하므로 같은 문서를 재적재해도 중복 없이 덮어쓴다(upsert).
    """
    for section_key, section_text in sections.items():
        if not section_text:
            continue  # 섹션을 찾지 못한 경우 건너뜀
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
# sec-edgar-downloader: SEC EDGAR 공식 API를 통해 기업 공시 파일을 다운로드하는 라이브러리.
# download_details=True 로 설정해야 primary-document.html 파일이 생성된다.
# (False 로 하면 full-submission.txt 하나만 생성되어 HTML 파일이 없음)
#
# 다운로드 경로: {download_dir}/sec-edgar-filings/{ticker}/10-K/{접수번호}/primary-document.html
# 예) /tmp/tmpXXXX/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/primary-document.html
#
# [참고] download_dir로 tempfile.TemporaryDirectory()를 사용하므로
#        main() 실행이 끝나면 다운로드 파일은 자동 삭제된다.
#        파일을 직접 보려면 main()의 tmpdir를 고정 경로로 변경할 것.

import tempfile
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup


TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def download_latest_10k_text(ticker: str, download_dir: str) -> str:
    """SEC EDGAR에서 가장 최근 10-K를 다운로드하고 순수 텍스트로 변환해 반환한다.

    HTML → BeautifulSoup.get_text() 로 태그를 제거한 평문 텍스트를 반환한다.
    """
    # SEC EDGAR에 요청할 때 사용하는 식별 정보 (이름, 이메일은 임의값 가능)
    dl = Downloader("student-project", "student@example.com", download_dir)
    # limit=1: 가장 최신 제출 1건만 다운로드
    dl.get("10-K", ticker, limit=1, download_details=True)

    # 다운로드된 디렉토리 탐색 (접수번호 = 디렉토리명)
    base = Path(download_dir) / "sec-edgar-filings" / ticker / "10-K"
    filing_dirs = sorted(base.iterdir())
    if not filing_dirs:
        raise FileNotFoundError(f"{ticker} 10-K 다운로드 실패")
    filing_dir = filing_dirs[-1]  # 가장 최신 접수 디렉토리

    # HTML 파일 찾기 (.htm 또는 .html)
    htm_files = list(filing_dir.glob("*.htm")) + list(filing_dir.glob("*.html"))
    if not htm_files:
        raise FileNotFoundError(f"{ticker}: .htm 파일을 찾을 수 없음: {filing_dir}")

    # HTML 파싱 후 태그 제거 → 줄바꿈 기준의 평문 텍스트
    html = htm_files[0].read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


# ─── 임베딩 ──────────────────────────────────────────────────────────────────
# OpenAI text-embedding-3-small 모델로 텍스트를 1536차원 벡터로 변환한다.
# 이 벡터가 ES의 dense_vector 필드에 저장되어 kNN 검색에 사용된다.
#
# [배치 처리] API 호출 횟수를 줄이기 위해 100개씩 묶어 한 번에 요청한다.
# [재시도 로직] Rate Limit / Timeout 발생 시 최대 3회, 지수 백오프(5→10→15초)로 재시도한다.

import time
from openai import OpenAI as _OpenAI, RateLimitError, APITimeoutError, APIError

_openai_client: _OpenAI | None = None  # 모듈 레벨 싱글톤

_BATCH_SIZE = 100   # 1회 API 호출당 처리할 텍스트 수
_MAX_RETRIES = 3    # 최대 재시도 횟수
_RETRY_DELAY = 5    # 기본 대기 시간(초), 재시도마다 배수 증가 (5 → 10 → 15)


def get_openai_client() -> _OpenAI:
    """OpenAI 클라이언트 싱글톤을 반환한다. 최초 호출 시에만 초기화된다."""
    global _openai_client
    if _openai_client is None:
        # OPENAI_API_KEY 환경변수를 자동으로 읽음 (settings 통해 .env에서 로드됨)
        _openai_client = _OpenAI()
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 목록을 임베딩 벡터 목록으로 변환한다. 배치 처리 + 지수 백오프 재시도.

    반환값: 입력 texts와 동일한 순서의 1536차원 float 벡터 목록
    """
    client = get_openai_client()
    all_vectors: list[list[float]] = []

    # _BATCH_SIZE(100)개씩 나눠서 API 호출
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small",  # 1536차원 벡터 생성
                )
                # resp.data는 입력 순서와 동일한 Embedding 객체 목록
                all_vectors.extend([item.embedding for item in resp.data])
                print(f"  임베딩 생성: {len(all_vectors)}/{len(texts)}")
                break  # 성공 시 재시도 루프 탈출
            except RateLimitError:
                # API 사용량 초과 → 대기 후 재시도
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
# ES bulk API로 여러 문서를 한 번에 적재한다.
# "_id"에 chunk_id를 지정해 같은 chunk_id로 재실행해도 중복 생성 없이 덮어쓴다(upsert).

from datetime import datetime, timezone
from elasticsearch import Elasticsearch, helpers as es_helpers


def ingest_ticker(
    es_client: Elasticsearch,
    index_name: str,
    ticker: str,
    fiscal_year: str,
    download_dir: str,
) -> int:
    """티커 1개의 전체 파이프라인을 실행한다. 적재된 문서 수를 반환한다.

    파이프라인: 다운로드 → 섹션 추출 → 청킹 → 임베딩 → ES 적재
    """
    # 1. SEC EDGAR에서 최신 10-K HTML을 다운로드하고 순수 텍스트로 변환
    print(f"\n[{ticker}] 다운로드 중...")
    text = download_latest_10k_text(ticker, download_dir)

    # 2. 전체 텍스트에서 item1 / item1a / item7 섹션만 추출
    print(f"[{ticker}] 섹션 추출 중...")
    sections = extract_sections(text)

    # 3. 섹션 텍스트를 512토큰 청크로 분할하고 ES 문서 dict 목록 생성
    docs = list(build_docs(ticker, fiscal_year, sections))
    if not docs:
        print(f"[{ticker}] 추출된 청크 없음 — 건너뜀")
        return 0

    # 4. 각 청크 텍스트를 OpenAI API로 1536차원 임베딩 벡터로 변환
    print(f"[{ticker}] 임베딩 중... ({len(docs)}개 청크)")
    texts = [d["text"] for d in docs]
    vectors = embed_texts(texts)  # docs와 동일한 순서의 벡터 목록 반환

    # 5. 문서 dict에 임베딩 벡터와 적재 시각을 추가하여 ES bulk upsert
    ingested_at = datetime.now(timezone.utc).isoformat()
    actions = [
        {
            "_index": index_name,
            "_id": doc["chunk_id"],  # chunk_id를 ES 문서 ID로 사용 → 재실행 시 중복 방지
            "_source": {**doc, "embedding": vec, "ingested_at": ingested_at},
        }
        for doc, vec in zip(docs, vectors)
    ]

    success, _ = es_helpers.bulk(es_client, actions)
    print(f"[{ticker}] 적재 완료: {success}개")
    return success


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main() -> None:
    """파이프라인 진입점.
    .env의 ES 접속 정보를 읽어 전체 적재를 실행한다."""
    from app.core.config import settings

    fiscal_year = "2024"  # 적재 메타데이터용 (현재 검색 필터에는 미사용)

    # ES 클라이언트 초기화 (.env의 ES_URL, ES_USERNAME, ES_PASSWORD 사용)
    es_kwargs: dict = {"hosts": [settings.ES_URL]}
    if settings.ES_USERNAME and settings.ES_PASSWORD:
        es_kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)
    es = Elasticsearch(**es_kwargs)
    prefix = settings.ES_INDEX_PREFIX

    # 인덱스 준비 (없으면 생성, 있으면 재사용)
    index_name = build_index_name(prefix)
    ensure_index(es, index_name)

    # 임시 디렉토리에 다운로드 → 처리 완료 후 자동 삭제
    with tempfile.TemporaryDirectory() as tmpdir:
        total = 0
        for ticker in TICKERS:
            total += ingest_ticker(es, index_name, ticker, fiscal_year, tmpdir)

    print(f"\n완료: 총 {total}개 문서 적재")


if __name__ == "__main__":
    main()
