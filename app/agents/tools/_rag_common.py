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
