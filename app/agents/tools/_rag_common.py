"""RAG 파이프라인에서 사용하는 ES 벡터 클라이언트, 임베딩, 리랭커 싱글톤.

모듈 로드 시점에 클라이언트를 생성하지 않고, 처음 호출될 때 한 번만 생성(lazy init)한다.
여러 도구 함수에서 동일 클라이언트를 재사용하여 연결 비용을 줄인다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# TYPE_CHECKING 블록: 타입 힌트 전용 import.
# 런타임에는 실행되지 않아 무거운 라이브러리(elasticsearch, openai 등)를
# 모듈 로드 시점에 불러오지 않아도 된다.
if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from openai import OpenAI

# ─── 싱글톤 전역 변수 ──────────────────────────────────────────────────────────
# None으로 초기화 → 처음 get_*() 호출 시 생성(lazy init)
_es_client: "Elasticsearch | None" = None
_openai_client: "OpenAI | None" = None


def get_es_client() -> "Elasticsearch":
    """ES 클라이언트 싱글톤을 반환한다.

    최초 호출 시 settings에서 URL / 인증 정보를 읽어 생성한다.
    이후 호출은 이미 생성된 인스턴스를 재사용한다.
    """
    global _es_client
    if _es_client is None:
        from elasticsearch import Elasticsearch
        from app.core.config import settings
        kwargs: dict = {"hosts": [settings.ES_URL]}
        # basic_auth는 ES_USERNAME / ES_PASSWORD 모두 설정된 경우에만 전달
        if settings.ES_USERNAME and settings.ES_PASSWORD:
            kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)
        _es_client = Elasticsearch(**kwargs)
    return _es_client


def get_openai_client() -> "OpenAI":
    """OpenAI 클라이언트 싱글톤을 반환한다.

    임베딩 API 호출에 사용된다.
    """
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        from app.core.config import settings
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


def embed_query(query: str) -> list[float]:
    """쿼리 문자열을 1536차원 벡터로 변환한다.

    text-embedding-3-small 모델을 사용한다.
    ingest_10k.py의 embed_texts()와 동일 모델을 써야 kNN 검색이 올바르게 동작한다.
    """
    client = get_openai_client()
    resp = client.embeddings.create(input=[query], model="text-embedding-3-small")
    return resp.data[0].embedding


def rerank_hits(query: str, hits: list[dict]) -> list[dict] | None:
    """ES Inference API로 hits를 리랭킹한다.

    ES_RERANKER_INFERENCE_ID가 설정되지 않았거나 호출에 실패하면 None을 반환한다.
    None이면 호출 측에서 score 내림차순 정렬로 fallback해야 한다.

    ES Inference rerank API:
    - 입력: query(질문), input(텍스트 목록)
    - 출력: rerank 결과 목록 (index: 원본 hits의 인덱스, score: 관련도 점수)
    - 결과는 score 내림차순으로 이미 정렬되어 반환된다.
    """
    try:
        from app.core.config import settings
        inference_id = settings.ES_RERANKER_INFERENCE_ID
        if not inference_id or not hits:
            return None

        es = get_es_client()
        texts = [h["_source"]["text"] for h in hits]

        resp = es.inference.inference(
            task_type="rerank",
            inference_id=inference_id,
            body={"query": query, "input": texts},
        )

        # resp["rerank"]: [{"index": 2, "score": 0.94, ...}, ...]
        # index는 입력 texts/hits 목록의 인덱스와 대응됨
        ranked = resp.get("rerank", [])
        return [hits[item["index"]] for item in ranked]
    except Exception:
        return None


def format_hits(hits: list[dict]) -> str:
    """ES 검색 결과(hits)를 LLM이 읽기 좋은 문자열로 변환한다.

    각 hit의 section, ticker, text를 번호 목록 형식으로 이어 붙인다.
    결과가 없으면 안내 메시지를 반환한다.
    """
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
