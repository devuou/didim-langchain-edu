from elasticsearch import Elasticsearch
from app.core.config import settings


def get_es_client() -> Elasticsearch:
    """ES 클라이언트를 반환합니다. username/password 또는 비인증으로 연결."""
    kwargs: dict = {"hosts": [settings.ES_URL]}
    if settings.ES_USERNAME and settings.ES_PASSWORD:
        kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)
    return Elasticsearch(**kwargs)


# 모듈 레벨 싱글턴
es_client = get_es_client()
