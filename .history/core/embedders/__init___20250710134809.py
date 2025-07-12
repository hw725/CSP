"""SA 임베더 모듈 패키지 - 의존성 및 함수 호출 에러 방지 철저 체크"""

import logging

logger = logging.getLogger(__name__)

from .bge import get_embed_func as bge_get_embed_func
from .openai import compute_embeddings_with_cache as openai_embedder

__all__ = [
    'bge_get_embed_func', 'openai_embedder'
]