"""SA 임베더 모듈 패키지 - 의존성 및 함수 호출 에러 방지 철저 체크"""

import importlib
import logging

logger = logging.getLogger(__name__)

from .bge import get_embed_func as bge_get_embed_func
from .bge import compute_embeddings_with_cache as bge_embedder
from .openai import compute_embeddings_with_cache as openai_embedder
from .bge import BaseEmbedder, BGEEmbedder  # ← 이 부분만 남기세요

def get_embedder(name: str, device_id=None, model_name=None, openai_api_key=None):
    """임베더 이름에 따라 함수 반환 (openai/bge, device_id/model_name 지정 가능, 에러 방지)"""
    if name == "openai":
        return openai_embedder
    elif name == "bge" or name is None:  # 기본값은 bge
        return bge_get_embed_func(device_id=device_id)
    else:
        raise ValueError(f"지원하지 않는 임베더: {name}. 지원: openai, bge")

__all__ = [
    'get_embedder', 'bge_get_embed_func', 'bge_embedder', 'openai_embedder',
    'BaseEmbedder', 'BGEEmbedder'
]