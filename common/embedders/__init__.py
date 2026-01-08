"""임베더 모듈 패키지 - 간소화 버전"""

import logging

logger = logging.getLogger(__name__)

# 안전한 임포트
try:
    from .openai_embedder import compute_embeddings_with_cache as openai_embedder
    from .openai_embedder import compute_embeddings_with_cache
except ImportError as e:
    logger.error(f"OpenAI 임베더 임포트 실패: {e}")
    openai_embedder = None
    compute_embeddings_with_cache = None

try:
    from .bge import (
        compute_embeddings_with_cache as bge_embedder, 
        get_embed_func as bge_get_embed_func, 
        get_embed_func, 
        get_embedding_manager
    )
except ImportError as e:
    logger.error(f"BGE 임베더 임포트 실패: {e}")
    bge_embedder = None
    bge_get_embed_func = None
    get_embed_func = None
    get_embedding_manager = None

def get_embedder(name: str, device_id=None, model_name=None):
    """임베더 이름에 따라 함수 반환"""
    if name == "openai":
        if openai_embedder is None:
            raise ImportError("OpenAI 임베더가 임포트되지 않았습니다. openai 패키지 설치 필요")
        return openai_embedder
    elif name == "bge" or name is None:
        if bge_get_embed_func is None:
            raise ImportError("BGE 임베더가 임포트되지 않았습니다. FlagEmbedding 패키지 설치 필요")
        return bge_get_embed_func(device_id=device_id)
    else:
        raise ValueError(f"지원하지 않는 임베더: {name}. 지원: openai, bge")

__all__ = ['openai_embedder', 'bge_embedder', 'get_embedder', 'compute_embeddings_with_cache', 'get_embed_func']