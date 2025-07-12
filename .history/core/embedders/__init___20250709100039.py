"""SA 임베더 모듈 패키지 - 의존성 및 함수 호출 에러 방지 철저 체크"""

import logging

logger = logging.getLogger(__name__)

# BaseEmbedder 클래스 정의
class BaseEmbedder:
    """임베더 기본 클래스"""
    def __init__(self, name: str):
        self.name = name
    
    def encode(self, texts: list) -> list:
        """텍스트 리스트를 임베딩으로 변환"""
        raise NotImplementedError("Subclasses must implement encode method")

class BGEEmbedder(BaseEmbedder):
    """BGE 임베더 클래스"""
    def __init__(self, device_id=None):
        super().__init__("bge")
        self.device_id = device_id
        self._embed_func = None
    
    def encode(self, texts: list) -> list:
        if self._embed_func is None:
            from .bge import get_embed_func
            self._embed_func = get_embed_func(self.device_id)
        return self._embed_func(texts)

# 기존 함수들 import
try:
    from .bge import get_embed_func as bge_get_embed_func
    from .bge import compute_embeddings_with_cache as bge_embedder
except ImportError as e:
    logger.warning(f"BGE embedder import failed: {e}")
    bge_get_embed_func = None
    bge_embedder = None

try:
    from .openai import compute_embeddings_with_cache as openai_embedder
except ImportError as e:
    logger.warning(f"OpenAI embedder import failed: {e}")
    openai_embedder = None

def get_embedder(name: str, device_id=None, model_name=None, openai_api_key=None):
    """임베더 이름에 따라 함수 반환"""
    if name == "openai":
        if openai_embedder is None:
            raise ImportError("OpenAI embedder is not available. Check logs for details.")
        return lambda texts: openai_embedder(texts, model_name or "text-embedding-3-large", openai_api_key)
    elif name == "bge" or name is None:
        if bge_get_embed_func is None:
            raise ImportError("BGE embedder is not available. Check logs for details.")
        return bge_get_embed_func(device_id=device_id)
    else:
        raise ValueError(f"지원하지 않는 임베더: {name}. 지원: openai, bge")

__all__ = [
    'get_embedder', 'bge_get_embed_func', 'bge_embedder', 'openai_embedder',
    'BaseEmbedder', 'BGEEmbedder'
]