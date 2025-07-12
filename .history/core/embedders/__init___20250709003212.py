"""SA 임베더 모듈 패키지 - 의존성 및 함수 호출 에러 방지 철저 체크"""

import logging

logger = logging.getLogger(__name__)

try:
    from .bge import get_embed_func as bge_get_embed_func
    BGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BGE embedder import failed: {e}")
    bge_get_embed_func = None
    BGE_AVAILABLE = False

try:
    from .openai import compute_embeddings_with_cache as openai_embedder
    OPENAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenAI embedder import failed: {e}")
    openai_embedder = None
    OPENAI_AVAILABLE = False

# BaseEmbedder와 BGEEmbedder 클래스 정의
class BaseEmbedder:
    """기본 임베더 인터페이스"""
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
    
    def embed(self, texts):
        raise NotImplementedError

class BGEEmbedder(BaseEmbedder):
    """BGE 임베더 구현"""
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        super().__init__(model_name, device)
        if BGE_AVAILABLE and bge_get_embed_func:
            self.embed_func = bge_get_embed_func(device_id=device)
        else:
            self.embed_func = None
    
    def embed(self, texts):
        if self.embed_func:
            return self.embed_func(texts)
        else:
            raise RuntimeError("BGE embedder is not available")

def get_embedder(name: str, device_id=None, model_name=None, openai_api_key=None):
    """임베더 이름에 따라 함수 반환 (openai/bge, device_id/model_name 지정 가능, 에러 방지)"""
    if name == "openai":
        if not OPENAI_AVAILABLE or openai_embedder is None:
            raise ValueError("OpenAI embedder is not available. Check logs for details.")
        return openai_embedder
    elif name == "bge" or name is None:  # 기본값은 bge
        if not BGE_AVAILABLE or bge_get_embed_func is None:
            raise ValueError("BGE embedder is not available. Check logs for details.")
        return bge_get_embed_func(device_id=device_id)
    else:
        raise ValueError(f"지원하지 않는 임베더: {name}. 지원: openai, bge")

__all__ = [
    'get_embedder', 'bge_get_embed_func', 'openai_embedder',
    'BaseEmbedder', 'BGEEmbedder', 'BGE_AVAILABLE', 'OPENAI_AVAILABLE'
]