"""Common Embedders Package

BGE, OpenAI, Gemini 임베더 통합 인터페이스
"""

from typing import Callable, List
import logging

logger = logging.getLogger(__name__)


def get_embedder(name: str = "bge", device_id: int = None) -> Callable[[List[str]], List]:
    """임베더 함수 반환
    
    Args:
        name: 임베더 이름 ('bge', 'openai', 'gemini')
        device_id: GPU 디바이스 ID (BGE용)
    
    Returns:
        임베딩 함수 (texts: List[str]) -> List[embedding]
    """
    name = name.lower()
    
    if name == "bge" or name == "bge-m3":
        from common.embedders.bge import get_embed_func
        return get_embed_func(device_id=device_id)
    
    elif name == "openai":
        from common.embedders.openai_embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder()
        return embedder.compute_embeddings
    
    elif name == "gemini":
        from common.embedders.gemini_embedder import GeminiEmbedder
        embedder = GeminiEmbedder()
        return embedder.compute_embeddings
    
    else:
        logger.warning(f"Unknown embedder '{name}', falling back to BGE")
        from common.embedders.bge import get_embed_func
        return get_embed_func(device_id=device_id)


# 편의 함수들
def get_bge_embedder(device_id: int = None):
    return get_embedder("bge", device_id=device_id)


def get_openai_embedder():
    return get_embedder("openai")


def get_gemini_embedder():
    return get_embedder("gemini")
