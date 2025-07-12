import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

def get_embedder(name: str, **kwargs) -> Optional[Callable]:
    """임베더 팩토리 - 사용 가능한 임베더만 반환"""
    
    if name == "openai":
        try:
            from .embedders.openai import compute_embeddings_with_cache, OPENAI_AVAILABLE
            if OPENAI_AVAILABLE:
                # API 키 확인
                import os
                api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                    return None
                
                def openai_embedder(texts):
                    return compute_embeddings_with_cache(texts, api_key=api_key)
                return openai_embedder
            else:
                logger.warning("OpenAI package not available")
                return None
        except Exception as e:
            logger.error(f"OpenAI embedder initialization failed: {e}")
            return None
    
    elif name == "bge":
        try:
            from .embedders.bge import get_embed_func, BGE_AVAILABLE
            if BGE_AVAILABLE:
                return get_embed_func(**kwargs)
            else:
                logger.warning("BGE embedder not available")
                return None
        except Exception as e:
            logger.error(f"BGE embedder initialization failed: {e}")
            return None
    
    else:
        logger.error(f"Unknown embedder: {name}")
        return None
