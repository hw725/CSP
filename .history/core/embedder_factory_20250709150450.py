import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

def get_embedder(name: str, **kwargs) -> Optional[Callable]:
    """임베더 팩토리 - 사용 가능한 임베더만 반환"""
    
    if name == "bge":
        try:
            from .embedders.bge import get_embed_func, BGE_AVAILABLE
            if not BGE_AVAILABLE:
                logger.warning("BGE embedder not available - FlagEmbedding package missing")
                return None
            
            device_id = kwargs.get('device_id')
            model_name = kwargs.get('model_name', 'BAAI/bge-m3')
            
            embed_func = get_embed_func(device_id=device_id, model_name=model_name)
            if embed_func is None:
                logger.error("Failed to create BGE embedder")
                return None
            
            logger.info(f"BGE embedder initialized successfully (device_id={device_id})")
            return embed_func
            
        except Exception as e:
            logger.error(f"BGE embedder initialization failed: {e}")
            return None
    
    elif name == "openai":
        try:
            from .embedders.openai import compute_embeddings_with_cache, OPENAI_AVAILABLE
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI embedder not available")
                return None
            
            # API 키 확인
            import os
            api_key = kwargs.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                return None
            
            openai_model = kwargs.get('openai_model', 'text-embedding-ada-002')
            
            def openai_embedder(texts):
                return compute_embeddings_with_cache(texts, model=openai_model, api_key=api_key)
            
            logger.info("OpenAI embedder initialized successfully")
            return openai_embedder
            
        except Exception as e:
            logger.error(f"OpenAI embedder initialization failed: {e}")
            return None
    
    else:
        logger.error(f"Unknown embedder: {name}")
        return None
