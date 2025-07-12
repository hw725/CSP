import logging
from typing import List, Callable, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import FlagModel
    BGE_AVAILABLE = True
    logger.info("FlagEmbedding package successfully imported")
except ImportError as e:
    logger.warning(f"FlagEmbedding package not found: {e}. BGE embedder will not be available.")
    BGE_AVAILABLE = False
    FlagModel = None

class BaseEmbedder:
    """기본 임베더 추상 클래스"""
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

class BGEEmbedder(BaseEmbedder):
    """BGE 임베더 구현"""
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        if not BGE_AVAILABLE:
            raise ImportError("FlagEmbedding package is not available")
        
        self.model = FlagModel(model_name, use_fp16=False)
        self.device = device
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"BGE embedding failed: {e}")
            raise

def get_embed_func(device_id: Optional[int] = None, model_name: str = "BAAI/bge-m3") -> Optional[Callable]:
    """BGE 임베딩 함수 반환"""
    if not BGE_AVAILABLE:
        logger.warning("BGE embedder not available")
        return None
    
    try:
        device = f"cuda:{device_id}" if device_id is not None else "cpu"
        embedder = BGEEmbedder(model_name=model_name, device=device)
        
        def embed_func(texts: List[str]) -> np.ndarray:
            return embedder.embed(texts)
        
        return embed_func
    except Exception as e:
        logger.error(f"BGE embedder creation failed: {e}")
        return None

def compute_embeddings_with_cache(texts: List[str], model_name: str = "BAAI/bge-m3") -> np.ndarray:
    """BGE 임베딩 계산 (캐시 지원)"""
    if not BGE_AVAILABLE:
        raise ImportError("FlagEmbedding package is not available")
    
    embedder = BGEEmbedder(model_name=model_name)
    return embedder.embed(texts)
