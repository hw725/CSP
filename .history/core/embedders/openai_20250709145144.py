import logging
from typing import List, Callable, Optional
import numpy as np
import os

logger = logging.getLogger(__name__)

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI package successfully imported")
except ImportError as e:
    logger.warning(f"OpenAI package not found: {e}. OpenAI embedder will not be available.")
    OPENAI_AVAILABLE = False
    openai = None
    OpenAI = None

def compute_embeddings_with_cache(
    texts: List[str], 
    model: str = "text-embedding-ada-002",
    api_key: Optional[str] = None
) -> np.ndarray:
    """OpenAI API를 사용한 임베딩 계산"""
    
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI package is not available")
    
    # API 키 확인
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    try:
        client = OpenAI(api_key=api_key)
        
        embeddings = []
        for text in texts:
            if not text.strip():
                # 빈 텍스트에 대해서는 영벡터 반환
                embeddings.append(np.zeros(1536))  # ada-002 차원
                continue
                
            response = client.embeddings.create(
                input=text,
                model=model
            )
            embeddings.append(np.array(response.data[0].embedding))
        
        return np.array(embeddings)
        
    except Exception as e:
        logger.error(f"OpenAI API 호출 실패: {e}")
        raise
