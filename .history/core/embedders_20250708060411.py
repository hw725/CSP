"""임베딩 모듈 - 다양한 임베딩 모델 지원"""
import numpy as np
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from FlagEmbedding import FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class BaseEmbedder:
    """기본 임베더 클래스"""
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        raise NotImplementedError
    
    def is_available(self) -> bool:
        """임베더 사용 가능 여부"""
        return True

class BGEEmbedder(BaseEmbedder):
    """BGE (FlagEmbedding) 임베더"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = None
        
        if FLAG_EMBEDDING_AVAILABLE:
            try:
                self.model = FlagModel(model_name, use_fp16=True)
                logger.info(f"BGE 모델 로드 성공: {model_name}")
            except Exception as e:
                logger.error(f"BGE 모델 로드 실패: {e}")
        else:
            logger.warning("FlagEmbedding이 설치되지 않았습니다.")
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """BGE 모델을 사용한 임베딩"""
        if not self.is_available():
            raise RuntimeError("BGE 모델을 사용할 수 없습니다.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"BGE 임베딩 실패: {e}")
            raise

class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer 임베더"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"SentenceTransformer 모델 로드 성공: {model_name}")
            except Exception as e:
                logger.error(f"SentenceTransformer 모델 로드 실패: {e}")
        else:
            logger.warning("sentence-transformers가 설치되지 않았습니다.")
    
    def is_available(self) -> bool:
        return self.model is not None
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """SentenceTransformer를 사용한 임베딩"""
        if not self.is_available():
            raise RuntimeError("SentenceTransformer 모델을 사용할 수 없습니다.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(texts)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"SentenceTransformer 임베딩 실패: {e}")
            raise

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI 임베더"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = None
        
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"OpenAI 클라이언트 초기화 성공: {model_name}")
            except Exception as e:
                logger.error(f"OpenAI 클라이언트 초기화 실패: {e}")
        else:
            logger.warning("OpenAI API 키가 없거나 openai 패키지가 설치되지 않았습니다.")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """OpenAI API를 사용한 임베딩"""
        if not self.is_available():
            raise RuntimeError("OpenAI 클라이언트를 사용할 수 없습니다.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"OpenAI 임베딩 실패: {e}")
            raise

def get_embedder(embedder_type: str = "bge", **kwargs) -> BaseEmbedder:
    """임베더 타입에 따른 적절한 임베더 반환"""
    if embedder_type.lower() == "bge":
        return BGEEmbedder(**kwargs)
    elif embedder_type.lower() == "sentence_transformer":
        return SentenceTransformerEmbedder(**kwargs)
    elif embedder_type.lower() == "openai":
        return OpenAIEmbedder(**kwargs)
    else:
        logger.warning(f"알 수 없는 임베더 타입: {embedder_type}, BGE로 fallback")
        return BGEEmbedder(**kwargs)
