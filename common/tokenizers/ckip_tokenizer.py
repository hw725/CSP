"""
CKIP 토크나이저 공통 모듈
중국어 전문 형태소 분석기
"""

import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class CkipTokenizer:
    """CKIP 토크나이저 래퍼 클래스"""
    
    def __init__(self, device: int = -1):
        """
        Args:
            device: -1 for CPU, >=0 for GPU device ID
        """
        self.device = device
        self.word_segmenter = None
        self.pos_tagger = None
        self._initialized = False
    
    def _initialize(self):
        """지연 초기화"""
        if self._initialized:
            return
            
        try:
            from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
            
            logger.info(f"CKIP 모델 초기화 중... (device: {self.device})")
            
            self.word_segmenter = CkipWordSegmenter(device=self.device)
            self.pos_tagger = CkipPosTagger(device=self.device)
            
            logger.info("CKIP 모델 초기화 완료")
            self._initialized = True
            
        except ImportError:
            logger.error("ckip-transformers 패키지가 설치되지 않음")
            raise
        except Exception as e:
            logger.error(f"CKIP 모델 초기화 실패: {e}")
            raise
    
    def tokenize(self, texts: List[str]) -> List[List[str]]:
        """텍스트들을 토큰화"""
        self._initialize()
        
        try:
            result = self.word_segmenter(texts)
            return result
        except Exception as e:
            logger.error(f"CKIP 토큰화 실패: {e}")
            return [text.split() for text in texts]  # 폴백: 공백 분할
    
    def pos_tag(self, texts: List[str]) -> List[List[tuple]]:
        """품사 태깅"""
        self._initialize()
        
        try:
            tokens = self.tokenize(texts)
            pos_results = self.pos_tagger(tokens)
            
            # (단어, 품사) 튜플 리스트로 변환
            result = []
            for token_list, pos_list in zip(tokens, pos_results):
                tagged = list(zip(token_list, pos_list))
                result.append(tagged)
            
            return result
        except Exception as e:
            logger.error(f"CKIP 품사 태깅 실패: {e}")
            return [[(token, 'UNK') for token in text.split()] for text in texts]

# 전역 인스턴스 (싱글톤)
_ckip_tokenizer = None

def get_ckip_tokenizer(device: int = -1) -> CkipTokenizer:
    """CKIP 토크나이저 싱글톤 인스턴스 반환"""
    global _ckip_tokenizer
    
    if _ckip_tokenizer is None:
        _ckip_tokenizer = CkipTokenizer(device=device)
    
    return _ckip_tokenizer

def ckip_tokenize(texts: List[str], device: int = -1) -> List[List[str]]:
    """CKIP 토큰화 편의 함수"""
    tokenizer = get_ckip_tokenizer(device=device)
    return tokenizer.tokenize(texts)

def ckip_pos_tag(texts: List[str], device: int = -1) -> List[List[tuple]]:
    """CKIP 품사 태깅 편의 함수"""
    tokenizer = get_ckip_tokenizer(device=device)
    return tokenizer.pos_tag(texts)
