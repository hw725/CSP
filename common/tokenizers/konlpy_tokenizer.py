"""
KoNLPy 한국어 토크나이저 공통 모듈
한국어 형태소 분석기들의 통합 인터페이스
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class KonlpyTokenizer:
    """KoNLPy 토크나이저 래퍼 클래스"""
    
    def __init__(self, analyzer: str = "Okt"):
        """
        Args:
            analyzer: 사용할 분석기 ('Okt', 'Komoran', 'Hannanum', 'Kkma', 'Mecab')
        """
        self.analyzer_name = analyzer
        self.analyzer = None
        self._initialized = False
    
    def _initialize(self):
        """지연 초기화"""
        if self._initialized:
            return
            
        try:
            if self.analyzer_name == "Okt":
                from konlpy.tag import Okt
                self.analyzer = Okt()
            elif self.analyzer_name == "Komoran":
                from konlpy.tag import Komoran
                self.analyzer = Komoran()
            elif self.analyzer_name == "Hannanum":
                from konlpy.tag import Hannanum
                self.analyzer = Hannanum()
            elif self.analyzer_name == "Kkma":
                from konlpy.tag import Kkma
                self.analyzer = Kkma()
            elif self.analyzer_name == "Mecab":
                from konlpy.tag import Mecab
                self.analyzer = Mecab()
            else:
                raise ValueError(f"지원하지 않는 분석기: {self.analyzer_name}")
            
            logger.info(f"KoNLPy {self.analyzer_name} 초기화 완료")
            self._initialized = True
            
        except ImportError as e:
            logger.error(f"KoNLPy 또는 {self.analyzer_name} 설치되지 않음: {e}")
            raise
        except Exception as e:
            logger.error(f"KoNLPy {self.analyzer_name} 초기화 실패: {e}")
            raise
    
    def morphs(self, text: str) -> List[str]:
        """형태소 분석 (단어 단위)"""
        self._initialize()
        
        try:
            return self.analyzer.morphs(text)
        except Exception as e:
            logger.error(f"KoNLPy 형태소 분석 실패: {e}")
            return text.split()  # 폴백: 공백 분할
    
    def pos(self, text: str) -> List[Tuple[str, str]]:
        """품사 태깅"""
        self._initialize()
        
        try:
            return self.analyzer.pos(text)
        except Exception as e:
            logger.error(f"KoNLPy 품사 태깅 실패: {e}")
            return [(word, 'UNK') for word in text.split()]
    
    def nouns(self, text: str) -> List[str]:
        """명사 추출"""
        self._initialize()
        
        try:
            return self.analyzer.nouns(text)
        except Exception as e:
            logger.error(f"KoNLPy 명사 추출 실패: {e}")
            return []

# 전역 인스턴스들 (분석기별)
_konlpy_tokenizers = {}

def get_konlpy_tokenizer(analyzer: str = "Okt") -> KonlpyTokenizer:
    """KoNLPy 토크나이저 싱글톤 인스턴스 반환"""
    global _konlpy_tokenizers
    
    if analyzer not in _konlpy_tokenizers:
        _konlpy_tokenizers[analyzer] = KonlpyTokenizer(analyzer=analyzer)
    
    return _konlpy_tokenizers[analyzer]

def konlpy_morphs(text: str, analyzer: str = "Okt") -> List[str]:
    """KoNLPy 형태소 분석 편의 함수"""
    tokenizer = get_konlpy_tokenizer(analyzer=analyzer)
    return tokenizer.morphs(text)

def konlpy_pos(text: str, analyzer: str = "Okt") -> List[Tuple[str, str]]:
    """KoNLPy 품사 태깅 편의 함수"""
    tokenizer = get_konlpy_tokenizer(analyzer=analyzer)
    return tokenizer.pos(text)

def konlpy_nouns(text: str, analyzer: str = "Okt") -> List[str]:
    """KoNLPy 명사 추출 편의 함수"""
    tokenizer = get_konlpy_tokenizer(analyzer=analyzer)
    return tokenizer.nouns(text)
