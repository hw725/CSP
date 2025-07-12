"""토크나이저 모듈 - 다양한 언어의 토큰화 지원"""
import re
from typing import List, Optional
import logging

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseTokenizer:
    """기본 토크나이저 클래스"""
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할"""
        raise NotImplementedError

class SimpleTokenizer(BaseTokenizer):
    """간단한 공백 기반 토크나이저"""
    
    def tokenize(self, text: str) -> List[str]:
        """공백 기준으로 토큰화"""
        if not text or not text.strip():
            return []
        return text.strip().split()

class ChineseTokenizer(BaseTokenizer):
    """중국어 토크나이저 (jieba 사용)"""
    
    def __init__(self):
        if not JIEBA_AVAILABLE:
            logger.warning("jieba가 설치되지 않았습니다. SimpleTokenizer로 fallback합니다.")
            self.fallback = SimpleTokenizer()
        else:
            self.fallback = None
    
    def tokenize(self, text: str) -> List[str]:
        """jieba를 사용한 중국어 토큰화"""
        if not text or not text.strip():
            return []
        
        if self.fallback:
            return self.fallback.tokenize(text)
        
        try:
            return list(jieba.cut(text.strip()))
        except Exception as e:
            logger.error(f"jieba 토큰화 실패: {e}")
            return SimpleTokenizer().tokenize(text)

class KoreanTokenizer(BaseTokenizer):
    """한국어 토크나이저 (MeCab 사용)"""
    
    def __init__(self, dicdir: Optional[str] = None):
        self.mecab = None
        if MECAB_AVAILABLE:
            try:
                if dicdir:
                    self.mecab = MeCab.Tagger(f'-d {dicdir}')
                else:
                    self.mecab = MeCab.Tagger()
            except Exception as e:
                logger.error(f"MeCab 초기화 실패: {e}")
        
        if not self.mecab:
            logger.warning("MeCab을 사용할 수 없습니다. SimpleTokenizer로 fallback합니다.")
            self.fallback = SimpleTokenizer()
        else:
            self.fallback = None
    
    def tokenize(self, text: str) -> List[str]:
        """MeCab을 사용한 한국어 토큰화"""
        if not text or not text.strip():
            return []
        
        if self.fallback:
            return self.fallback.tokenize(text)
        
        try:
            parsed = self.mecab.parse(text.strip())
            tokens = []
            for line in parsed.split('\n'):
                if line.strip() and line != 'EOS':
                    token = line.split('\t')[0]
                    if token:
                        tokens.append(token)
            return tokens
        except Exception as e:
            logger.error(f"MeCab 토큰화 실패: {e}")
            return SimpleTokenizer().tokenize(text)

def get_tokenizer(language: str = 'auto', **kwargs) -> BaseTokenizer:
    """언어에 따른 적절한 토크나이저 반환"""
    if language == 'korean' or language == 'ko':
        return KoreanTokenizer(**kwargs)
    elif language == 'chinese' or language == 'zh':
        return ChineseTokenizer()
    else:
        return SimpleTokenizer()