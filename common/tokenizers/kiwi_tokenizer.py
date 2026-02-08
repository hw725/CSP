"""
Kiwipiepy 한국어 토크나이저 모듈
한글 형태소 분석 전용
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import re

try:
    from common.disk_cache import DiskCache
except ImportError:
    try:
        from disk_cache import DiskCache
    except ImportError:
        DiskCache = None

logger = logging.getLogger(__name__)

# analyze() 결과 캐시 (morphs, pos, extract_particles 모두 이 결과에서 파생)
_kiwi_analyze_cache = DiskCache("tokenizer_kiwi_analyze", save_every=200) if DiskCache else None
_kiwi_sentences_cache = DiskCache("tokenizer_kiwi_sentences", save_every=200) if DiskCache else None

class KiwipieTokenizer:
    """Kiwipiepy 토크나이저 래퍼 클래스"""

    def __init__(self, model_type: str = "base"):
        """
        Args:
            model_type: 사용할 모델 타입 ('base', 'large' 등)
        """
        self.model_type = model_type
        self.kiwi = None
        self._initialized = False

    def _initialize(self):
        """지연 초기화"""
        if self._initialized:
            return

        try:
            from kiwipiepy import Kiwi

            # 모델 타입에 따른 초기화
            if self.model_type == "large":
                self.kiwi = Kiwi(model_type="knlm")  # Korean Neural Language Model
            else:
                self.kiwi = Kiwi()  # 기본 모델

            logger.info(f"Kiwipiepy {self.model_type} 모델 초기화 완료")
            self._initialized = True

        except ImportError as e:
            logger.error(f"Kiwipiepy 설치되지 않음: {e}")
            raise
        except Exception as e:
            logger.error(f"Kiwipiepy 초기화 실패: {e}")
            raise

    def _analyze_cached(self, text: str):
        """analyze() 결과를 캐싱하여 반환. morphs/pos/extract_particles에서 공유."""
        if _kiwi_analyze_cache is not None:
            cached = _kiwi_analyze_cache.get(text)
            if cached is not None:
                return cached
        self._initialize()
        result = self.kiwi.analyze(text)
        # (token, pos) 튜플 리스트로 정규화하여 저장
        tokens = [(token, pos_tag) for token, pos_tag, _, _ in result[0][0]]
        if _kiwi_analyze_cache is not None:
            _kiwi_analyze_cache.put(text, tokens)
        return tokens

    def morphs(self, text: str) -> List[str]:
        """형태소 분석 (단어 단위)"""
        try:
            tokens = self._analyze_cached(text)
            return [token for token, _ in tokens]
        except Exception as e:
            logger.error(f"Kiwipiepy 형태소 분석 실패: {e}")
            return text.split()

    def pos(self, text: str) -> List[Tuple[str, str]]:
        """품사 태깅"""
        try:
            return self._analyze_cached(text)
        except Exception as e:
            logger.error(f"Kiwipiepy 품사 태깅 실패: {e}")
            return [(word, "UNK") for word in text.split()]

    def tokenize(self, text: str, return_pos: bool = False) -> List[str]:
        """토큰화 (옵션으로 품사 정보 포함)"""
        if return_pos:
            return self.pos(text)
        else:
            return self.morphs(text)

    def split_sentences(self, text: str) -> List[str]:
        """문장 분할"""
        if _kiwi_sentences_cache is not None:
            cached = _kiwi_sentences_cache.get(text)
            if cached is not None:
                return cached
        self._initialize()
        try:
            sentences = self.kiwi.split_into_sents(text)
            result = [s.text for s in sentences]
            if _kiwi_sentences_cache is not None:
                _kiwi_sentences_cache.put(text, result)
            return result
        except Exception as e:
            logger.error(f"Kiwipiepy 문장 분할 실패: {e}")
            return re.split(r"[.!?]\s+", text)

    def extract_particles(self, text: str) -> List[Tuple[str, str, int]]:
        """조사 및 어미 추출 (_analyze_cached 결과에서 파생)"""
        particles = []
        try:
            tokens = self._analyze_cached(text)
            position = 0
            for token, pos_tag in tokens:
                if pos_tag in [
                    "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC",
                ]:
                    particles.append((token, "조사", position))
                elif pos_tag in ["EP", "EF", "EC", "ETN", "ETM"]:
                    particles.append((token, "어미", position))
                position += len(token)
        except Exception as e:
            logger.error(f"Kiwipiepy 조사/어미 추출 실패: {e}")
        return particles

# 전역 인스턴스들
_kiwi_tokenizers = {}

def get_kiwi_tokenizer(model_type: str = "base") -> KiwipieTokenizer:
    """Kiwipiepy 토크나이저 싱글톤 인스턴스 반환"""
    global _kiwi_tokenizers

    if model_type not in _kiwi_tokenizers:
        _kiwi_tokenizers[model_type] = KiwipieTokenizer(model_type=model_type)

    return _kiwi_tokenizers[model_type]

def kiwi_morphs(text: str, model_type: str = "base") -> List[str]:
    """Kiwipiepy 형태소 분석 편의 함수"""
    tokenizer = get_kiwi_tokenizer(model_type=model_type)
    return tokenizer.morphs(text)

def kiwi_pos(text: str, model_type: str = "base") -> List[Tuple[str, str]]:
    """Kiwipiepy 품사 태깅 편의 함수"""
    tokenizer = get_kiwi_tokenizer(model_type=model_type)
    return tokenizer.pos(text)

def kiwi_extract_particles(
    text: str, model_type: str = "base"
) -> List[Tuple[str, str, int]]:
    """Kiwipiepy 조사/어미 추출 편의 함수"""
    tokenizer = get_kiwi_tokenizer(model_type=model_type)
    return tokenizer.extract_particles(text)
