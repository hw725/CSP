"""
Kiwipiepy 한국어 토크나이저 모듈
한글 형태소 분석 전용
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import re

logger = logging.getLogger(__name__)

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

    def morphs(self, text: str) -> List[str]:
        """형태소 분석 (단어 단위)"""
        self._initialize()

        try:
            result = self.kiwi.analyze(text)
            morphs = []
            for token, pos, _, _ in result[0][0]:
                morphs.append(token)
            return morphs
        except Exception as e:
            logger.error(f"Kiwipiepy 형태소 분석 실패: {e}")
            return text.split()  # 폴백: 공백 분할

    def pos(self, text: str) -> List[Tuple[str, str]]:
        """품사 태깅"""
        self._initialize()

        try:
            result = self.kiwi.analyze(text)
            pos_tags = []
            for token, pos, _, _ in result[0][0]:
                pos_tags.append((token, pos))
            return pos_tags
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
        self._initialize()

        try:
            sentences = self.kiwi.split_into_sents(text)
            return [s.text for s in sentences]
        except Exception as e:
            logger.error(f"Kiwipiepy 문장 분할 실패: {e}")
            # 폴백: 간단한 문장 분할
            return re.split(r"[.!?]\s+", text)

    def extract_particles(self, text: str) -> List[Tuple[str, str, int]]:
        """조사 및 어미 추출"""
        self._initialize()

        particles = []
        try:
            result = self.kiwi.analyze(text)
            position = 0

            for token, pos, _, _ in result[0][0]:
                # 조사와 어미 추출
                if pos in [
                    "JKS",
                    "JKC",
                    "JKG",
                    "JKO",
                    "JKB",
                    "JKV",
                    "JKQ",
                    "JX",
                    "JC",
                ]:  # 조사
                    particles.append((token, "조사", position))
                elif pos in ["EP", "EF", "EC", "ETN", "ETM"]:  # 어미
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
