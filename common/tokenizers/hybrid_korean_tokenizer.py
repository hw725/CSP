"""
하이브리드 한국어 토크나이저
한국어 번역문의 한자: RoBERTa Korean-Hanja → SikuBERT 순서
한글 파트: Kiwipiepy
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple
import re

logger = logging.getLogger(__name__)

class HybridKoreanTokenizer:
    """하이브리드 한국어 토크나이저 클래스"""

    def __init__(
        self,
        roberta_model: str = "klue/roberta-large",
        kiwi_model: str = "base",
        device: str = "auto",
        fallback_to_siku: bool = True,
    ):
        """
        Args:
            roberta_model: RoBERTa 모델명
            kiwi_model: Kiwi 모델 타입
            device: GPU 디바이스
            fallback_to_siku: RoBERTa 실패시 SikuBERT 사용 여부
        """
        self.roberta_model = roberta_model
        self.kiwi_model = kiwi_model
        self.device = device
        self.fallback_to_siku = fallback_to_siku

        # 토크나이저 인스턴스들
        self.roberta_tokenizer = None
        self.kiwi_tokenizer = None
        self.siku_tokenizer = None

        self._initialized = False

    def _initialize(self):
        """지연 초기화"""
        if self._initialized:
            return

        try:
            # RoBERTa 한자 토크나이저
            import sys
            import os

            current_dir = os.path.dirname(__file__)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            from roberta_hanja_tokenizer import get_roberta_hanja_tokenizer

            self.roberta_tokenizer = get_roberta_hanja_tokenizer(
                model_name=self.roberta_model, device=self.device
            )

            # Kiwi 한글 토크나이저
            from kiwi_tokenizer import get_kiwi_tokenizer

            self.kiwi_tokenizer = get_kiwi_tokenizer(model_type=self.kiwi_model)

            # SikuBERT 폴백 토크나이저
            if self.fallback_to_siku:
                from siku_tokenizer import get_siku_tokenizer

                self.siku_tokenizer = get_siku_tokenizer()

            logger.info("하이브리드 한국어 토크나이저 초기화 완료")
            self._initialized = True

        except Exception as e:
            logger.error(f"하이브리드 한국어 토크나이저 초기화 실패: {e}")
            raise

    def tokenize_korean_text(
        self, text: str, text_type: str = "translation"
    ) -> Dict[str, Any]:
        """
        한국어 텍스트 하이브리드 토큰화

        Args:
            text: 토큰화할 텍스트
            text_type: 텍스트 유형 ("translation" 또는 "original")

        Returns:
            토큰화 결과 딕셔너리
        """
        self._initialize()

        result = {
            "original_text": text,
            "text_type": text_type,
            "segments": [],
            "all_tokens": [],
            "hanja_tokens": [],
            "hangul_tokens": [],
        }

        try:
            # 1단계: RoBERTa로 한자 부분 식별 및 토큰화
            roberta_analysis = self.roberta_tokenizer.tokenize_text_with_hanja_priority(
                text
            )

            for segment in roberta_analysis:
                if segment["type"] == "hanja":
                    # 한자 부분: RoBERTa 우선, 실패시 SikuBERT
                    try:
                        tokens = segment["tokens"]
                        if not tokens and self.fallback_to_siku:
                            # RoBERTa 실패시 SikuBERT 사용
                            tokens = self.siku_tokenizer.tokenize(segment["text"])
                            segment["tokenizer_used"] = "siku_fallback"
                        else:
                            segment["tokenizer_used"] = "roberta_hanja"

                        segment["tokens"] = tokens
                        result["hanja_tokens"].extend(tokens)

                    except Exception as e:
                        logger.warning(f"한자 토큰화 실패, 문자 단위로 분할: {e}")
                        tokens = list(segment["text"])
                        segment["tokens"] = tokens
                        segment["tokenizer_used"] = "char_fallback"
                        result["hanja_tokens"].extend(tokens)

                elif segment["type"] == "hangul":
                    # 한글 부분: Kiwipiepy 사용
                    try:
                        tokens = self.kiwi_tokenizer.morphs(segment["text"])
                        segment["tokens"] = tokens
                        segment["tokenizer_used"] = "kiwi"
                        result["hangul_tokens"].extend(tokens)

                    except Exception as e:
                        logger.warning(f"한글 토큰화 실패, 공백 분할: {e}")
                        tokens = segment["text"].split()
                        segment["tokens"] = tokens
                        segment["tokenizer_used"] = "whitespace_fallback"
                        result["hangul_tokens"].extend(tokens)

                result["segments"].append(segment)
                result["all_tokens"].extend(segment["tokens"])

        except Exception as e:
            logger.error(f"하이브리드 토큰화 실패: {e}")
            # 완전 폴백: 공백 분할
            result["all_tokens"] = text.split()
            result["segments"] = [
                {
                    "text": text,
                    "type": "fallback",
                    "tokens": result["all_tokens"],
                    "tokenizer_used": "whitespace_emergency",
                }
            ]

        return result

    def separate_hanja_hangul(self, text: str) -> Tuple[List[str], List[str]]:
        """
        텍스트에서 한자 부분과 한글 부분을 분리

        Returns:
            Tuple[List[str], List[str]]: (한자_부분들, 한글_부분들)
        """
        self._initialize()

        # 한자 패턴 (유니코드 한자 범위)
        hanja_pattern = re.compile(r"[\u4e00-\u9fff]+")

        hanja_parts = []
        hangul_parts = []

        # 전체 텍스트에서 한자 부분 찾기
        last_end = 0

        for match in hanja_pattern.finditer(text):
            start, end = match.span()

            # 한자 앞의 한글 부분
            if start > last_end:
                hangul_text = text[last_end:start].strip()
                if hangul_text:
                    hangul_parts.append(hangul_text)

            # 한자 부분
            hanja_text = match.group().strip()
            if hanja_text:
                hanja_parts.append(hanja_text)

            last_end = end

        # 마지막 한자 뒤의 한글 부분
        if last_end < len(text):
            hangul_text = text[last_end:].strip()
            if hangul_text:
                hangul_parts.append(hangul_text)

        # 한자가 없는 경우 전체를 한글로 처리
        if not hanja_parts and text.strip():
            hangul_parts.append(text.strip())

        return hanja_parts, hangul_parts

    def tokenize(self, text: str, text_type: str = "translation") -> List[str]:
        """
        텍스트를 토큰화하여 토큰 리스트 반환 (기존 인터페이스 호환)

        Args:
            text: 토큰화할 텍스트
            text_type: 텍스트 유형

        Returns:
            토큰 리스트
        """
        result = self.tokenize_korean_text(text, text_type)
        return result.get("all_tokens", [])

    def extract_particles_and_endings(self, text: str) -> List[Tuple[str, str, int]]:
        """한국어 조사 및 어미 추출 (Kiwi 사용)"""
        self._initialize()

        try:
            return self.kiwi_tokenizer.extract_particles(text)
        except Exception as e:
            logger.error(f"조사/어미 추출 실패: {e}")
            return []

    def get_tokenization_summary(self, text: str) -> Dict[str, Any]:
        """토큰화 요약 정보"""
        result = self.tokenize_korean_text(text)

        summary = {
            "total_tokens": len(result["all_tokens"]),
            "hanja_tokens": len(result["hanja_tokens"]),
            "hangul_tokens": len(result["hangul_tokens"]),
            "segments_count": len(result["segments"]),
            "tokenizers_used": [],
        }

        # 사용된 토크나이저 목록
        used_tokenizers = set()
        for segment in result["segments"]:
            used_tokenizers.add(segment.get("tokenizer_used", "unknown"))

        summary["tokenizers_used"] = list(used_tokenizers)

        return summary

    def split_semantic_units(self, text: str, min_length: int = 2) -> List[str]:
        """의미 단위 분할 (SA용)"""
        self._initialize()

        try:
            # Kiwi로 문장 분할
            sentences = self.kiwi_tokenizer.split_sentences(text)

            # 길이 기준 필터링
            filtered_sentences = []
            for sentence in sentences:
                if len(sentence.strip()) >= min_length:
                    filtered_sentences.append(sentence.strip())

            return filtered_sentences

        except Exception as e:
            logger.error(f"의미 단위 분할 실패: {e}")
            # 폴백: 문장부호 기준 분할
            import re

            return [
                s.strip()
                for s in re.split(r"[.!?。]", text)
                if len(s.strip()) >= min_length
            ]

# 전역 인스턴스
_hybrid_korean_tokenizer = None

def get_hybrid_korean_tokenizer(
    roberta_model: str = "klue/roberta-large",
    kiwi_model: str = "base",
    device: str = "auto",
) -> HybridKoreanTokenizer:
    """하이브리드 한국어 토크나이저 싱글톤 인스턴스 반환"""
    global _hybrid_korean_tokenizer

    if _hybrid_korean_tokenizer is None:
        _hybrid_korean_tokenizer = HybridKoreanTokenizer(
            roberta_model=roberta_model, kiwi_model=kiwi_model, device=device
        )

    return _hybrid_korean_tokenizer

def hybrid_tokenize_korean(text: str, text_type: str = "translation") -> Dict[str, Any]:
    """하이브리드 한국어 토큰화 편의 함수"""
    tokenizer = get_hybrid_korean_tokenizer()
    return tokenizer.tokenize_korean_text(text, text_type)

def hybrid_extract_particles(text: str) -> List[Tuple[str, str, int]]:
    """하이브리드 조사/어미 추출 편의 함수"""
    tokenizer = get_hybrid_korean_tokenizer()
    return tokenizer.extract_particles_and_endings(text)
