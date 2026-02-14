"""S2P (Sentence to Phrase) package — splits sentence-level parallel text into phrase-level 1:1 aligned pairs.

S2P (문장→구 분할) 패키지

핵심 기능:
- BiLSTM + Guided Attention 기반 구 경계 예측
- Viterbi 디코딩을 통한 최적 분할
- BGE-M3 임베딩 기반 유사도 평가
- 100% 텍스트 무결성 보장

원본 텍스트 무결성 보장:
- 모든 분할은 공백 기준으로만 수행
- 토크나이저는 분석/메타데이터용으로만 사용
- 사용자 입력 구조 완전 보존
"""

from .s2p_aligner import (
    # 핵심 분할 함수들
    split_src_meaning_units,
    split_tgt_meaning_units,
    # 처리 함수들
    process_single_row,
    align_translation_to_source,
    process_sa_alignment,
    # 유틸리티
    _distribute_words_evenly,
)

# 호환성을 위한 별명들 (기존 코드 지원)
split_tgt_meaning_units_sequential = split_tgt_meaning_units
split_tgt_by_src_units = split_tgt_meaning_units
split_tgt_by_src_units_semantic = split_tgt_meaning_units

__all__ = [
    # 핵심 함수들
    "split_src_meaning_units",
    "split_tgt_meaning_units",
    "process_single_row",
    "align_translation_to_source",
    "process_sa_alignment",
    # 호환성 별명들
    "split_tgt_meaning_units_sequential",
    "split_tgt_by_src_units",
    "split_tgt_by_src_units_semantic",
]

__version__ = "2.0.0"
__author__ = "SA Development Team"
