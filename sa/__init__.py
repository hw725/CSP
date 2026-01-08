"""
SA (Semantic Alignment) 패키지

핵심 기능:
- 원문 공백 단위 분할 (완전한 무결성 보장)
- 번역문 원문 단위 맞춤 정렬
- 메타데이터 생성 (시대 정보 등)

원본 텍스트 무결성 보장:
- 모든 분할은 공백 기준으로만 수행
- 토크나이저는 분석/메타데이터용으로만 사용
- 사용자 입력 구조 완전 보존
"""

from .sa_aligner import (
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
    'split_src_meaning_units',
    'split_tgt_meaning_units',
    'process_single_row', 
    'align_translation_to_source',
    'process_sa_alignment',
    
    # 호환성 별명들
    'split_tgt_meaning_units_sequential',
    'split_tgt_by_src_units',
    'split_tgt_by_src_units_semantic',
]

__version__ = "2.0.0"
__author__ = "SA Development Team"
