"""SA 토크나이저 모듈 패키지 - 최적화 버전"""

from .jieba_mecab import (
    # 🎯 실제로 사용되는 핵심 함수들만
    split_src_meaning_units,              # 원문 공백 분할
    split_tgt_meaning_units_sequential,   # 번역문 의미 분할 
    process_single_row,                   # SA 행 처리 함수
    tokenize_text,                        # 기본 토큰화
    
    # 🔄 호환성 함수들
    split_tgt_by_src_units_semantic,      # 호환성
    split_tgt_meaning_units,              # 호환성 
    split_tgt_by_src_units,               # 호환성
)

__all__ = [
    # 실제 사용 함수들
    'split_src_meaning_units',
    'split_tgt_meaning_units_sequential', 
    'process_single_row',
    'tokenize_text',
    
    # 호환성 함수들
    'split_tgt_by_src_units_semantic',
    'split_tgt_meaning_units',
    'split_tgt_by_src_units',
]