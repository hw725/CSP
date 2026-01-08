"""
SA (Semantic Alignment) 모듈
원문을 공백 단위로 분할하고 번역문을 정렬하는 핵심 기능

완전한 원본 텍스트 무결성 보장:
- 원문: 공백으로만 분할, 토크나이저는 분석용만
- 번역문: 원문 단위에 맞춰 정렬
- 메타데이터: 시대 정보 등 참고용 데이터
"""

import logging
from typing import List, Dict, Optional, Any
import sys
import os
import pandas as pd
import numpy as np

# 공통 토크나이저 모듈 import - 전근대 고전 전용 모델 우선
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.tokenizers import (
    detect_chinese_period,
    # 전근대 고전 전용 모델들
    get_siku_tokenizer,
    get_anchi_tokenizer, 
    siku_get_embeddings,
    siku_similarity,
    anchi_get_embeddings,
    anchi_similarity,
    # 전근대 고전 전용 토크나이저 사용 (교체 완료)
    get_siku_tokenizer,
    get_anchi_tokenizer,
    siku_get_embeddings,
    siku_similarity,
    anchi_get_embeddings,
    anchi_similarity
)

logger = logging.getLogger(__name__)

def split_src_meaning_units(text: str, **kwargs) -> List[str]:
    """
    SA 핵심 기능: 원문을 공백 단위로 분할 (무결성 보장)
    
    Args:
        text: 분할할 원문 텍스트
        **kwargs: 추가 옵션 (호환성용, 분할에는 영향 없음)
    
    Returns:
        공백으로 분할된 의미 단위 리스트 (원본 구조 완전 보존)
    """
    if not text or not text.strip():
        return []
    
    # 🎯 SA 핵심 원칙: 무조건 공백 단위로만 분할
    # 사용자가 입력한 공백 구조를 그대로 보존
    words = text.split()
    
    # 📊 내부 분석 (분할에는 절대 영향 없음, 로깅/메타데이터용만)
    try:
        if logger.isEnabledFor(logging.DEBUG):
            # 전근대 고전 텍스트로 가정 (시대 감지 불필요)
            logger.debug(f"전근대 고전 텍스트 처리 중 (분할에 영향 없음)")
            logger.debug(f"실제 분할: 공백 기준 {len(words)}개 - {words}")
    except Exception as e:
        logger.warning(f"내부 분석 실패 (분할에는 영향 없음): {e}")
    
    return words

def split_tgt_meaning_units(text: str, src_units_count: int, src_units: List[str] = None, use_semantic: bool = True, **kwargs) -> List[str]:
    """
    번역문을 원문 단위에 맞춰 분할 (의미 기반 매칭)
    
    Args:
        text: 번역문 텍스트  
        src_units_count: 원문 단위 수
        src_units: 원문 단위 리스트 (의미 매칭용)
        use_semantic: 의미 기반 매칭 사용 여부
        **kwargs: 분할 옵션
    
    Returns:
        원문 단위 수에 맞춰 분할된 번역문 리스트
    """
    if not text or not text.strip() or src_units_count <= 0:
        return [''] * max(1, src_units_count)
    
    # 의미 기반 분할 시도
    if use_semantic and src_units and len(src_units) == src_units_count:
        try:
            result = _split_tgt_by_src_units_semantic(src_units, text, **kwargs)
            logger.debug(f"의미 기반 분할 성공: {result}")
            return result
        except Exception as e:
            logger.warning(f"의미 기반 분할 실패, 단순 분할로 대체: {e}")
    
    # 폴백: 단순 분할
    return _split_tgt_by_src_units_simple(text, src_units_count)

def _split_tgt_by_src_units_semantic(src_units: List[str], tgt_text: str, min_tokens: int = 1, **kwargs) -> List[str]:
    """원문 단위에 따른 번역문 분할 (시간 표현 감지 개선)"""
    
    tgt_tokens = tgt_text.split()
    N, T = len(src_units), len(tgt_tokens)
    
    if N == 0 or T == 0:
        return [''] * N if N > 0 else []
    
    # 🎯 빠른 처리: 단일 단위면 전체 반환
    if N == 1:
        return [tgt_text]
    
    # 🎯 빠른 처리: 토큰이 단위보다 적으면 1:1 매칭
    if T <= N:
        result = []
        for i in range(N):
            if i < T:
                result.append(tgt_tokens[i])
            else:
                result.append("")
        return result
    
def _split_tgt_by_src_units_semantic(src_units: List[str], tgt_text: str, min_tokens: int = 1, **kwargs) -> List[str]:
    """원문 단위에 따른 번역문 분할 (고어 패턴 감지 개선)"""
    
    tgt_tokens = tgt_text.split()
    N, T = len(src_units), len(tgt_tokens)
    
    if N == 0 or T == 0:
        return [''] * N if N > 0 else []
    
    # 🎯 빠른 처리: 단일 단위면 전체 반환
    if N == 1:
        return [tgt_text]
    
    # 🎯 빠른 처리: 토큰이 단위보다 적으면 1:1 매칭
    if T <= N:
        result = []
        for i in range(N):
            if i < T:
                result.append(tgt_tokens[i])
            else:
                result.append("")
        return result
    
    # 🆕 의미적 시간 표현 감지 (BGE 임베딩 활용)
    try:
        from common.embedders.bge import get_embedding_manager
        from common.korean_particle_matcher import get_archaic_bonus
        
        embedder = get_embedding_manager()
        
        # 시간 관련 원문 단위 찾기 (今 포함)
        time_src_idx = None
        for i, src_unit in enumerate(src_units):
            if '今' in src_unit:
                time_src_idx = i
                break
        
        # 시간 표현이 있는 경우 의미적 매칭 시도
        if time_src_idx is not None:
            time_src_unit = src_units[time_src_idx]
            
            # 번역문의 각 토큰과 시간 원문의 유사도 계산
            src_emb = embedder.compute_embeddings_with_cache([time_src_unit], batch_size=1)[0]
            token_embs = embedder.compute_embeddings_with_cache(tgt_tokens, batch_size=8)
            
            best_time_token_idx = None
            best_similarity = 0.0
            
            for i, token_emb in enumerate(token_embs):
                similarity = np.dot(src_emb, token_emb) / (
                    np.linalg.norm(src_emb) * np.linalg.norm(token_emb) + 1e-8
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_time_token_idx = i
            
            # 유사도가 충분히 높으면 우선 배치
            if best_time_token_idx is not None and best_similarity > 0.3:
                result = [''] * N
                result[time_src_idx] = tgt_tokens[best_time_token_idx]
                
                # 나머지 토큰들 분배
                remaining_tokens = tgt_tokens[:best_time_token_idx] + tgt_tokens[best_time_token_idx+1:]
                remaining_units = [i for i in range(N) if i != time_src_idx]
                
                if remaining_tokens and remaining_units:
                    avg_len = len(remaining_tokens) // len(remaining_units)
                    remainder = len(remaining_tokens) % len(remaining_units)
                    
                    start_idx = 0
                    for i, unit_idx in enumerate(remaining_units):
                        tokens_for_this_unit = avg_len
                        if i < remainder:
                            tokens_for_this_unit += 1
                        
                        end_idx = min(start_idx + tokens_for_this_unit, len(remaining_tokens))
                        
                        if start_idx < len(remaining_tokens):
                            unit_text = " ".join(remaining_tokens[start_idx:end_idx])
                            
                            # 🆕 고어 패턴 보너스 적용
                            archaic_bonus = get_archaic_bonus(unit_text, mode='SA')
                            if archaic_bonus > 0.05:
                                logger.debug(f"고어 패턴 감지: {unit_text} (보너스: {archaic_bonus})")
                            
                            result[unit_idx] = unit_text
                        
                        start_idx = end_idx
                
                return result
                
    except Exception as e:
        logger.warning(f"의미적 시간 표현 매칭 실패: {e}")
    
    # ⚡ 기본 모드: 평균 길이 기반 분할 (고어 패턴 보정 적용)
    try:
        from common.korean_particle_matcher import get_archaic_bonus
        
        avg_len = T // N
        remainder = T % N
        
        result = []
        start_idx = 0
        
        for i in range(N):
            # 각 원문 단위당 할당할 토큰 수
            tokens_for_this_unit = avg_len
            if i < remainder:  # 나머지를 앞쪽 단위들에 분배
                tokens_for_this_unit += 1
            
            end_idx = min(start_idx + tokens_for_this_unit, T)
            
            if start_idx < T:
                unit_text = " ".join(tgt_tokens[start_idx:end_idx])
                
                # 🆕 고어 패턴 보정 적용
                archaic_bonus = get_archaic_bonus(unit_text, mode='SA')
                if archaic_bonus > 0.05:
                    logger.debug(f"기본 분할에서 고어 패턴 감지: {unit_text} (보너스: {archaic_bonus})")
                
                result.append(unit_text)
            else:
                result.append("")
            
            start_idx = end_idx
        
        return result
        start_idx = 0
        
        for i in range(N):
            # 각 원문 단위당 할당할 토큰 수
            tokens_for_this_unit = avg_len
            if i < remainder:  # 나머지를 앞쪽 단위들에 분배
                tokens_for_this_unit += 1
            
            end_idx = min(start_idx + tokens_for_this_unit, T)
            
            if start_idx < T:
                unit_text = " ".join(tgt_tokens[start_idx:end_idx])
                result.append(unit_text)
            else:
                result.append("")
            
            start_idx = end_idx
        
        return result
        
    except Exception as e:
        # 폴백: 동일 길이로 분할
        chunk_size = max(1, T // N)
        result = []
        for i in range(N):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < N - 1 else T
            if start < T:
                result.append(" ".join(tgt_tokens[start:end]))
            else:
                result.append("")

def _split_tgt_by_src_units_simple(text: str, target_count: int) -> List[str]:
    """번역문 단순 분할 (폴백용)"""
    trans_words = text.split()
    
    if len(trans_words) == target_count:
        return trans_words
    elif len(trans_words) < target_count:
        return trans_words + [''] * (target_count - len(trans_words))
    else:
        return _distribute_words_evenly(trans_words, target_count)

def process_single_row(row_data: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
    """
    단일 행 처리: 원문 분할 + 번역문 정렬 → 여러 행으로 분할
    
    Args:
        row_data: {'원문': '원문', '번역문': '번역문', '문장식별자': 'id', ...}
        **kwargs: 처리 옵션
    
    Returns:
        분할된 행 데이터 리스트 (원문 단위별로 개별 행)
    """
    source_text = row_data.get('원문', '')
    translation_text = row_data.get('번역문', '')
    base_id = row_data.get('문장식별자', '')
    
    # 원문 분할 (공백 기준)
    src_units = split_src_meaning_units(source_text, **kwargs)
    
    # 번역문 정렬 (의미 기반)
    trans_units = split_tgt_meaning_units(
        translation_text, 
        len(src_units), 
        src_units=src_units,  # 원문 단위 전달
        use_semantic=True,    # 의미 기반 매칭 사용
        **kwargs
    )
    
    # 시대 정보 분석
    try:
        period = detect_chinese_period(source_text, use_koichi=False)  # 전근대 고전 가정
    except Exception as e:
        logger.warning(f"시대 감지 실패: {e}")
        period = 'unknown'
    
    # 각 단위별로 개별 행 생성
    result_rows = []
    for i, (src_unit, trans_unit) in enumerate(zip(src_units, trans_units)):
        row = {
            '문장식별자': base_id if base_id else f"row_{i+1}",
            '원문': src_unit,
            '번역문': trans_unit,
            # 원본 데이터의 다른 컬럼들 유지
            **{k: v for k, v in row_data.items() 
               if k not in ['문장식별자', '원문', '번역문']}
        }
        result_rows.append(row)
    
    # 🆕 한글 토씨 매칭으로 SA 결과 보완
    try:
        from common.korean_particle_matcher import enhance_sa_results_with_particles
        result_rows = enhance_sa_results_with_particles(result_rows)
        logger.debug(f"SA 토씨 매칭 보완 완료: {len(result_rows)}개 행")
    except Exception as e:
        logger.warning(f"SA 토씨 매칭 보완 실패 (기존 결과 유지): {e}")
        # 실패해도 기존 result_rows 그대로 사용

    logger.info(f"SA 처리 완료: {len(src_units)}개 단위, 시대: {period}")
    
    return result_rows

def align_translation_to_source(src_units: List[str], translation: str, use_semantic: bool = True, **kwargs) -> List[str]:
    """
    번역문을 원문 단위에 맞춰 정렬 (의미 기반)
    
    Args:
        src_units: 원문 의미 단위들 (공백 분할된)
        translation: 번역문 전체
        use_semantic: 의미 기반 매칭 사용 여부
        **kwargs: 정렬 옵션
    
    Returns:
        원문 단위 수에 맞춰 분할된 번역문 리스트
    """
    if not src_units or not translation:
        return []
    
    # 의미 기반 분할 사용
    return split_tgt_meaning_units(
        translation, 
        len(src_units), 
        src_units=src_units,
        use_semantic=use_semantic,
        **kwargs
    )
    
    logger.debug(f"번역문 정렬: {len(src_units)}개 원문 → {len(aligned)}개 번역문")
    
    return aligned

def _distribute_words_evenly(words: List[str], target_count: int) -> List[str]:
    """단어들을 목표 개수에 맞춰 균등 분배"""
    if not words or target_count <= 0:
        return []
    
    if target_count == 1:
        return [' '.join(words)]
    
    # 단어를 그룹으로 나누기
    words_per_group = len(words) / target_count
    result = []
    
    for i in range(target_count):
        start_idx = int(i * words_per_group)
        end_idx = int((i + 1) * words_per_group) if i < target_count - 1 else len(words)
        
        group_words = words[start_idx:end_idx]
        result.append(' '.join(group_words) if group_words else '')
    
    return result

def process_sa_alignment(src_text: str, translation: str, **kwargs) -> Dict[str, List[str]]:
    """
    SA 통합 처리: 원문 분할 + 번역문 정렬
    
    Args:
        src_text: 원문 텍스트
        translation: 번역문 텍스트
        **kwargs: 처리 옵션
    
    Returns:
        {
            'source_units': [...],    # 분할된 원문
            'translation_units': [...], # 정렬된 번역문
            'metadata': {...}         # 메타데이터
        }
    """
    # 원문 분할 (공백 단위)
    src_units = split_src_meaning_units(src_text, **kwargs)
    
    # 번역문 정렬
    trans_units = align_translation_to_source(src_units, translation, **kwargs)
    
    # 메타데이터
    metadata = {
        'source_count': len(src_units),
        'translation_count': len(trans_units),
        'alignment_method': 'space_based'
    }
    
    # 시대 정보 추가 (분석용)
    try:
        period = detect_chinese_period(src_text, use_koichi=False)  # 전근대 고전 가정
        metadata['detected_period'] = period
    except Exception as e:
        logger.warning(f"시대 감지 실패: {e}")
        metadata['detected_period'] = 'unknown'
    
    result = {
        'source_units': src_units,
        'translation_units': trans_units,
        'metadata': metadata
    }
    
    logger.info(f"SA 처리 완료: {metadata['source_count']}개 단위, 시대: {metadata['detected_period']}")
    
    return result
